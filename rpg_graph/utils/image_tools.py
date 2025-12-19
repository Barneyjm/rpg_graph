"""Image generation tools for Santa's Workshop Adventure."""

import os
import base64
from typing import Annotated

import requests
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command


@tool
def generate_scene_image(
    scene_description: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = ""
) -> Command:
    """Generate an image to illustrate the current scene.

    Args:
        scene_description: A SHORT description of the SCENE ONLY (max 50 words).
            Focus on: location, weather, lighting, action happening.
            DO NOT describe the gingerbread character - that's automatic.
            DO NOT include style words - that's automatic.
            Example: "a cozy toy workshop with wooden shelves, warm candlelight,
            tiny hammers and half-built toys scattered on workbenches"
    """
    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        return Command(
            update={
                "messages": [{
                    "role": "tool",
                    "content": "Error: FIREWORKS_API_KEY not set",
                    "tool_call_id": tool_call_id
                }]
            }
        )

    # Consistent Gingerbread Scout character description for visual continuity
    hero_description = "an adorable living gingerbread cookie character with white royal icing details, colorful gumdrop buttons, a friendly frosted smile, a tiny red licorice scarf, and a small canvas satchel slung over one cookie shoulder"

    # Truncate scene description to prevent prompt length errors
    # FLUX schnell has token limits - keep scene_description under ~100 words
    max_scene_chars = 400
    if len(scene_description) > max_scene_chars:
        scene_description = scene_description[:max_scene_chars].rsplit(' ', 1)[0] + "..."

    # Add style context for consistent holiday visuals - Grand master style with strong anti-text
    styled_prompt = f"Grand master style oil painting, nostalgic American illustration, warm saturated colors, soft lighting, {hero_description}, {scene_description}, cozy Christmas scene, snowy winter wonderland, detailed realistic figures, sentimental heartwarming mood, vintage 1950s holiday aesthetic, masterful brushwork, no text no words no letters no writing no signs no labels no captions no watermarks"

    url = "https://api.fireworks.ai/inference/v1/workflows/accounts/fireworks/models/flux-1-schnell-fp8/text_to_image"
    headers = {
        "Content-Type": "application/json",
        "Accept": "image/jpeg",
        "Authorization": f"Bearer {api_key}",
    }
    data = {
        "prompt": styled_prompt,
        "aspect_ratio": "16:9",
        "guidance_scale": 3.5,
        "num_inference_steps": 4,
        "seed": -1
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)

        if response.status_code == 200:
            # Store image in state (not in messages to avoid context overflow)
            image_b64 = base64.b64encode(response.content).decode("utf-8")
            return Command(
                update={
                    # Store image separately in state for UI to display
                    "latest_image": f"data:image/jpeg;base64,{image_b64}",
                    # Send short confirmation to LLM (not the image itself)
                    "messages": [{
                        "role": "tool",
                        "content": "Scene image generated successfully.",
                        "tool_call_id": tool_call_id
                    }]
                }
            )
        else:
            # Include response text for debugging
            error_detail = response.text[:200] if response.text else "No details"
            return Command(
                update={
                    "messages": [{
                        "role": "tool",
                        "content": f"Image generation failed: {response.status_code} - {error_detail}",
                        "tool_call_id": tool_call_id
                    }]
                }
            )
    except requests.RequestException as e:
        return Command(
            update={
                "messages": [{
                    "role": "tool",
                    "content": f"Image generation error: {str(e)}",
                    "tool_call_id": tool_call_id
                }]
            }
        )
