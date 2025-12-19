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
    """Generate an image to illustrate the current scene. Call this after narrating
    a significant moment - discovering a region, finding a gift, encountering
    danger, or any dramatic scene worth visualizing.

    Args:
        scene_description: A vivid description of the scene to illustrate.
            Should be atmospheric and visual, describing the environment,
            lighting, and any key elements. Keep it under 200 words.
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

    # Add style context for consistent holiday visuals - Grand master style with strong anti-text
    styled_prompt = f"Grand master style oil painting, nostalgic American illustration, warm saturated colors, soft lighting, {hero_description}, {scene_description}, cozy Christmas scene, snowy winter wonderland, detailed realistic figures, sentimental heartwarming mood, vintage 1950s holiday aesthetic, masterful brushwork, no text no words no letters no writing no signs no labels no captions no watermarks"

    url = "https://api.fireworks.ai/inference/v1/workflows/accounts/fireworks/models/flux-1-schnell-fp8/text_to_image"
    headers = {
        "Content-Type": "application/json",
        "Accept": "image/png",
        "Authorization": f"Bearer {api_key}",
    }
    data = {
        "prompt": styled_prompt,
        "aspect_ratio": "16:9",
        "guidance_scale": 3.5,
        "num_inference_steps": 1,
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
                    "latest_image": f"data:image/png;base64,{image_b64}",
                    # Send short confirmation to LLM (not the image itself)
                    "messages": [{
                        "role": "tool",
                        "content": "Scene image generated successfully.",
                        "tool_call_id": tool_call_id
                    }]
                }
            )
        else:
            return Command(
                update={
                    "messages": [{
                        "role": "tool",
                        "content": f"Image generation failed: {response.status_code}",
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
