import os
import random
import base64
from typing import Annotated

import requests
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command

# Game data
colors = ['hearts', 'diamonds', 'spades', 'clubs']
deck = [{"value": value, "color": color}
        for value in range(1, 14) for color in colors]

regions = {
    "11": "Planting", "12": "Burrow", "13": "Clearing", "14": "Forest",
    "15": "Tree", "16": "Ruins", "21": "Hills", "22": "Tunnels",
    "23": "Swamp", "24": "Mountains", "25": "River", "26": "Lake",
    "31": "Ocean", "32": "Island", "33": "Plain", "34": "Glacier",
    "35": "Pond", "36": "Desert", "41": "Tundra", "42": "Caves",
    "43": "Meadow", "44": "Nest", "45": "City", "46": "Cliffs",
    "51": "Gardens", "52": "Jungle", "53": "Prairie", "54": "Wasteland",
    "55": "Hive", "56": "Canyon", "61": "Catacombs", "62": "Volcano",
    "63": "Wetlands", "64": "Tomb", "65": "Estuary", "66": "Hollow",
}


@tool
def roll_dice(modifier: int = 0) -> str:
    """Roll 2d6 and add a modifier. Used for taking actions in the game.

    Args:
        modifier: A bonus or penalty to add to the roll (default 0)
    """
    dice = [random.randint(1, 6) for _ in range(2)]
    total = sum(dice) + modifier
    return f"Rolled {dice[0]} + {dice[1]} = {sum(dice)}, with modifier {modifier} = {total}"


@tool
def draw_cards(count: int = 2) -> str:
    """Draw cards from the adventure deck to determine outcomes.

    Args:
        count: Number of cards to draw (default 2)
    """
    cards = [random.choice(deck) for _ in range(count)]
    cards_str = ", ".join([f"{c['value']} of {c['color']}" for c in cards])
    return f"Drew {count} cards: {cards_str}"


@tool
def take_action(
    action_type: str,
    modifier: int = 0,
    tool_call_id: Annotated[str, InjectedToolCallId] = ""
) -> Command:
    """Perform a game action with dice roll and card draw to determine the outcome.
    Automatically tracks fatigue on DARKNESS results.

    Actions: face_risk, search_relics, flashback, discover_region,
             fight_automaniac, avoid_danger, ask_oracle

    Args:
        action_type: The type of action being taken
        modifier: Modifier to add to the dice roll
    """
    dice = [random.randint(1, 6) for _ in range(2)]
    dice_total = sum(dice) + modifier
    cards = [random.choice(deck) for _ in range(2)]

    # Determine result
    fatigue_gain = 0
    if dice_total > cards[0]["value"] and dice_total > cards[1]["value"]:
        result = "LIGHT (Total Success)"
    elif dice_total > cards[0]["value"] or dice_total > cards[1]["value"]:
        result = "PENUMBRA (Partial Success)"
    else:
        result = "DARKNESS (Setback)"
        fatigue_gain = 1

    cards_str = ", ".join([f"{c['value']} of {c['color']}" for c in cards])

    message = f"""Action: {action_type}
Dice: {dice[0]} + {dice[1]} + {modifier} = {dice_total}
Cards: {cards_str}
Result: {result}"""

    if fatigue_gain > 0:
        message += f"\n⚠️ +{fatigue_gain} Fatigue"

    # Return Command to update state
    return Command(
        update={
            "fatigue": fatigue_gain,  # Will be added to current value
            "messages": [{"role": "tool", "content": message, "tool_call_id": tool_call_id}]
        }
    )


@tool
def discover_new_region(
    tool_call_id: Annotated[str, InjectedToolCallId] = ""
) -> Command:
    """Discover a new region by rolling 2d6 to determine the location type.
    Automatically adds the region to discovered_regions."""
    dice = [random.randint(1, 6) for _ in range(2)]
    key = f"{dice[0]}{dice[1]}"
    region = regions.get(key, "Unknown Territory")

    message = f"Rolled {dice[0]}, {dice[1]} - Discovered: {region}"

    return Command(
        update={
            "current_region": region,
            "discovered_regions": [region],
            "messages": [{"role": "tool", "content": message, "tool_call_id": tool_call_id}]
        }
    )


@tool
def check_for_vivarium(
    tool_call_id: Annotated[str, InjectedToolCallId] = ""
) -> Command:
    """When discovering a region, check if a Vivarium is present.
    A Vivarium is found if a face card (Jack=11, Queen=12, King=13) is drawn.
    Automatically tracks vivariums_found."""
    card = random.choice(deck)
    found = card["value"] >= 11
    card_name = {11: "Jack", 12: "Queen", 13: "King"}.get(
        card["value"], str(card["value"]))

    if found:
        message = f"Drew {card_name} of {card['color']} - 🌱 VIVARIUM FOUND! One of the six seeds to restore Saharantis!"
        return Command(
            update={
                "vivariums_found": 1,  # Will be added
                "messages": [{"role": "tool", "content": message, "tool_call_id": tool_call_id}]
            }
        )

    message = f"Drew {card_name} of {card['color']} - No Vivarium here."
    return Command(
        update={
            "messages": [{"role": "tool", "content": message, "tool_call_id": tool_call_id}]
        }
    )


@tool
def rest(tool_call_id: Annotated[str, InjectedToolCallId] = "") -> Command:
    """Rest to recover from fatigue. Resets fatigue to 0.
    Use this when fatigue is high or the adventure deck needs reshuffling."""
    message = "You find a safe place to rest. Your fatigue fades as you recover your strength. ✨ Fatigue reset to 0."
    return Command(
        update={
            "fatigue": 0,
            "messages": [{"role": "tool", "content": message, "tool_call_id": tool_call_id}]
        },
        # Note: fatigue=0 here is absolute, not additive
        # We handle this by using a custom reducer or setting directly
    )


@tool
def get_game_status(
    tool_call_id: Annotated[str, InjectedToolCallId] = ""
) -> Command:
    """Get the current game status. Call this to check fatigue, vivariums found, and regions.
    IMPORTANT: Always call this at the start of your response to know the current game state."""
    # This returns a Command that doesn't change state but prompts the agent
    # The actual state values will be injected by the pre_model_hook
    return Command(
        update={
            "messages": [{"role": "tool", "content": "Status retrieved. Check the game state context above.", "tool_call_id": tool_call_id}]
        }
    )


@tool
def generate_scene_image(
    scene_description: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = ""
) -> Command:
    """Generate an image to illustrate the current scene. Call this after narrating
    a significant moment - discovering a region, finding a vivarium, encountering
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

    # Add style context for consistent visuals
    styled_prompt = f"Sci-fi post-apocalyptic desert moon landscape, alien terrain, {scene_description}, atmospheric lighting, cinematic, detailed environment art, no text, no words"

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


# Export all tools
tools = [roll_dice, draw_cards, take_action,
         discover_new_region, check_for_vivarium, rest, get_game_status,
         generate_scene_image]
