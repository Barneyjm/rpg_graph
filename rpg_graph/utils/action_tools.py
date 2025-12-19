"""Core game action tools for Santa's Workshop Adventure."""

import random
from typing import Annotated, List

from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command, interrupt

from rpg_graph.utils.helpers import (
    request_dice, request_cards, check_fatigue, check_time_remaining,
    parse_card, normalize_response,
)
from rpg_graph.utils.game_data import REGIONS, GIFT_NAMES


def calculate_item_modifiers(inventory: List[dict], action_type: str) -> tuple[int, List[str], List[dict]]:
    """Calculate modifiers from inventory items for a given action.

    Args:
        inventory: List of item dicts with name, weight, effect
        action_type: The type of action being taken

    Returns:
        Tuple of (total_modifier, list of item names used, updated inventory with consumed items removed)
    """
    modifier = 0
    items_used = []
    updated_inventory = list(inventory)  # Copy to avoid mutating original

    for item in inventory:
        name = item.get("name", "")
        effect = item.get("effect", "")

        # Candy Cane: +1 to any roll (consumed)
        if name == "Candy Cane" and "+1 to next roll" in effect:
            modifier += 1
            items_used.append("Candy Cane (+1)")
            # Remove from updated inventory (find and remove first match)
            for j, inv_item in enumerate(updated_inventory):
                if inv_item.get("name") == "Candy Cane":
                    updated_inventory.pop(j)
                    break
            break  # Only use one candy cane per action

    # Gingerbread Friend: +1 to sneak actions (passive, not consumed)
    for item in inventory:
        name = item.get("name", "")
        if name == "Gingerbread Friend" and action_type == "sneak_past":
            modifier += 1
            items_used.append("Gingerbread Friend (+1 sneak)")
            break  # Only count once

    return modifier, items_used, updated_inventory


@tool
def roll_dice(
    modifier: int = 0,
    tool_call_id: Annotated[str, InjectedToolCallId] = ""
) -> Command:
    """Roll 2d6 and add a modifier. Used for taking actions in the game.
    The player will physically roll the dice!

    Args:
        modifier: A bonus or penalty to add to the roll (default 0)
    """
    response = request_dice(2, "Roll 2d6 for your action!")
    dice = response.get("dice_values", [1, 1])
    dice_total = response.get("dice_total", sum(dice))
    total = dice_total + modifier

    message = f"🎲 Rolled {dice[0]} + {dice[1]} = {dice_total}, with modifier {modifier} = {total}"

    return Command(
        update={
            "last_dice": None,
            "last_card": None,
            "messages": [{"role": "tool", "content": message, "tool_call_id": tool_call_id}]
        }
    )


@tool
def draw_cards(
    count: int = 2,
    tool_call_id: Annotated[str, InjectedToolCallId] = ""
) -> Command:
    """Draw cards from the adventure deck to determine outcomes.
    The player will physically draw the cards!

    Args:
        count: Number of cards to draw (default 2)
    """
    response = request_cards(count, f"Draw {count} card(s) from the deck!")
    card_strings = response.get("cards", ["Ace of hearts"])

    message = f"🃏 Drew {count} card(s): {', '.join(card_strings)}"

    return Command(
        update={
            "last_dice": None,
            "last_card": None,
            "messages": [{"role": "tool", "content": message, "tool_call_id": tool_call_id}]
        }
    )


@tool
def take_action(
    action_type: str,
    modifier: int = 0,
    tool_call_id: Annotated[str, InjectedToolCallId] = ""
) -> Command:
    """Perform a game action with dice roll and card draw to determine the outcome.
    The player rolls dice and draws cards! Automatically tracks sleepiness on SNOWDRIFT results.
    Costs 1 turn on the Christmas Clock.

    Actions: brave_blizzard, search_treats, holiday_memory, explore_location,
             chase_gremlins, sneak_past

    Args:
        action_type: The type of action being taken
        modifier: Modifier to add to the dice roll
    """
    dice_response = request_dice(2, f"Roll 2d6 for: {action_type}")

    # Check fatigue before allowing action
    is_too_sleepy, fatigue_msg, _ = check_fatigue(dice_response)
    if is_too_sleepy:
        return Command(
            update={
                "last_dice": None,
                "last_card": None,
                "messages": [{"role": "tool", "content": fatigue_msg, "tool_call_id": tool_call_id}]
            }
        )

    # Check time remaining (costs 1 turn)
    is_game_over, ending_msg, turns_after, _ = check_time_remaining(dice_response, turn_cost=1)
    if is_game_over:
        return Command(
            update={
                "turns_remaining": 0,
                "last_dice": None,
                "last_card": None,
                "messages": [{"role": "tool", "content": ending_msg, "tool_call_id": tool_call_id}]
            }
        )

    # Calculate item modifiers from inventory
    inventory = dice_response.get("inventory", [])
    item_modifier, items_used, updated_inventory = calculate_item_modifiers(inventory, action_type)
    total_modifier = modifier + item_modifier

    dice = dice_response.get("dice_values", [1, 1])
    dice_total = dice_response.get("dice_total", sum(dice)) + total_modifier

    # Request cards from user
    cards_response = request_cards(2, "Draw 2 cards to challenge your roll!")
    card_strings = cards_response.get("cards", ["Ace of hearts", "Ace of hearts"])

    # Parse card strings to get values
    cards = [parse_card(c) for c in card_strings]

    # Determine result
    is_sparkle = dice_total > cards[0]["value"] and dice_total > cards[1]["value"]
    is_flurry = dice_total > cards[0]["value"] or dice_total > cards[1]["value"]

    # Action-specific consequences
    sleepiness_gain = 0
    extra_effect = ""

    if action_type == "sneak_past":
        if is_sparkle:
            result = "✨ SPARKLE (Total Success!)"
            extra_effect = "🤫 Perfect stealth!"
        elif is_flurry:
            result = "❄️ FLURRY (Partial Success)"
        else:
            result = "🌨️ SNOWDRIFT (Setback)"
            sleepiness_gain = 2  # Caught! Extra tired from running
            extra_effect = "😱 You were spotted! Extra sleepiness from fleeing!"
    elif action_type == "chase_gremlins":
        if is_sparkle:
            result = "✨ SPARKLE (Total Success!)"
            extra_effect = "🎉 Caught the gremlin! It drops something shiny..."
        elif is_flurry:
            result = "❄️ FLURRY (Partial Success)"
            extra_effect = "🏃 The gremlin slips away but drops a clue!"
        else:
            result = "🌨️ SNOWDRIFT (Setback)"
            sleepiness_gain = 1
            extra_effect = "😤 The gremlin escapes, laughing mischievously!"
    else:
        # Standard actions: brave_blizzard, search_treats, holiday_memory
        if is_sparkle:
            result = "✨ SPARKLE (Total Success!)"
        elif is_flurry:
            result = "❄️ FLURRY (Partial Success)"
        else:
            result = "🌨️ SNOWDRIFT (Setback)"
            sleepiness_gain = 1

    cards_str = ", ".join(card_strings)

    # Build message
    modifier_str = f" + {total_modifier}" if total_modifier else ""
    message = f"""🎲 Action: {action_type}
Dice: {dice[0]} + {dice[1]}{modifier_str} = {dice_total}
Cards: {cards_str}
Result: {result}"""

    if items_used:
        message += f"\n🎁 Items used: {', '.join(items_used)}"

    if extra_effect:
        message += f"\n{extra_effect}"

    if sleepiness_gain > 0:
        message += f"\n😴 +{sleepiness_gain} Sleepiness (too many cookies!)"

    message += f"\n⏰ Turns remaining: {turns_after}"

    # Build update dict
    update = {
        "sleepiness": sleepiness_gain,
        "turns_remaining": turns_after,
        "last_dice": None,
        "last_card": None,
        "region_action_taken": True,
        "messages": [{"role": "tool", "content": message, "tool_call_id": tool_call_id}]
    }

    # Update inventory if items were consumed
    if items_used:
        update["inventory"] = updated_inventory

    return Command(update=update)


@tool
def discover_new_region(
    tool_call_id: Annotated[str, InjectedToolCallId] = ""
) -> Command:
    """Explore a new magical location by rolling 2d6 to determine where you end up.
    The player rolls the dice! Automatically adds the location to explored places.
    Costs 1 turn on the Christmas Clock."""
    dice_response = request_dice(2, "Roll 2d6 to discover a new location!")

    # Check fatigue before allowing action
    is_too_sleepy, fatigue_msg, _ = check_fatigue(dice_response)
    if is_too_sleepy:
        return Command(
            update={
                "last_dice": None,
                "last_card": None,
                "messages": [{"role": "tool", "content": fatigue_msg, "tool_call_id": tool_call_id}]
            }
        )

    # Check time remaining (costs 1 turn)
    is_game_over, ending_msg, turns_after, _ = check_time_remaining(dice_response, turn_cost=1)
    if is_game_over:
        return Command(
            update={
                "turns_remaining": 0,
                "last_dice": None,
                "last_card": None,
                "messages": [{"role": "tool", "content": ending_msg, "tool_call_id": tool_call_id}]
            }
        )

    dice = dice_response.get("dice_values", [1, 1])

    key = f"{dice[0]}{dice[1]}"
    location = REGIONS.get(key, "Mysterious Snowy Place")

    message = f"🗺️ Rolled {dice[0]}, {dice[1]} - You found: **{location}**! ✨\n💡 Explore or take an action here before searching for a Magic Gift!\n⏰ Turns remaining: {turns_after}"

    return Command(
        update={
            "current_location": location,
            "discovered_regions": [location],
            "turns_remaining": turns_after,
            "last_dice": None,
            "last_card": None,
            "region_action_taken": False,
            "messages": [{"role": "tool", "content": message, "tool_call_id": tool_call_id}]
        }
    )


@tool
def check_for_gift(
    tool_call_id: Annotated[str, InjectedToolCallId] = ""
) -> Command:
    """Search the current location for a Magic Gift!
    The player draws a card! A Magic Gift is found if a face card (Jack=11, Queen=12, King=13) is drawn.
    IMPORTANT: You must take an action in the region first before searching for a gift!
    Costs 1 turn on the Christmas Clock."""
    cards_response = request_cards(1, "Draw a card to search for a Magic Gift!")

    # Check fatigue before allowing action
    is_too_sleepy, fatigue_msg, _ = check_fatigue(cards_response)
    if is_too_sleepy:
        return Command(
            update={
                "last_dice": None,
                "last_card": None,
                "messages": [{"role": "tool", "content": fatigue_msg, "tool_call_id": tool_call_id}]
            }
        )

    # Check time remaining (costs 1 turn)
    is_game_over, ending_msg, turns_after, _ = check_time_remaining(cards_response, turn_cost=1)
    if is_game_over:
        return Command(
            update={
                "turns_remaining": 0,
                "last_dice": None,
                "last_card": None,
                "messages": [{"role": "tool", "content": ending_msg, "tool_call_id": tool_call_id}]
            }
        )

    # Check if player has taken an action in this region
    region_action_taken = cards_response.get("region_action_taken", False)
    if not region_action_taken:
        location = cards_response.get("current_location", "this location")
        message = f"🚫 You need to explore or take an action in **{location}** before you can search for a Magic Gift!\nTry braving the blizzard, searching for items, or interacting with the environment first."
        return Command(
            update={
                "last_dice": None,
                "last_card": None,
                "messages": [{"role": "tool", "content": message, "tool_call_id": tool_call_id}]
            }
        )

    card_strings = cards_response.get("cards", ["Ace of hearts"])
    card_str = card_strings[0] if card_strings else "Ace of hearts"
    card = parse_card(card_str)

    found = card["value"] >= 11

    if found:
        gift = random.choice(GIFT_NAMES)
        message = f"🃏 Drew {card_str} - 🎁✨ **MAGIC GIFT FOUND!** ✨🎁\nYou discovered the {gift}! Christmas is one step closer to being saved!\n⏰ Turns remaining: {turns_after}"
        return Command(
            update={
                "gifts_found": 1,
                "turns_remaining": turns_after,
                "last_dice": None,
                "last_card": None,
                "messages": [{"role": "tool", "content": message, "tool_call_id": tool_call_id}]
            }
        )

    message = f"🃏 Drew {card_str} - No Magic Gift here... but keep searching! 🔍\n⏰ Turns remaining: {turns_after}"
    return Command(
        update={
            "turns_remaining": turns_after,
            "last_dice": None,
            "last_card": None,
            "messages": [{"role": "tool", "content": message, "tool_call_id": tool_call_id}]
        }
    )


@tool
def hot_cocoa_break(tool_call_id: Annotated[str, InjectedToolCallId] = "") -> Command:
    """Take a cozy break with hot cocoa to recover from sleepiness! Resets sleepiness to 0.
    WARNING: This costs 2 turns on the Christmas Clock - time passes while you rest!
    Use this when your elf is getting too drowsy from all those cookies!"""
    response = normalize_response(interrupt({
        "type": "game_input",
        "request": "state_check",
        "reason": "Taking a hot cocoa break..."
    }))

    # Check time remaining (costs 2 turns!)
    is_game_over, ending_msg, turns_after, _ = check_time_remaining(response, turn_cost=2)
    if is_game_over:
        return Command(
            update={
                "turns_remaining": 0,
                "messages": [{"role": "tool", "content": ending_msg, "tool_call_id": tool_call_id}]
            }
        )

    message = f"""☕ **Hot Cocoa Break!** ☕

You find a cozy spot by a warm fireplace. Mrs. Claus hands you a steaming mug of hot cocoa with extra marshmallows!

*~sip sip~* 🍫

You feel refreshed and ready for more adventure! ✨ Sleepiness reset to 0!

⏰ But time passes... (-2 turns) Turns remaining: {turns_after}"""
    return Command(
        update={
            "sleepiness": 0,
            "turns_remaining": turns_after,
            "messages": [{"role": "tool", "content": message, "tool_call_id": tool_call_id}]
        },
    )


@tool
def ask_snow_globe(
    question: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = ""
) -> Command:
    """Ask the magical Snow Globe a yes/no question about your adventure.
    Requires a Mini Snow Globe in your inventory to use!
    Does not cost a turn - it's just a quick peek into the swirling snow.

    Args:
        question: A yes/no question to ask the Snow Globe
    """
    cards_response = request_cards(1, f"Draw a card to consult the Snow Globe: '{question}'")

    # Check if player has the Snow Globe
    inventory = cards_response.get("inventory", [])
    has_globe = any(item.get("name") == "Mini Snow Globe" for item in inventory)

    if not has_globe:
        message = """🔮 You reach for a Snow Globe... but you don't have one!

Find a Mini Snow Globe first to peer into its mystical depths."""
        return Command(
            update={
                "last_dice": None,
                "last_card": None,
                "messages": [{"role": "tool", "content": message, "tool_call_id": tool_call_id}]
            }
        )

    card_strings = cards_response.get("cards", ["Ace of hearts"])
    card_str = card_strings[0] if card_strings else "Ace of hearts"
    card = parse_card(card_str)

    value = card["value"]

    # Interpret the answer based on card value
    if value >= 10:  # 10, J, Q, K = YES
        answer = "✨ **YES!** ✨\nThe snow swirls with golden light, forming shapes of joy and success!"
    elif value >= 6:  # 6-9 = MAYBE
        answer = "🌀 **MAYBE...** 🌀\nThe vision is unclear, clouded by swirling snowflakes. The future is uncertain."
    else:  # 1-5 = NO
        answer = "❄️ **NO...** ❄️\nThe snow falls still and cold. The spirits give their answer in silence."

    message = f"""🔮 **The Snow Globe Speaks!** 🔮

You gaze deep into the swirling snow...
*"{question}"*

🃏 Drew {card_str}

{answer}"""

    return Command(
        update={
            "last_dice": None,
            "last_card": None,
            "messages": [{"role": "tool", "content": message, "tool_call_id": tool_call_id}]
        }
    )
