"""Inventory management tools for Santa's Workshop Adventure."""

import random
from typing import Annotated

from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command

from rpg_graph.utils.helpers import (
    request_cards, request_inventory_state, check_fatigue,
    check_time_remaining, parse_card, get_inventory_weight,
)
from rpg_graph.utils.state import GAME_ITEMS, DEFAULT_INVENTORY_CAPACITY
from rpg_graph.utils.game_data import ITEM_POOLS


@tool
def check_inventory(
    tool_call_id: Annotated[str, InjectedToolCallId] = ""
) -> Command:
    """Check what items are in the elf's satchel and how much space is left.
    Use this to see your inventory before picking up or dropping items."""
    state = request_inventory_state()
    inventory = state.get("inventory", [])
    capacity = state.get("inventory_capacity", DEFAULT_INVENTORY_CAPACITY)
    current_weight = get_inventory_weight(inventory)

    if inventory:
        items_list = "\n".join([f"  - {item['name']} (wt:{item['weight']})" for item in inventory])
        message = f"""🎒 **Satchel Contents:**
{items_list}

Weight: {current_weight}/{capacity}"""
    else:
        message = f"🎒 **Satchel is empty!** (Capacity: {capacity} weight)"

    return Command(
        update={
            "messages": [{"role": "tool", "content": message, "tool_call_id": tool_call_id}]
        }
    )


@tool
def pick_up_item(
    item_key: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = ""
) -> Command:
    """Pick up an item and add it to the elf's satchel.
    Will fail if the satchel is too full (over weight capacity).

    Args:
        item_key: The item identifier (e.g., 'candy_cane', 'hot_cocoa_thermos', 'music_box')

    Available items: candy_cane, jingle_bell, snowflake_cookie, holly_sprig, ribbon,
                    hot_cocoa_thermos, snow_globe, gingerbread_man, elf_lantern, toy_hammer,
                    sack_of_toys, frozen_turkey, music_box, giant_candy_cane
    """
    if item_key not in GAME_ITEMS:
        message = f"❌ Unknown item: {item_key}. Check available items!"
        return Command(
            update={
                "messages": [{"role": "tool", "content": message, "tool_call_id": tool_call_id}]
            }
        )

    item = GAME_ITEMS[item_key]

    state = request_inventory_state()
    inventory = state.get("inventory", [])
    capacity = state.get("inventory_capacity", DEFAULT_INVENTORY_CAPACITY)
    current_weight = get_inventory_weight(inventory)

    # Check if item fits
    if current_weight + item.weight > capacity:
        message = f"""❌ **Satchel too full!**
Can't pick up {item.name} (weight: {item.weight})
Current weight: {current_weight}/{capacity}
Drop something first!"""
        return Command(
            update={
                "messages": [{"role": "tool", "content": message, "tool_call_id": tool_call_id}]
            }
        )

    # Add item to inventory
    new_inventory = inventory + [item.to_dict()]
    new_weight = current_weight + item.weight

    message = f"""🎁 **Picked up: {item.name}!**
{item.description}
{f'Effect: {item.effect}' if item.effect else ''}

Satchel: {new_weight}/{capacity} weight"""

    return Command(
        update={
            "inventory": new_inventory,
            "messages": [{"role": "tool", "content": message, "tool_call_id": tool_call_id}]
        }
    )


@tool
def drop_item(
    item_name: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = ""
) -> Command:
    """Drop an item from the elf's satchel to make room for other things.

    Args:
        item_name: The name of the item to drop (e.g., 'Candy Cane', 'Music Box')
    """
    state = request_inventory_state()
    inventory = state.get("inventory", [])

    # Find and remove the item (case-insensitive match)
    new_inventory = []
    found = False
    dropped_item = None
    for item in inventory:
        if not found and item["name"].lower() == item_name.lower():
            found = True
            dropped_item = item
        else:
            new_inventory.append(item)

    if not found:
        message = f"❌ **{item_name}** is not in your satchel!"
        return Command(
            update={
                "messages": [{"role": "tool", "content": message, "tool_call_id": tool_call_id}]
            }
        )

    new_weight = get_inventory_weight(new_inventory)
    capacity = state.get("inventory_capacity", DEFAULT_INVENTORY_CAPACITY)

    message = f"""🗑️ **Dropped: {dropped_item['name']}**
You leave it behind in the snow.

Satchel: {new_weight}/{capacity} weight"""

    return Command(
        update={
            "inventory": new_inventory,
            "messages": [{"role": "tool", "content": message, "tool_call_id": tool_call_id}]
        }
    )


@tool
def use_item(
    item_name: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = ""
) -> Command:
    """Use an item from the elf's satchel. Some items are consumed when used.

    Args:
        item_name: The name of the item to use (e.g., 'Snowflake Cookie', 'Hot Cocoa Thermos')
    """
    state = request_inventory_state()
    inventory = state.get("inventory", [])

    # Find the item (case-insensitive match)
    found_item = None
    item_index = -1
    for i, item in enumerate(inventory):
        if item["name"].lower() == item_name.lower():
            found_item = item
            item_index = i
            break

    if not found_item:
        message = f"❌ **{item_name}** is not in your satchel!"
        return Command(
            update={
                "messages": [{"role": "tool", "content": message, "tool_call_id": tool_call_id}]
            }
        )

    # Determine effect and whether item is consumed
    effect = found_item.get("effect", "")
    sleepiness_change = 0
    consume_item = False

    # Apply effects based on item
    if "sleepiness" in effect.lower() or "cookie" in found_item["name"].lower():
        if "full" in effect.lower() or "thermos" in found_item["name"].lower():
            sleepiness_change = -99  # Will be clamped to 0
            consume_item = True
            effect_msg = "You feel completely refreshed! Sleepiness reset to 0!"
        else:
            sleepiness_change = -1
            consume_item = True
            effect_msg = "You feel a bit more awake! -1 Sleepiness"
    elif effect:
        effect_msg = f"Effect: {effect}"
        consume_item = False
    else:
        effect_msg = "Nothing special happens, but it was fun!"
        consume_item = False

    # Remove item if consumed
    if consume_item:
        new_inventory = inventory[:item_index] + inventory[item_index + 1:]
        consume_msg = f"*The {found_item['name']} is used up*"
    else:
        new_inventory = inventory
        consume_msg = f"*You put the {found_item['name']} back in your satchel*"

    message = f"""✨ **Used: {found_item['name']}!**
{effect_msg}

{consume_msg}"""

    update = {
        "inventory": new_inventory,
        "messages": [{"role": "tool", "content": message, "tool_call_id": tool_call_id}]
    }

    # Handle sleepiness reduction
    if sleepiness_change == -99:
        update["sleepiness"] = 0
    elif sleepiness_change < 0:
        update["sleepiness"] = sleepiness_change

    return Command(update=update)


@tool
def search_for_items(
    tool_call_id: Annotated[str, InjectedToolCallId] = ""
) -> Command:
    """Search the current location for items to pick up.
    Draw a card - face cards (J/Q/K) find rare items, number cards find common items,
    and Aces find nothing. Found items are automatically added to your satchel if there's room.
    Costs 1 turn on the Christmas Clock."""
    cards_response = request_cards(1, "Draw a card to search for items!")

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

    card_strings = cards_response.get("cards", ["Ace of hearts"])
    card_str = card_strings[0] if card_strings else "Ace of hearts"
    card = parse_card(card_str)

    inventory = cards_response.get("inventory", [])
    capacity = cards_response.get("inventory_capacity", DEFAULT_INVENTORY_CAPACITY)
    current_weight = get_inventory_weight(inventory)

    value = card["value"]

    if value == 1:  # Ace - nothing found
        message = f"🃏 Drew {card_str} - You search thoroughly but find nothing useful... 🔍\n⏰ Turns remaining: {turns_after}"
        return Command(
            update={
                "turns_remaining": turns_after,
                "last_dice": None,
                "last_card": None,
                "region_action_taken": True,
                "messages": [{"role": "tool", "content": message, "tool_call_id": tool_call_id}]
            }
        )

    # Determine item rarity based on card value
    if value >= 11:  # Face card - rare item
        item_pool = ITEM_POOLS["rare"]
    elif value >= 6:  # Mid card - medium item
        item_pool = ITEM_POOLS["medium"]
    else:  # Low card - common item
        item_pool = ITEM_POOLS["common"]

    found_key = random.choice(item_pool)
    found_item = GAME_ITEMS[found_key]

    # Try to auto-add to inventory
    if current_weight + found_item.weight <= capacity:
        new_inventory = inventory + [found_item.to_dict()]
        new_weight = current_weight + found_item.weight
        message = f"""🃏 Drew {card_str} - **Found something!** ✨

🎁 **{found_item.name}** (Weight: {found_item.weight})
{found_item.description}
{f'Effect: {found_item.effect}' if found_item.effect else ''}

*Added to satchel!* ({new_weight}/{capacity} weight)
⏰ Turns remaining: {turns_after}"""

        return Command(
            update={
                "turns_remaining": turns_after,
                "last_dice": None,
                "last_card": None,
                "inventory": new_inventory,
                "region_action_taken": True,
                "messages": [{"role": "tool", "content": message, "tool_call_id": tool_call_id}]
            }
        )
    else:
        message = f"""🃏 Drew {card_str} - **Found something!** ✨

🎁 **{found_item.name}** (Weight: {found_item.weight})
{found_item.description}
{f'Effect: {found_item.effect}' if found_item.effect else ''}

⚠️ **Satchel too full!** ({current_weight}/{capacity} weight)
Drop something to pick this up, or leave it behind.
⏰ Turns remaining: {turns_after}"""

        return Command(
            update={
                "turns_remaining": turns_after,
                "last_dice": None,
                "last_card": None,
                "region_action_taken": True,
                "messages": [{"role": "tool", "content": message, "tool_call_id": tool_call_id}]
            }
        )
