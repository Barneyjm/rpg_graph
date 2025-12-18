import os
import random
import base64
from typing import Annotated, List

import requests
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command, interrupt

# Game data
colors = ['hearts', 'diamonds', 'spades', 'clubs']
deck = [{"value": value, "color": color}
        for value in range(1, 14) for color in colors]

# Card value to name mapping
CARD_NAMES = {1: "Ace", 11: "Jack", 12: "Queen", 13: "King"}


def request_dice(num_dice: int, reason: str) -> dict:
    """Request dice roll from user via interrupt.

    Returns dict with dice_values and dice_total from user input.
    """
    response = interrupt({
        "type": "game_input",
        "request": "dice",
        "num_dice": num_dice,
        "reason": reason
    })
    return response


def request_cards(num_cards: int, reason: str) -> dict:
    """Request card draw from user via interrupt.

    Returns dict with cards list from user input.
    """
    response = interrupt({
        "type": "game_input",
        "request": "card",
        "num_cards": num_cards,
        "reason": reason
    })
    return response


# Maximum sleepiness before the elf falls asleep
MAX_SLEEPINESS = 5


def check_fatigue(response: dict) -> tuple[bool, str, int]:
    """Check if the elf is too sleepy to act.

    Args:
        response: The interrupt response containing sleepiness state

    Returns:
        Tuple of (is_too_sleepy, message, current_sleepiness)
    """
    sleepiness = response.get("sleepiness", 0)
    if sleepiness >= MAX_SLEEPINESS:
        message = f"""😴 **Too Sleepy to Continue!** 😴

*Your eyelids are so heavy... those cookies were SO good...*

You've reached maximum sleepiness ({sleepiness}/{MAX_SLEEPINESS})!
You need a **Hot Cocoa Break** before you can do anything else!

☕ Use the hot_cocoa_break action to recover!"""
        return True, message, sleepiness
    return False, "", sleepiness


def get_ending_message(gifts_found: int) -> str:
    """Get the appropriate ending message based on gifts found."""
    if gifts_found >= 6:
        return """🎄✨ **PERFECT CHRISTMAS!** ✨🎄

*The first rays of Christmas morning light up the sky...*

You did it, little elf! ALL SIX Magic Gifts have been found and delivered!
Children around the world wake up to the most magical Christmas ever!

Santa swoops down in his sleigh, his belly shaking with joyful laughter.
"Ho ho ho! You've saved Christmas, my dear elf! This calls for EXTRA cookies!"

🎁 **CONGRATULATIONS!** 🎁
You are a TRUE Christmas Hero!"""
    elif gifts_found >= 4:
        return f"""🎄 **Bittersweet Christmas** 🎄

*Dawn breaks over the North Pole...*

You found {gifts_found} of the 6 Magic Gifts! Most children will have a wonderful Christmas,
though a few will receive simpler presents from Santa's backup workshop.

Santa pats you on the head warmly. "You did your best, little one.
That's what the holiday spirit is all about!"

🎁 A good effort - but can you do better next time?"""
    elif gifts_found >= 2:
        return f"""⛄ **Troubled Christmas** ⛄

*Christmas morning arrives...*

Only {gifts_found} Magic Gifts were recovered. Santa has to improvise,
calling in emergency toy-making crews to fill the gaps.

It's not the Christmas anyone hoped for, but the spirit of giving prevails.
"Every gift given with love is magical," Santa reminds you gently.

🎁 The North Pole needs you to try again!"""
    else:
        return f"""❄️ **Christmas Crisis** ❄️

*The sun rises on a quiet Christmas morning...*

With only {gifts_found} Magic Gift(s) found, Santa had to make difficult choices.
Children around the world receive hand-written IOUs promising extra-special
gifts next year.

Santa looks tired but hopeful. "Next Christmas, little elf.
We'll make it the best one ever."

🎁 Don't give up! Christmas needs you!"""


def check_time_remaining(response: dict, turn_cost: int = 1) -> tuple[bool, str, int, int]:
    """Check if time has run out (Christmas morning arrived).

    Args:
        response: The interrupt response containing turns_remaining and gifts_found
        turn_cost: How many turns this action costs (default 1)

    Returns:
        Tuple of (is_game_over, message, turns_after_action, gifts_found)
    """
    turns = response.get("turns_remaining", 24)
    gifts_found = response.get("gifts_found", 0)

    # Calculate turns after this action
    turns_after = turns - turn_cost

    if turns_after <= 0:
        message = get_ending_message(gifts_found)
        return True, message, 0, gifts_found

    return False, "", turns_after, gifts_found


def parse_card(card_str: str) -> dict:
    """Parse a card string like '5 of hearts' or 'K of diamonds' into value and color."""
    parts = card_str.lower().split(" of ")
    if len(parts) != 2:
        return {"value": 1, "color": "hearts"}  # fallback

    value_str, color = parts
    # Convert card names to values (support both full names and abbreviations)
    name_to_value = {
        "ace": 1, "a": 1,
        "jack": 11, "j": 11,
        "queen": 12, "q": 12,
        "king": 13, "k": 13,
    }
    if value_str in name_to_value:
        value = name_to_value[value_str]
    else:
        try:
            value = int(value_str)
        except ValueError:
            value = 1

    return {"value": value, "color": color}

regions = {
    "11": "Candy Cane Forest", "12": "Gingerbread Village", "13": "Snowflake Meadow", "14": "Reindeer Stables",
    "15": "The Giant Christmas Tree", "16": "Frozen Toy Workshop", "21": "Sugarplum Hills", "22": "Secret Elf Tunnels",
    "23": "Marshmallow Marsh", "24": "Peppermint Mountains", "25": "Hot Cocoa River", "26": "Frozen Cranberry Lake",
    "31": "Northern Lights Bay", "32": "Gingerbread Island", "33": "Snowy Plains", "34": "Crystal Ice Glacier",
    "35": "Enchanted Skating Pond", "36": "Cookie Dough Desert", "41": "Cozy Blanket Tundra", "42": "Tinsel Caves",
    "43": "Mistletoe Meadow", "44": "Owl's Winter Nest", "45": "Toy Town Square", "46": "Icicle Cliffs",
    "51": "Mrs. Claus's Gardens", "52": "Evergreen Jungle", "53": "Snowdrift Prairie", "54": "The Naughty List Wasteland",
    "55": "Busy Bee Honeycomb Bakery", "56": "Ribbon Canyon", "61": "Gift Wrapping Catacombs", "62": "Warm Hearth Mountain",
    "63": "Frosted Wetlands", "64": "Ancient Ornament Vault", "65": "Snowmelt Stream", "66": "Cozy Woodland Hollow",
}


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
    # Request dice from user via interrupt
    response = request_dice(2, "Roll 2d6 for your action!")
    dice = response.get("dice_values", [1, 1])
    dice_total = response.get("dice_total", sum(dice))
    total = dice_total + modifier

    message = f"🎲 Rolled {dice[0]} + {dice[1]} = {dice_total}, with modifier {modifier} = {total}"

    return Command(
        update={
            "last_dice": None,  # Clear after consuming
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
    # Request cards from user via interrupt
    response = request_cards(count, f"Draw {count} card(s) from the deck!")
    card_strings = response.get("cards", ["Ace of hearts"])

    message = f"🃏 Drew {count} card(s): {', '.join(card_strings)}"

    return Command(
        update={
            "last_dice": None,
            "last_card": None,  # Clear after consuming
            "messages": [{"role": "tool", "content": message, "tool_call_id": tool_call_id}]
        }
    )


def format_card_value(value: int) -> str:
    """Format a card value as a readable name."""
    return CARD_NAMES.get(value, str(value))


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
        effect = item.get("effect", "")
        if name == "Gingerbread Friend" and action_type == "sneak_past":
            modifier += 1
            items_used.append("Gingerbread Friend (+1 sneak)")
            break  # Only count once

    return modifier, items_used, updated_inventory


@tool
def take_action(
    action_type: str,
    modifier: int = 0,
    tool_call_id: Annotated[str, InjectedToolCallId] = ""
) -> Command:
    """Perform a game action with dice roll and card draw to determine the outcome.
<<<<<<< HEAD
    The player rolls dice and draws cards! Automatically tracks sleepiness on SNOWDRIFT results.
    Costs 1 turn on the Christmas Clock.

    Actions: brave_blizzard, search_treats, holiday_memory, explore_location,
             chase_gremlins, sneak_past
=======
    Automatically tracks sleepiness on SNOWDRIFT results.

    Actions: brave_blizzard, search_treats, holiday_memory, explore_location,
             chase_gremlins, sneak_past, ask_snowglobe
>>>>>>> b91ffe8e817aafb942400d6623cfdecb9cbefc89

    Args:
        action_type: The type of action being taken
        modifier: Modifier to add to the dice roll
    """
    # Request dice from user
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
    cards_response = request_cards(2, f"Draw 2 cards to challenge your roll!")
    card_strings = cards_response.get("cards", ["Ace of hearts", "Ace of hearts"])

    # Parse card strings to get values
    cards = [parse_card(c) for c in card_strings]

    # Determine result
<<<<<<< HEAD
    is_sparkle = dice_total > cards[0]["value"] and dice_total > cards[1]["value"]
    is_flurry = dice_total > cards[0]["value"] or dice_total > cards[1]["value"]
    is_snowdrift = not is_flurry

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
=======
    sleepiness_gain = 0
    if dice_total > cards[0]["value"] and dice_total > cards[1]["value"]:
        result = "✨ SPARKLE (Total Success!)"
    elif dice_total > cards[0]["value"] or dice_total > cards[1]["value"]:
        result = "❄️ FLURRY (Partial Success)"
    else:
        result = "🌨️ SNOWDRIFT (Setback)"
        sleepiness_gain = 1
>>>>>>> b91ffe8e817aafb942400d6623cfdecb9cbefc89

    cards_str = ", ".join(card_strings)

<<<<<<< HEAD
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
=======
    message = f"""🎲 Action: {action_type}
Dice: {dice[0]} + {dice[1]} + {modifier} = {dice_total}
Cards: {cards_str}
Result: {result}"""

    if sleepiness_gain > 0:
        message += f"\n😴 +{sleepiness_gain} Sleepiness (too many cookies!)"

    # Return Command to update state
    return Command(
        update={
            "fatigue": sleepiness_gain,  # Will be added to current value (sleepiness!)
            "messages": [{"role": "tool", "content": message, "tool_call_id": tool_call_id}]
        }
    )
>>>>>>> b91ffe8e817aafb942400d6623cfdecb9cbefc89


@tool
def discover_new_region(
    tool_call_id: Annotated[str, InjectedToolCallId] = ""
) -> Command:
    """Explore a new magical location by rolling 2d6 to determine where you end up.
<<<<<<< HEAD
    The player rolls the dice! Automatically adds the location to explored places.
    Costs 1 turn on the Christmas Clock."""
    # Request dice from user
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
    location = regions.get(key, "Mysterious Snowy Place")

    message = f"🗺️ Rolled {dice[0]}, {dice[1]} - You found: **{location}**! ✨\n💡 Explore or take an action here before searching for a Magic Gift!\n⏰ Turns remaining: {turns_after}"

    return Command(
        update={
            "current_location": location,
            "discovered_regions": [location],
            "turns_remaining": turns_after,
            "last_dice": None,
            "last_card": None,
            "region_action_taken": False,
=======
    Automatically adds the location to explored places!"""
    dice = [random.randint(1, 6) for _ in range(2)]
    key = f"{dice[0]}{dice[1]}"
    location = regions.get(key, "Mysterious Snowy Place")

    message = f"🗺️ Rolled {dice[0]}, {dice[1]} - You found: **{location}**! ✨"

    return Command(
        update={
            "current_region": location,
            "discovered_regions": [location],
>>>>>>> b91ffe8e817aafb942400d6623cfdecb9cbefc89
            "messages": [{"role": "tool", "content": message, "tool_call_id": tool_call_id}]
        }
    )


def request_region_state() -> dict:
    """Request current region state from frontend via interrupt.

    Returns dict with region_action_taken and current_location.
    """
    response = interrupt({
        "type": "game_input",
        "request": "region_state",
        "reason": "Checking region status..."
    })
    return response


@tool
def check_for_gift(
    tool_call_id: Annotated[str, InjectedToolCallId] = ""
) -> Command:
<<<<<<< HEAD
    """Search the current location for a Magic Gift!
    The player draws a card! A Magic Gift is found if a face card (Jack=11, Queen=12, King=13) is drawn.
    IMPORTANT: You must take an action in the region first before searching for a gift!
    Costs 1 turn on the Christmas Clock."""
    # Request card from user - also get region state and sleepiness to avoid multiple interrupts
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

    # Check if player has taken an action in this region (from the same response)
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

=======
    """When exploring a new location, check if a Magic Gift is hidden there!
    A Magic Gift is found if a face card (Jack=11, Queen=12, King=13) is drawn.
    Automatically tracks gifts found!"""
    card = random.choice(deck)
>>>>>>> b91ffe8e817aafb942400d6623cfdecb9cbefc89
    found = card["value"] >= 11

    if found:
<<<<<<< HEAD
        gift_names = ["Sparkling Snow Globe", "Golden Jingle Bell", "Magical Toy Train",
                      "Enchanted Nutcracker", "Glowing Star Ornament", "Crystal Candy Cane"]
        gift = random.choice(gift_names)
        message = f"🃏 Drew {card_str} - 🎁✨ **MAGIC GIFT FOUND!** ✨🎁\nYou discovered the {gift}! Christmas is one step closer to being saved!\n⏰ Turns remaining: {turns_after}"
        return Command(
            update={
                "gifts_found": 1,
                "turns_remaining": turns_after,
                "last_dice": None,
                "last_card": None,
=======
        gift_names = ["a Sparkling Snow Globe", "the Golden Jingle Bell", "a Magical Toy Train",
                      "the Enchanted Nutcracker", "a Glowing Star Ornament", "the Crystal Candy Cane"]
        gift = random.choice(gift_names)
        message = f"🃏 Drew {card_name} of {card['color']} - 🎁✨ **MAGIC GIFT FOUND!** ✨🎁\nYou discovered {gift}! Christmas is one step closer to being saved!"
        return Command(
            update={
                "vivariums_found": 1,  # Will be added (tracking gifts found)
>>>>>>> b91ffe8e817aafb942400d6623cfdecb9cbefc89
                "messages": [{"role": "tool", "content": message, "tool_call_id": tool_call_id}]
            }
        )

<<<<<<< HEAD
    message = f"🃏 Drew {card_str} - No Magic Gift here... but keep searching! 🔍\n⏰ Turns remaining: {turns_after}"
=======
    message = f"🃏 Drew {card_name} of {card['color']} - No Magic Gift here... but keep searching! 🔍"
>>>>>>> b91ffe8e817aafb942400d6623cfdecb9cbefc89
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
<<<<<<< HEAD
    WARNING: This costs 2 turns on the Christmas Clock - time passes while you rest!
    Use this when your elf is getting too drowsy from all those cookies!"""
    # Hot cocoa break needs to get current state to check time
    # We use a simple interrupt to get turns_remaining
    response = interrupt({
        "type": "game_input",
        "request": "state_check",
        "reason": "Taking a hot cocoa break..."
    })

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
=======
    Use this when your elf is getting too drowsy from all those cookies!"""
    message = """☕ **Hot Cocoa Break!** ☕
>>>>>>> b91ffe8e817aafb942400d6623cfdecb9cbefc89

You find a cozy spot by a warm fireplace. Mrs. Claus hands you a steaming mug of hot cocoa with extra marshmallows!

*~sip sip~* 🍫

<<<<<<< HEAD
You feel refreshed and ready for more adventure! ✨ Sleepiness reset to 0!

⏰ But time passes... (-2 turns) Turns remaining: {turns_after}"""
=======
You feel refreshed and ready for more adventure! ✨ Sleepiness reset to 0!"""
>>>>>>> b91ffe8e817aafb942400d6623cfdecb9cbefc89
    return Command(
        update={
            "sleepiness": 0,
            "turns_remaining": turns_after,
            "messages": [{"role": "tool", "content": message, "tool_call_id": tool_call_id}]
        },
    )


@tool
def generate_scene_image(
    scene_description: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = ""
) -> Command:
<<<<<<< HEAD
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

    # Add style context for consistent holiday visuals - Norman Rockwell style with strong anti-text
    styled_prompt = f"Norman Rockwell style oil painting, nostalgic American illustration, warm saturated colors, soft lighting, {scene_description}, cozy Christmas scene, snowy winter wonderland, detailed realistic figures, sentimental heartwarming mood, vintage 1950s holiday aesthetic, masterful brushwork, no text no words no letters no writing no signs no labels no captions no watermarks"

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


# Import inventory items from state
from rpg_graph.utils.state import GAME_ITEMS, DEFAULT_INVENTORY_CAPACITY


def get_inventory_weight(inventory: List[dict]) -> int:
    """Calculate total weight of inventory items."""
    return sum(item.get("weight", 0) for item in inventory)


def request_inventory_state() -> dict:
    """Request current inventory state from frontend via interrupt.

    Returns dict with inventory list and capacity.
    """
    response = interrupt({
        "type": "game_input",
        "request": "inventory_state",
        "reason": "Checking satchel contents..."
    })
    return response


@tool
def check_inventory(
    tool_call_id: Annotated[str, InjectedToolCallId] = ""
) -> Command:
    """Check what items are in the elf's satchel and how much space is left.
    Use this to see your inventory before picking up or dropping items."""
    # Request current inventory from frontend
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

    # Request current inventory from frontend
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
    # Request current inventory from frontend
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
    # Request current inventory from frontend
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
        # Items with passive effects aren't consumed
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

    # Handle sleepiness reduction (absolute set for full recovery)
    if sleepiness_change == -99:
        update["sleepiness"] = 0
    elif sleepiness_change < 0:
        # For partial recovery, we need current sleepiness - but we can use additive
        # Note: This adds a negative number, reducing sleepiness
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
    # Request card from user (single interrupt - avoid multiple interrupts per tool call)
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

    # Also get inventory from the same response to avoid a second interrupt
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
    if value >= 11:  # Face card - rare/heavy item
        item_pool = ["hot_cocoa_thermos", "snow_globe", "gingerbread_man", "elf_lantern",
                     "music_box", "sack_of_toys"]
    elif value >= 6:  # Mid card - medium item
        item_pool = ["hot_cocoa_thermos", "snow_globe", "gingerbread_man", "elf_lantern", "toy_hammer"]
    else:  # Low card - common/light item
        item_pool = ["candy_cane", "jingle_bell", "snowflake_cookie", "holly_sprig", "ribbon"]

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
    # Request card and inventory state
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
=======
    """Check how the Christmas rescue mission is going! Shows sleepiness, gifts found, and explored locations.
    IMPORTANT: Always call this at the start of your response to know the current game state."""
    # This returns a Command that doesn't change state but prompts the agent
    # The actual state values will be injected by the pre_model_hook
    return Command(
        update={
            "messages": [{"role": "tool", "content": "🎄 Status retrieved! Check the game state above to see how close we are to saving Christmas! 🎅", "tool_call_id": tool_call_id}]
>>>>>>> b91ffe8e817aafb942400d6623cfdecb9cbefc89
        }
    )


# Export all tools
tools = [roll_dice, draw_cards, take_action,
<<<<<<< HEAD
         discover_new_region, check_for_gift, hot_cocoa_break,
         generate_scene_image, check_inventory, pick_up_item,
         drop_item, use_item, search_for_items, ask_snow_globe]
=======
         discover_new_region, check_for_gift, hot_cocoa_break, get_game_status]
>>>>>>> b91ffe8e817aafb942400d6623cfdecb9cbefc89
