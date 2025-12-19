"""Shared helper functions for the RPG game tools."""

from langgraph.types import interrupt

# Maximum sleepiness before the elf falls asleep
MAX_SLEEPINESS = 5

# Card value to name mapping
CARD_NAMES = {1: "Ace", 11: "Jack", 12: "Queen", 13: "King"}


def normalize_response(response: dict) -> dict:
    """Normalize frontend response to use snake_case keys.

    Supports both camelCase (JS convention) and snake_case (Python convention).
    """
    key_mapping = {
        "regionActionTaken": "region_action_taken",
        "currentLocation": "current_location",
        "turnsRemaining": "turns_remaining",
        "giftsFound": "gifts_found",
        "diceValues": "dice_values",
        "diceTotal": "dice_total",
        "inventoryCapacity": "inventory_capacity",
    }

    normalized = dict(response)
    for camel, snake in key_mapping.items():
        if camel in response and snake not in response:
            normalized[snake] = response[camel]

    return normalized


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
    return normalize_response(response)


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
    return normalize_response(response)


def request_inventory_state() -> dict:
    """Request current inventory state from frontend via interrupt.

    Returns dict with inventory list and capacity.
    """
    response = interrupt({
        "type": "game_input",
        "request": "inventory_state",
        "reason": "Checking satchel contents..."
    })
    return normalize_response(response)


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
        return """🎄✨ **A PERFECT CHRISTMAS!** ✨🎄

*The first rays of Christmas morning light up the sky...*

You did it, little Gingerbread Scout! ALL SIX Magic Gifts have been found and delivered!
Children around the world wake up to the most magical Christmas ever!

Santa swoops down in his sleigh, his belly shaking with joyful laughter.
"Ho ho ho! You've saved Christmas, my brave little cookie! You're the sweetest hero of all!"

Mrs. Claus gives you a place of honor on the mantelpiece, where you can watch
the joy you brought to the world. 🍪

🎁 **CONGRATULATIONS!** 🎁
You are a TRUE Christmas Hero!"""
    elif gifts_found >= 4:
        return f"""🎄 **A Wonderful Christmas** 🎄

*Dawn breaks over the North Pole, painting the snow in shades of rose and gold...*

You found {gifts_found} of the 6 Magic Gifts! The children will have a beautiful Christmas morning.
The elves in the backup workshop worked through the night with songs and laughter
to make sure every child has something special under the tree.

Santa lifts you gently in his warm hands. "You braved the blizzard and gave it your all!
That courage and kindness - THAT is the true magic of Christmas, little cookie."

🎁 Merry Christmas, brave Gingerbread Scout! 🍪🎁"""
    elif gifts_found >= 2:
        return f"""⛄ **A Cozy Christmas** ⛄

*Christmas morning arrives with a gentle snowfall...*

You found {gifts_found} Magic Gifts! It wasn't easy out there in that blizzard,
especially for a little cookie made of gingerbread and dreams.
But here's a secret: Christmas isn't really about the presents at all.

Families gather around warm fireplaces. Children laugh and play together.
Hot cocoa is shared, carols are sung, and love fills every home.

Santa sits beside you by the fire. "The greatest gift is being together.
You reminded everyone what truly matters. Thank you, little one."

🎁 The spirit of Christmas shines bright! 🍪🎁"""
    else:
        return f"""❄️ **A Quiet Christmas** ❄️

*The sun rises soft and golden on Christmas morning...*

The Magic Gifts remained hidden in the snow - but you know what?
Christmas came anyway, just as it always does.

Children wake up to find hand-knitted mittens, homemade cookies,
and parents who love them. They play in the snow, build snowmen,
and discover that the best gifts don't come wrapped in boxes.

Santa finds you watching the sunrise. "Every act of love is a gift," he says gently.
"And you - a tiny gingerbread cookie - ventured into the storm to help others.
That makes YOU the sweetest gift of all."

🎁 Christmas is about love - and love is never lost! 🍪🎁"""


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


def format_card_value(value: int) -> str:
    """Format a card value as a readable name."""
    return CARD_NAMES.get(value, str(value))


def get_inventory_weight(inventory: list[dict]) -> int:
    """Calculate total weight of inventory items."""
    return sum(item.get("weight", 0) for item in inventory)
