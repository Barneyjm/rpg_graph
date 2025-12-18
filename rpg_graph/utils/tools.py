import random
from typing import Annotated

from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command

# Game data
colors = ['hearts', 'diamonds', 'spades', 'clubs']
deck = [{"value": value, "color": color}
        for value in range(1, 14) for color in colors]

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
    Automatically tracks sleepiness on SNOWDRIFT results.

    Actions: brave_blizzard, search_treats, holiday_memory, explore_location,
             chase_gremlins, sneak_past, ask_snowglobe

    Args:
        action_type: The type of action being taken
        modifier: Modifier to add to the dice roll
    """
    dice = [random.randint(1, 6) for _ in range(2)]
    dice_total = sum(dice) + modifier
    cards = [random.choice(deck) for _ in range(2)]

    # Determine result
    sleepiness_gain = 0
    if dice_total > cards[0]["value"] and dice_total > cards[1]["value"]:
        result = "✨ SPARKLE (Total Success!)"
    elif dice_total > cards[0]["value"] or dice_total > cards[1]["value"]:
        result = "❄️ FLURRY (Partial Success)"
    else:
        result = "🌨️ SNOWDRIFT (Setback)"
        sleepiness_gain = 1

    cards_str = ", ".join([f"{c['value']} of {c['color']}" for c in cards])

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


@tool
def discover_new_region(
    tool_call_id: Annotated[str, InjectedToolCallId] = ""
) -> Command:
    """Explore a new magical location by rolling 2d6 to determine where you end up.
    Automatically adds the location to explored places!"""
    dice = [random.randint(1, 6) for _ in range(2)]
    key = f"{dice[0]}{dice[1]}"
    location = regions.get(key, "Mysterious Snowy Place")

    message = f"🗺️ Rolled {dice[0]}, {dice[1]} - You found: **{location}**! ✨"

    return Command(
        update={
            "current_region": location,
            "discovered_regions": [location],
            "messages": [{"role": "tool", "content": message, "tool_call_id": tool_call_id}]
        }
    )


@tool
def check_for_gift(
    tool_call_id: Annotated[str, InjectedToolCallId] = ""
) -> Command:
    """When exploring a new location, check if a Magic Gift is hidden there!
    A Magic Gift is found if a face card (Jack=11, Queen=12, King=13) is drawn.
    Automatically tracks gifts found!"""
    card = random.choice(deck)
    found = card["value"] >= 11
    card_name = {11: "Jack", 12: "Queen", 13: "King"}.get(
        card["value"], str(card["value"]))

    if found:
        gift_names = ["a Sparkling Snow Globe", "the Golden Jingle Bell", "a Magical Toy Train",
                      "the Enchanted Nutcracker", "a Glowing Star Ornament", "the Crystal Candy Cane"]
        gift = random.choice(gift_names)
        message = f"🃏 Drew {card_name} of {card['color']} - 🎁✨ **MAGIC GIFT FOUND!** ✨🎁\nYou discovered {gift}! Christmas is one step closer to being saved!"
        return Command(
            update={
                "vivariums_found": 1,  # Will be added (tracking gifts found)
                "messages": [{"role": "tool", "content": message, "tool_call_id": tool_call_id}]
            }
        )

    message = f"🃏 Drew {card_name} of {card['color']} - No Magic Gift here... but keep searching! 🔍"
    return Command(
        update={
            "messages": [{"role": "tool", "content": message, "tool_call_id": tool_call_id}]
        }
    )


@tool
def hot_cocoa_break(tool_call_id: Annotated[str, InjectedToolCallId] = "") -> Command:
    """Take a cozy break with hot cocoa to recover from sleepiness! Resets sleepiness to 0.
    Use this when your elf is getting too drowsy from all those cookies!"""
    message = """☕ **Hot Cocoa Break!** ☕

You find a cozy spot by a warm fireplace. Mrs. Claus hands you a steaming mug of hot cocoa with extra marshmallows!

*~sip sip~* 🍫

You feel refreshed and ready for more adventure! ✨ Sleepiness reset to 0!"""
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
    """Check how the Christmas rescue mission is going! Shows sleepiness, gifts found, and explored locations.
    IMPORTANT: Always call this at the start of your response to know the current game state."""
    # This returns a Command that doesn't change state but prompts the agent
    # The actual state values will be injected by the pre_model_hook
    return Command(
        update={
            "messages": [{"role": "tool", "content": "🎄 Status retrieved! Check the game state above to see how close we are to saving Christmas! 🎅", "tool_call_id": tool_call_id}]
        }
    )


# Export all tools
tools = [roll_dice, draw_cards, take_action,
         discover_new_region, check_for_gift, hot_cocoa_break, get_game_status]
