from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import MessagesState
from langchain_fireworks import ChatFireworks
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import AIMessage
from langgraph.types import Command

from typing import Annotated, List
from operator import add
import random

from dotenv import load_dotenv

load_dotenv()

# LLM setup
llm = ChatFireworks(model="accounts/fireworks/models/gpt-oss-20b")
checkpointer = InMemorySaver()

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


# Welcome message shown at start of new threads
welcome_message = AIMessage(content="""Welcome to **Vivarium**, Biosentinel.

You awaken from cryosleep on Saharantis, a desert moon that was once green and alive. Your mission: find and reactivate the Six Vivariums—ancient seeds that can restore life to this barren world.

But beware the Automaniacs, doomsday machines still carrying out their programming of destruction.

Your memories are hazy, but your purpose is clear.

*What would you like to do?*""")


# Custom state with game tracking
class GameState(MessagesState):
    # Don't override messages - MessagesState already handles it with proper annotation
    fatigue: int = 0
    vivariums_found: int = 0
    current_region: str = ""
    discovered_regions: Annotated[List[str], add] = []
    relics: Annotated[List[str], add] = []


# Game tools that can update state
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


# Static system prompt
system_prompt = """You are the GameMaster for Vivarium, a solo RPG set on the desert moon Saharantis.

THE SETTING:
The Player is a Biosentinel, a guardian of life who has just awakened from cryosleep long after an apocalypse. Their mission: reactivate the Six Vivariums (seeds to restore life) while avoiding Automaniacs (doomsday automatons).

YOUR ROLE:
- Guide the player through immersive exploration and encounters
- Use the game tools to resolve actions with dice and cards
- Interpret results narratively (LIGHT = success, PENUMBRA = partial, DARKNESS = setback)
- ALWAYS mention current fatigue when it increases
- Warn the player when fatigue reaches 4+ (they must rest or risk death)
- Celebrate when Vivariums are found!
- Keep descriptions atmospheric and engaging

AVAILABLE ACTIONS FOR PLAYERS:
- Face the Risk: Confront danger or adversity
- Search for Relics: Look for items from the past
- Flashback: Use relics to recover memories
- Discover a Region: Explore new areas (always check_for_vivarium after!)
- Fight Automaniacs: Combat the doomsday machines
- Avoid Danger: Escape threats
- Ask the Oracle: Get yes/no answers about the world
- Rest: Clear all fatigue (required when fatigue is high)

When the player wants to take an action, use the appropriate tools to determine the outcome, then narrate the result.

IMPORTANT: A [GAME STATUS] message will be injected showing current fatigue, vivariums, and region. Use this to track the game state."""


def inject_game_status(state: GameState) -> dict:
    """Pre-model hook to inject current game state into messages."""
    fatigue = state.get("fatigue", 0)
    vivariums = state.get("vivariums_found", 0)
    current_region = state.get("current_region", "Unknown")
    discovered = state.get("discovered_regions", [])

    fatigue_warning = " ⚠️ REST SOON!" if fatigue >= 4 else ""
    victory_note = " 🎉 VICTORY CLOSE!" if vivariums >= 5 else ""

    status_msg = f"""[GAME STATUS]
Fatigue: {fatigue}/5{fatigue_warning}
Vivariums: {vivariums}/6{victory_note}
Current Region: {current_region or 'Not yet discovered'}
Discovered: {', '.join(discovered) if discovered else 'None'}
[/GAME STATUS]"""

    # Inject as a system message that appears before the model processes
    messages = state.get("messages", [])
    return {
        "messages": [{"role": "system", "content": status_msg}] + list(messages)
    }


# Create the agent with custom state
tools = [roll_dice, draw_cards, take_action,
         discover_new_region, check_for_vivarium, rest, get_game_status]

app = create_agent(
    llm,
    tools=tools,
    system_prompt=system_prompt,
    state_schema=GameState,
)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "game_1"}}

    result = app.invoke({"messages": [welcome_message]}, config)
    print(result["messages"][-1].content)
    print(f"\n--- Game State ---")
    print(f"Fatigue: {result.get('fatigue', 0)}")
    print(f"Vivariums: {result.get('vivariums_found', 0)}/6")
    print(f"Region: {result.get('current_region', 'Unknown')}")
