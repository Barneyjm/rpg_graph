from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_fireworks import ChatFireworks

from rpg_graph.utils.state import GameState
from rpg_graph.utils.tools import tools

# LLM setup
llm = ChatFireworks(model="accounts/fireworks/models/gpt-oss-20b")

# Summarization middleware to manage context length
GAME_SUMMARY_PROMPT = """You are summarizing the conversation history for a solo RPG called Vivarium.

Preserve these critical narrative elements in your summary:
- Key story events and discoveries (regions found, encounters, outcomes)
- Important NPCs or creatures encountered
- Relics found and their significance
- Memorable moments (both successes and failures)
- Current narrative threads or unresolved situations
- The player's established character traits or decisions

DO NOT include in the summary (these are tracked separately in game state):
- Fatigue level
- Number of vivariums found
- List of discovered regions
- Current region

Write the summary as a narrative recap that captures the adventure's story so far."""

summarization_middleware = SummarizationMiddleware(
    model=llm,
    trigger=("tokens", 8000),
    keep=("messages", 10),
    summary_prompt=GAME_SUMMARY_PROMPT,
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

IMAGE GENERATION:
After narrating dramatic or significant moments, use generate_scene_image to create a visual illustration. Good moments for images include:
- Discovering a new region (show the landscape)
- Finding a Vivarium (show the glowing seed)
- Encountering Automaniacs (show the threatening machines)
- Dramatic action outcomes (especially DARKNESS setbacks)
- Atmospheric moments during rest or exploration
Describe the scene vividly for the image prompt, focusing on visual elements like lighting, terrain, and mood.

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


# Create the agent graph
graph = create_agent(
    llm,
    tools=tools,
    system_prompt=system_prompt,
    state_schema=GameState,
    middleware=[summarization_middleware],
)
