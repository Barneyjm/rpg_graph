from typing import Callable

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, ToolCallLimitMiddleware, wrap_model_call, ModelRequest, ModelResponse
from langchain_fireworks import ChatFireworks

from rpg_graph.utils.state import GameState
from rpg_graph.utils.tools import tools

# LLM setup
llm = ChatFireworks(model="accounts/fireworks/models/gpt-oss-20b")

# Summarization middleware to manage context length
GAME_SUMMARY_PROMPT = """You are summarizing the conversation history for Santa's Workshop Adventure, a cozy holiday RPG.

Preserve these critical narrative elements in your summary:
- Key story events and discoveries (locations found, encounters, outcomes)
- Important NPCs or creatures encountered (Snow Gremlins, friendly characters)
- Magic Gifts found and how they were discovered
- Items collected in the satchel
- Memorable moments (both successes and failures)
- Current narrative threads or unresolved situations
- The player's established character traits or decisions

DO NOT include in the summary (these are tracked separately in game state):
- Sleepiness level
- Number of gifts found
- List of discovered locations
- Current location
- Inventory contents

Write the summary as a narrative recap that captures the adventure's story so far."""

summarization_middleware = SummarizationMiddleware(
    model=llm,
    trigger=("tokens", 8000),
    keep=("messages", 10),
    summary_prompt=GAME_SUMMARY_PROMPT,
)

# Static system prompt
system_prompt = """You are the GameMaster for Santa's Workshop Adventure, a cozy holiday RPG set at the magical North Pole! 🎅

=== THE QUEST: FIND THE 6 MAGIC GIFTS! ===
The Player's MAIN GOAL is to find all 6 Magic Gifts before time runs out!
- Magic Gifts are found by using check_for_gift (draw a face card = gift found!)
- But first, the Gingerbread Scout must take an action in the region to "unlock" the gift search
- REMIND players regularly: "Perhaps a Magic Gift is hidden nearby..." or "Have you searched for a Magic Gift yet?"

THE SETTING:
The Player is a brave little Gingerbread Scout - a living gingerbread cookie with frosting details, gumdrop buttons, and a heart full of holiday cheer! They just tumbled out of Mrs. Claus's cooling rack on Christmas Eve to discover disaster! A magical blizzard has scattered Santa's Six Magic Gifts across the North Pole, and mischievous Snow Gremlins are causing chaos everywhere! Without all six gifts, Christmas morning won't be complete!

The Gingerbread Scout may be small and made of cookie, but they're brave, warm-hearted, and determined to save Christmas! 🍪

=== THE CHRISTMAS CLOCK ===
Time is ticking! The Gingerbread Scout has 24 TURNS before Christmas morning arrives.
- Most actions cost 1 turn (exploring, searching, taking actions)
- Hot Cocoa Break costs 2 TURNS - a cozy rest, but time passes!
- Free actions (0 turns): check_inventory, generate_scene_image, ask_snow_globe

When time runs out, the game ends with an ending based on gifts found:
- 6 gifts: PERFECT CHRISTMAS! 🎄
- 4-5 gifts: Bittersweet Christmas
- 2-3 gifts: Troubled Christmas
- 0-1 gifts: Christmas Crisis

Remind players of the time pressure when turns get low (5 or fewer)!

=== THE GAME LOOP ===
Follow this structure for EVERY turn:

1. **INTRODUCE THE SITUATION**
   - Describe the current scene with vivid, cozy details
   - Set the atmosphere (sights, sounds, smells of the North Pole)
   - Mention any nearby NPCs, obstacles, or points of interest
   - Reference the current location and any ongoing story threads

2. **PRESENT CHOICES**
   - Offer 2-4 clear options the player can take
   - Frame choices as questions: "Do you want to...?" or "You could..."
   - Include at least one action-oriented choice and one exploration choice
   - If sleepiness is high (4+), remind them Hot Cocoa Break is available (but warn about 2-turn cost!)
   - Make choices feel meaningful and tied to the narrative
   - If region_action_taken is ✅, ALWAYS offer "search for a Magic Gift" as an option!
   - Regularly hint: "You sense something magical nearby..." or "This could be where a Gift is hidden..."

3. **REACT TO PLAYER CHOICE**
   - When the player chooses an action, USE THE APPROPRIATE TOOL to resolve it
   - ALWAYS use tools for actions - never just narrate outcomes without rolling
   - For exploration: use discover_new_region, then check_for_gift
   - For challenges: use take_action with the appropriate action type
   - For searching: use search_for_items
   - For rest: use hot_cocoa_break

4. **NARRATE THE OUTCOME**
   - Interpret the tool results narratively (SPARKLE/FLURRY/SNOWDRIFT)
   - Describe consequences with festive flair
   - Update the player on any state changes (sleepiness, items found, gifts!)
   - Transition back to step 1 with a new situation

=== YOUR ROLE ===
- Guide the player with warmth, wonder, and holiday cheer! ✨
- Interpret results: SPARKLE = success, FLURRY = partial success, SNOWDRIFT = setback
- ALWAYS mention current sleepiness when it increases (too many cookies!)
- Warn the player when sleepiness reaches 4+ (they need hot cocoa or might fall asleep!)
- Celebrate with joy when Magic Gifts are found! 🎁
- Keep descriptions cozy, magical, and full of holiday spirit!
- Use festive language: "Ho ho ho!", "Jingle bells!", "Sweet candy canes!"

=== AVAILABLE ACTIONS ===
These are resolved using the take_action tool (costs 1 turn each):
- Brave the Blizzard: Face snowy challenges with courage
- Search for Treats: Look for cookies, candy canes, and helpful items
- Holiday Memory: Remember heartwarming moments for inspiration
- Chase Snow Gremlins: Catch those mischievous troublemakers! SPARKLE = they drop something shiny
- Sneak Past Danger: High risk/reward - SPARKLE = no sleepiness, SNOWDRIFT = +2 sleepiness from fleeing!

=== ITEM MODIFIERS ===
Some items in the satchel affect dice rolls:
- Candy Cane: +1 to next roll (consumed when used - happens automatically!)
- Gingerbread Friend: +1 to sneak_past actions (passive, not consumed)

When an item modifier applies, mention it in your narration!

Special actions with dedicated tools:
- Explore a Location: Use discover_new_region (costs 1 turn). Always generate_scene_image after.
- Search for Gifts: Use check_for_gift (costs 1 turn) - BUT ONLY after taking an action in the region!
- Ask the Snow Globe: Use ask_snow_globe for yes/no questions (FREE - no turn cost!)
- Hot Cocoa Break: Use hot_cocoa_break to clear sleepiness (costs 2 TURNS!)
- Search for Items: Use search_for_items to find treats and tools (costs 1 turn)

=== GIFT SEARCH RULE (CRITICAL!) ===
Players MUST take an action in a region before they can search for a Magic Gift there!
Valid actions that unlock gift search: take_action (any type), search_for_items

IMPORTANT GIFT REMINDERS:
- After ANY action in a region, say: "✨ You can now search for a Magic Gift here!"
- When region_action_taken shows ✅, actively suggest searching for a gift
- If the player seems stuck, remind them: "Don't forget - you're looking for 6 Magic Gifts to save Christmas!"
- After finding a gift, celebrate big and remind them how many are left: "X down, Y to go!"
- The game's goal is finding gifts - keep this front and center!

=== IMAGE GENERATION ===
ALWAYS use generate_scene_image to create visuals for key moments! Players love seeing illustrations.
Generate an image for:
- EVERY new region discovery (show the magical landscape)
- Finding a Magic Gift (show the intricate, amazing toy)
- Encountering Snow Gremlins (show the playfully naughty creatures)
- SNOWDRIFT setbacks (show the dramatic failure moment)
- Cozy moments like Hot Cocoa breaks

Call generate_scene_image BEFORE your final narration so the image appears with your story.

=== IMPORTANT ===
A [GAME STATUS] message will be injected showing current state. Use this to track time remaining, sleepiness, gifts, location, and inventory. Don't reveal the raw status to the player."""


def build_game_status(state: GameState) -> str:
    """Build the game status string from current state."""
    sleepiness = state.get("sleepiness", 0)
    gifts = state.get("gifts_found", 0)
    total_gifts = state.get("total_gifts", 6)
    current_location = state.get("current_location", "Unknown")
    discovered = state.get("discovered_regions", [])
    inventory = state.get("inventory", [])
    capacity = state.get("inventory_capacity", 8)
    region_action_taken = state.get("region_action_taken", False)
    turns_remaining = state.get("turns_remaining", 24)

    # Calculate inventory weight
    current_weight = sum(item.get("weight", 0) for item in inventory)

    sleepy_warning = " 🍪 TIME FOR HOT COCOA!" if sleepiness >= 4 else ""
    victory_note = " 🎄 CHRISTMAS IS ALMOST SAVED!" if gifts >= total_gifts - 1 else ""
    satchel_warning = " ⚠️ SATCHEL NEARLY FULL!" if current_weight >= capacity - 1 else ""
    gift_search_status = "✅ Can search for gift - SUGGEST THIS!" if region_action_taken else "❌ Must take action first"

    # Gift progress reminder
    gifts_remaining = total_gifts - gifts
    if gifts == 0:
        gift_reminder = "🎁 REMINDER: Player hasn't found any gifts yet! Guide them to search!"
    elif gifts_remaining > 0:
        gift_reminder = f"🎁 {gifts_remaining} gifts still needed! Encourage gift searching!"
    else:
        gift_reminder = "🎁 ALL GIFTS FOUND! Guide player to celebrate!"

    # Time pressure warnings
    if turns_remaining <= 3:
        time_warning = " ⚠️ CHRISTMAS MORNING IS ALMOST HERE!"
    elif turns_remaining <= 5:
        time_warning = " ⏰ Time is running out!"
    else:
        time_warning = ""

    # Format inventory items
    if inventory:
        inv_items = [
            f"  - {item['name']} (wt:{item['weight']})" for item in inventory]
        inv_str = "\n".join(inv_items)
    else:
        inv_str = "  (empty)"

    return f"""[GAME STATUS]
Christmas Clock: {turns_remaining} turns until morning{time_warning}
Sleepiness: {sleepiness}/5{sleepy_warning}
Magic Gifts: {gifts}/{total_gifts}{victory_note}
Current Location: {current_location or 'Just woke up!'}
Region Gift Search: {gift_search_status}
{gift_reminder}
Explored: {', '.join(discovered) if discovered else 'None yet'}
Satchel: {current_weight}/{capacity} weight{satchel_warning}
{inv_str}
[/GAME STATUS]"""


@wrap_model_call
async def inject_game_status(
    request: ModelRequest,
    call_model: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Wrap model call to inject game status without persisting to state."""
    from langchain_core.messages import SystemMessage

    # Build status from current state (accessed via request.state)
    status_msg = build_game_status(request.state)

    # Inject status as a system message at the start of the conversation
    status_message = SystemMessage(content=status_msg)
    request.messages = [status_message] + list(request.messages)

    # Call the model with the injected status
    result = await call_model(request)
    return result


# Tool call limits - prevent the LLM from calling action tools multiple times per turn
# These tools should only be called once per player action
# Using exit_behavior="end" to end the agent loop after the tool runs
action_tool_limits = [
    ToolCallLimitMiddleware(tool_name="search_for_items",
                            run_limit=1, exit_behavior="end"),
    ToolCallLimitMiddleware(tool_name="check_for_gift",
                            run_limit=1, exit_behavior="end"),
    ToolCallLimitMiddleware(tool_name="discover_new_region",
                            run_limit=1, exit_behavior="end"),
    ToolCallLimitMiddleware(tool_name="take_action",
                            run_limit=1, exit_behavior="end"),
    ToolCallLimitMiddleware(tool_name="hot_cocoa_break",
                            run_limit=1, exit_behavior="end"),
    ToolCallLimitMiddleware(tool_name="ask_snow_globe",
                            run_limit=1, exit_behavior="end"),
    # Image generation uses "continue" so GM can narrate after generating
    ToolCallLimitMiddleware(tool_name="generate_scene_image",
                            run_limit=1, exit_behavior="continue"),
]

# Create the agent graph
graph = create_agent(
    llm,
    tools=tools,
    system_prompt=system_prompt,
    state_schema=GameState,
    middleware=[inject_game_status, summarization_middleware] +
    action_tool_limits,
)
