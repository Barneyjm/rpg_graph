from langgraph.graph import StateGraph, MessagesState, END
from langchain.memory import ConversationEntityMemory
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate


from langchain_fireworks import ChatFireworks

from typing import TypedDict, List
import random
import os

from dotenv import load_dotenv

load_dotenv()

llm = ChatFireworks(model="accounts/fireworks/models/llama-v3p1-70b-instruct")
# Example invocation with entity memory
thread_id = "game_thread_1"
# memory_file = f"{thread_id}_memory.json"
# memory = ConversationEntityMemory(llm=llm, file_path=memory_file)

memory = MemorySaver()


# Prompts from https://vivarium.tiddlyhost.com/

game_master_prompt = """You are a GameMaster helping host a game of Vivarium. You will guide The Player through the game, provide descriptions, and help The Player make decisions. Don't introduce yourself. Keep the game as immsersive as possible.
Vivarium is won by reactivating the Six Vivariums. The six vivariums are the seeds that the desert moon Saharantis will use to restore its greenery and summon communities once again. They still emit a subtle ultrasonic hum.
The Player are a Biosentinel , a guardian of life. And The Player have just awakened from The Player cryopod,
long after the apocalypse, in what is now a desert.
The Player memories are mostly lost and hazy at best.
"""

player_prompt = """
The Player remember the mission:
to find the Vivariums and revitalize The Player lunar homeworld: Saharantis;
to avoid and deactivate the remaining Automaniacs , the doomsday automatons
; and maybe, just maybe... find others, roaming or still in their cryopods.

As The Player explore the world and regain memory of the past, The Player will be able to seed the life of the future world.

The player awakes in a region of the world. The region is a {location}.

Their most recent turn actions are contained within the "action" field below:"""

summary_prompt = """
The Player has taken the following actions:
{summary}

use these historical actions to craft the game into a memorable experience for The Player.
"""

foundation_prompt = """
The player is the last of the BIO-SENTINELS . The Player awoke long after the apocalypse, with forgotten and blurred memories, but a clear mission: reactivate the Vivariums , while avoiding and fighting the Automaniacs, the automatons of the apocalypse that still carry with them the programming of destruction.

As The Player explore the world, The Player recall the past, seeding and reviving a green world that lies beneath what is now the desert of The Playerr lunar homeworld, Saharantis.

Taking Actions
Actions guide The Playerr journey through the world. Each action helps The Player resolve whatever questions The Player have, or whatever The Player decide to do.

When The Player perform an Action , do the following:

Take 2 cards from The Playerr Adventure Deck .
Roll The Playerr Action Dice and add up.
Add the modifier to get The Playerr Score .
Then, interpret the results and discard the cards:

If The Playerr score is higher than both cards, there is Light .
If The Playerr score is only higher than one of the cards, there is Penumbra .
Otherwise, there is Darkness .
Aces are worth 1, Jacks are worth 11, Queens are worth 12, and Kings are worth 13. Each action is stated as Action+(Modifier), and helps The Player determine what happens when there is Light , Dark , or Darkness ."""

game_prompt = """
Reactivating the Six Vivariums
The six vivariums are the seeds that the desert moon Saharantis will use to restore its greenery and summon communities once again. They still emit a subtle ultrasonic hum. To locate the vivariums, do Discover a Region . If the card you add is a face card (Jack (11), Queen (12), or King (13)), there is one.

Rest and Fatigue
The Player may be required to mark a Fatigue when taking an action. When The Playerr fatigue marker fills or The Playerr Adventure Deck empties , The Player must rest or flee to stay alive—or risk snuffing out The Playerr own light. On a Rest , write an entry in The Playerr Journal , shuffle the cards The Player discarded during The Playerr journey, and erase all Fatigue .

Fighting Automaniacs
If The Player find any, The Player may fight them. The Player will stack cards according to Fighting Automaniacs . When the total stacked cards matches the Automaniac's Vitality, then it is defeated and deactivated. If The Player flee, the stacked cards remain until the next encounter.

Terradroid (5 cards) : Quadrupedal, headless terrestrial android with an upper front arm. Fires stun darts. Red ♦♥ cards are worth double when stacked.
Aerodrone (8 cards) : Its propellers allow high maneuverability and landing/taking off almost anywhere. It launches nets to confine its victims. Black ♠ ♣ cards are worth double when stacked.
Pack (13 cards) : Groups of three Aerodrones or three Terradroids that hunt in an efficient and terrifying group. Swords ♠ are worth double when stacked.
Mixed Pack (21 cards) : Trios that combine air and ground attack/recon for greater effectiveness and lethality. Hearts ♥ are worth double when stacked."""

actions_prompt = """
You are the game master. You will guide the player through the game. The player will take actions and you will determine the outcome of those actions.
Here are the types of actions they can take along with the general results. Tailor the story to the results based on the actions chosen by the player:

Facing the Risk
    When facing adversity, Action+Style .
    With Light , he has a total success .
    With Penumbra , The Player have partial success .
    With Darkness , The Player have a setback , mark 1 Fatigue .

Search for Relics
    When searching for something lost from the past, use Action+Style to determine how many Relics The Player find.
    With Light , there are 2 relics .
    With Penumbra , there is 1 relic .
    With Darkness , there is an Automaniac , it scores 1 Fatigue .

Flashback
    When a relic evokes knowledge, determine the number of Relics to pay, and Action+Relic .
    With Luz , The Player get detailed information
    With Penumbra , The Player get incomplete information .
    With Darkness , The Player get Ambiguous .

Discover a region
    When The Player are looking for a new path, Action+Style .
    With Light , add the two cards to The Player map.
    With Penumbra , add one of the cards to The Player map.
    With Darkness , Impassable Path, mark 1 Fatigue . Keep just one card and either GET AN ANSWER FROM THE ORACLE or FACE THE RISK .

Fighting against Curse
    When The Player face a Curse, Action+Style . Repeat until the stacked cards equal Vitality.
    With Light , stack the two cards against the Curse.
    With Penumbra , stack one of the cards against the Curse.
    With Darkness , AVOID DANGER.

Avoiding Danger
    When avoiding an imminent threat, Action+Style .
    With Light , The Player avoid danger .
    With Penumbra , score 1 Fatigue .
    With Darkness , score 2 Fatigues .

Getting an Answer from the Oracle
    When The Player require Yes/No answers, Action+2 if likely. Action+0 unlikely. Action+1 50/50
    With Light , the answer is " Yes, and... ".
    With Penumbra , the answer is " Yes, but... ".
    With Darkness , the answer is " No, and... ".
"""

class Card(TypedDict):
    value: int
    color: str

colors = ['heart', 'diamonds', 'spades', 'clubs']
deck = [Card({"value":value, "color":color}) for value in range(1, 14) for color in colors]

regions = {
    "11":	"Planting",
    "12":	"Burrow",
    "13":	"Clearing",
    "14":	"Forest",
    "15":	"Tree",
    "16":	"Ruins",
    "21":	"Hills",
    "22":	"Tunnels",
    "23":	"Swamp",
    "24":	"Mountains",
    "25":	"River",
    "26":	"Lake",
    "31":	"Ocean",
    "32":	"Island",
    "33":	"Plain",
    "34":	"Glacier",
    "35":	"Pond",
    "36":	"Desert",
    "41":	"Tundra",
    "42":	"Caves",
    "43":	"Meadow",
    "44":	"Nest",
    "45":	"City",
    "46":	"Cliffs",
    "51":	"Gardens",
    "52":	"Jungle",
    "53":	"Prairie",
    "54":	"Wasteland",
    "55":	"Hive",
    "56":	"Canyon",
    "61":	"Catacombs",
    "62":	"Volcano",
    "63":	"Wetlands",
    "64":	"Tomb",
    "65":	"Estuary",
    "66":	"Hollow",
}

themes = {
"11":	"beauty",
"12":	"Loyalty",
"13":	"Money",
"14":	"Life",
"15":	"Death",
"16":	"War",
"21":	"Peace",
"22":	"family",
"23":	"Power",
"24":	"Friendship",
"25":	"Change",
"26":	"Tradition",
"31":	"Survival",
"32":	"Liberty",
"33":	"Weather",
"34":	"Corruption",
"35":	"Hope",
"36":	"Love",
"41":	"Revenge",
"42":	"Identity",
"43":	"Redemption",
"44":	"Justice",
"45":	"Honor",
"46":	"Forgiveness",
"51":	"Ambition",
"52":	"Faith",
"53":	"Greed",
"54":	"Equality",
"55":	"Deception",
"56":	"Legacy",
"61":	"Truth",
"62":	"Sacrifice",
"63":	"Loneliness",
"64":	"Resilience",
"65":	"Betrayal",
"66":	"Fame",
}

events = {
"11":	"Directing",
"12":	"Redeem",
"13":	"Infiltrate",
"14":	"View",
"15":	"Pursue",
"16":	"Concentrate",
"21":	"Planning",
"22":	"Serve",
"23":	"Create",
"24":	"Escape",
"25":	"Parliamentarian",
"26":	"Insure",
"31":	"Stealing",
"32":	"search",
"33":	"Sabotage",
"34":	"Adapt",
"35":	"Inspire",
"36":	"betray",
"41":	"Deceiving",
"42":	"Hide",
"43":	"Recover",
"44":	"take advantage",
"45":	"Attacking",
"46":	"Observing",
"51":	"Sacrifice",
"52":	"Surviving",
"53":	"Following",
"54":	"Persuade",
"55":	"Explore",
"56":	"Execute",
"61":	"Revenge",
"62":	"Help",
"63":	"Forgive",
"64":	"Destroy",
"65":	"Protect",
"66":	"Learn",
}

people = {
"11":	"Soldier",
"12":	"Farmer",
"13":	"Thief",
"14":	"Gentleman",
"15":	"Noble",
"16":	"Peddler",
"21":	"Sailor",
"22":	"Peasant",
"23":	"Spy",
"24":	"Artisan",
"25":	"Pirate",
"26":	"Bandit",
"31":	"Monk",
"32":	"Healer",
"33":	"Guard",
"34":	"Nomad",
"35":	"Hunter",
"36":	"Leader",
"41":	"Assailant",
"42":	"Paria",
"43":	"Miner",
"44":	"Worker",
"45":	"Bardo",
"46":	"Strange",
"51":	"Guerrero",
"52":	"Scholar",
"53":	"Bounty Hunter",
"54":	"Creature",
"55":	"Mendigo",
"56":	"Tailor",
"61":	"Magician",
"62":	"Pilluelo",
"63":	"Engineer",
"64":	"Herbalist",
"65":	"Brewer",
"66":	"Tracker",
}


class Player(TypedDict):
    name: str
    fatigue: int = 0
    skill_0: str
    skill_1: str
    skill_2: str

class Turn(TypedDict):
    player: Player
    location: str
    cards: List[Card]
    rolls: List[int]
    result: str
    input: str
    action: str
    output: str

# Define the state for the game
class GameState(MessagesState):
    turns: List[Turn] = []
    player: Player = {}
    completed_regions: List[str] = []
    summary: str
    latest_action: str

# Define the logic for each node
def setup_node(state: GameState) -> GameState:
    dice = [random.randint(1, 6) for _ in range(2)]
    location = regions["".join([str(d) for d in dice])]
    if state.get("turns") is None:
        state["turns"] = []
    prompt = ChatPromptTemplate.from_messages([
            ("system", game_master_prompt),
            ("system", player_prompt)
        ])
    init = llm.invoke(prompt.format(location=location))
    state["turns"].append(Turn({"input": init, "output": init, "location": location}))
    return state

def action_selection_node(state: GameState) -> GameState:
    state["turns"].append(Turn({"input": state["turns"][-1]["output"], "location": state["turns"][-1]["location"]}))
    return state

def game_master_description_node(state: GameState) -> GameState:
    # cards = [random.choice(deck) for _ in range(2)]
    # dice = [random.randint(1, 6) for _ in range(2)]
    # state["turns"][-1]["cards"] = cards
    # state["turns"][-1]["dice"] = dice

    # dice_sum = sum(dice) + 2 # Could be state.difficulty [easy/medium/hard]

    # if dice_sum > cards[0]["value"] and dice_sum > cards[1]["value"]:
    #     state["turns"][-1]["result"] = "LIGHT"
    # elif dice_sum > cards[0]["value"] or dice_sum > cards[1]["value"]:
    #     state["turns"][-1]["result"] = "PENUMBRA"
    # else:
    #     state["turns"][-1]["result"] = "DARKNESS"
    # Describe the outcome and provide new options
    prompt = ChatPromptTemplate.from_messages([
            ("system", game_master_prompt),
            ("system", player_prompt),
            ("ai", state["turns"][-1]["input"].content),
            ("human", str(state["messages"]))
        ])
        
    state["turns"][-1]["output"] = llm.invoke(prompt.format(location=state["turns"][-1]["location"]))
    # state["turns"][-1]["output"] = llm.invoke(str(state["messages"]))
    # state["turns"][-1]["summary"] = llm.invoke(f"concisely summarize this turn: {state['turns'][-1]}")

    return state

def finalize_game(state: GameState) -> GameState:
    state["turns"].append(Turn({"input": "Game Over", "output": "Game Over"}))
    return state

def should_continue_game(state: GameState) -> str:
    # Simulate player choosing to continue or not
    if state["player"]["fatigue"] > 5:
        return "finalize_game"
    return "action_selection_node"
    # if "completed_regions" not in state.keys():
    #     state["completed_regions"] = []
    #     return "action_selection_node"
    # elif len(state["completed_regions"]) < 2:
    # else:

    # else:
    #     return "finalize_game"

def summarize_conversation(state: GameState) -> GameState:
    # First, we summarize the conversation
    summary = state.get("summary", "")
    if summary:
        # If a summary already exists, we use a different system prompt
        # to summarize it than if one didn't
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    messages = [turn["output"] for turn in state["turns"][-3:]] + [summary_message]
    state["summary"] = llm.invoke(messages)
    return state

def should_summarize(state: GameState) ->  str:
    """Return the next node to execute."""
    if state["turns"] > 3:
        return "summarize_conversation"
    # Otherwise we can just end
    return "action_selection_node"

# Create the graph
graph = StateGraph(GameState)

# Add nodes
graph.add_node("setup_node", setup_node)
graph.add_node("action_selection_node", action_selection_node)
# graph.add_node("summarize_conversation", summarize_conversation)
# graph.add_node("outcome_determination_node", outcome_determination_node)
graph.add_node("game_master_description_node", game_master_description_node)
graph.add_node("finalize_game", finalize_game)

# Define edges
graph.add_edge("setup_node", "action_selection_node")
graph.add_edge("action_selection_node", "game_master_description_node")
# graph.add_conditional_edges("action_selection_node", summarize_conversation, {"action_selection_node": "action_selection_node", "summarize_conversation": "summarize_conversation"})

# graph.add_edge("outcome_determination_node", "game_master_description_node")
graph.add_conditional_edges("game_master_description_node", should_continue_game, {"action_selection_node": "action_selection_node", "finalize_game": "finalize_game"})

# Set entry and finish points
graph.set_entry_point("setup_node")
graph.add_edge("game_master_description_node", "finalize_game")
graph.add_edge("finalize_game", END)

# Compile the graph
app = graph.compile(interrupt_before=["action_selection_node"])



initial_state = {"messages": []}
result = app.invoke(initial_state)
print(result)

