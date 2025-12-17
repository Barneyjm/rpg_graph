from typing import Annotated, List
from operator import add

from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage


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
