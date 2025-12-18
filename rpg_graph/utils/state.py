from typing import Annotated, List
from operator import add

from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage


# Welcome message shown at start of new threads
welcome_message = AIMessage(content="""🎄 **Welcome to Santa's Workshop Adventure!** 🎅

*~Jingle bells play softly in the distance~*

You're a cheerful Elf Helper at the North Pole, and you've just woken up from a cozy nap by the fireplace. But wait... something's wrong!

A magical blizzard has scattered **Six Magic Gifts** across the North Pole—and without them, Christmas morning won't be complete! To make matters worse, mischievous **Snow Gremlins** are causing chaos everywhere!

Santa is counting on you, little elf! ❄️

Your pointy ears tingle with determination. The smell of gingerbread fills the air. Somewhere nearby, a reindeer bells jingle encouragingly.

✨ *The adventure begins... What would you like to do?* ✨

---
🎁 *Find all 6 Magic Gifts to save Christmas!*
🍪 *Watch your sleepiness—grab hot cocoa when you need it!*
⛄ *Share your adventure with friends!*""")


# Custom state with game tracking
class GameState(MessagesState):
    # Don't override messages - MessagesState already handles it with proper annotation
    fatigue: int = 0
    vivariums_found: int = 0
    current_region: str = ""
    discovered_regions: Annotated[List[str], add] = []
    relics: Annotated[List[str], add] = []
