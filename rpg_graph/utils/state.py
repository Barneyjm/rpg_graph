import base64
from pathlib import Path
from typing import Annotated, List, Optional
from operator import add
from dataclasses import dataclass

from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage


@dataclass
class InventoryItem:
    """An item that can be carried in the elf's satchel."""
    name: str
    weight: int  # 1-3 typically
    description: str
    effect: Optional[str] = None  # Optional gameplay effect

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "weight": self.weight,
            "description": self.description,
            "effect": self.effect
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InventoryItem":
        return cls(
            name=data["name"],
            weight=data["weight"],
            description=data["description"],
            effect=data.get("effect")
        )


# Predefined items that can be found in the game
GAME_ITEMS = {
    # Light items (weight 1)
    "candy_cane": InventoryItem("Candy Cane", 1, "A sweet peppermint treat", "+1 to next roll"),
    "jingle_bell": InventoryItem("Jingle Bell", 1, "A tiny golden bell that tinkles merrily"),
    "snowflake_cookie": InventoryItem("Snowflake Cookie", 1, "A delicious frosted cookie", "Recover 1 sleepiness"),
    "holly_sprig": InventoryItem("Holly Sprig", 1, "Bright red berries and green leaves"),
    "ribbon": InventoryItem("Festive Ribbon", 1, "A sparkly red and gold ribbon"),

    # Medium items (weight 2)
    "hot_cocoa_thermos": InventoryItem("Hot Cocoa Thermos", 2, "Stays warm forever!", "Full sleepiness recovery"),
    "snow_globe": InventoryItem("Mini Snow Globe", 2, "Shows glimpses of other locations", "Reroll location dice"),
    "gingerbread_man": InventoryItem("Gingerbread Friend", 2, "A living cookie companion!", "+1 to sneak actions"),
    "elf_lantern": InventoryItem("Elf Lantern", 2, "Glows with warm magical light", "Reveal hidden items"),
    "toy_hammer": InventoryItem("Toy Hammer", 2, "A tiny but mighty tool"),

    # Heavy items (weight 3)
    "sack_of_toys": InventoryItem("Sack of Toys", 3, "A miniature version of Santa's bag"),
    "frozen_turkey": InventoryItem("Frozen Turkey", 3, "Why are you carrying this?"),
    "music_box": InventoryItem("Music Box", 3, "Plays 'Jingle Bells' when opened", "Distract Snow Gremlins"),
    "giant_candy_cane": InventoryItem("Giant Candy Cane", 3, "As tall as an elf!", "Can be used as a pole"),
}


def load_welcome_image() -> str:
    """Load the welcome image as a base64 data URL."""
    image_path = Path(__file__).parent.parent / "assets" / "welcome.png"
    if image_path.exists():
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{image_b64}"
    return ""


# Pre-load the welcome image
welcome_image = load_welcome_image()


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


# Default inventory capacity (in weight units)
DEFAULT_INVENTORY_CAPACITY = 8


# Custom state with game tracking
# Field names match what the UI expects
class GameState(MessagesState):
    # Don't override messages - MessagesState already handles it with proper annotation
    sleepiness: int = 0
    gifts_found: int = 0
    total_gifts: int = 6  # Total gifts needed to win
    current_location: str = ""
    discovered_regions: Annotated[List[str], add] = []
    # Store latest generated image separately (not in messages sent to LLM)
    latest_image: Optional[str] = None
    # Track last dice/card results for UI display
    last_dice: Optional[List[int]] = None
    last_card: Optional[List[str]] = None
    # Inventory system - list of item dicts with name, weight, description, effect
    inventory: List[dict] = []
    inventory_capacity: int = DEFAULT_INVENTORY_CAPACITY  # Max weight the elf can carry
    # Track if player has taken an action in the current region (required before searching for gift)
    region_action_taken: bool = False
    # Christmas Clock - turns remaining until Christmas morning
    turns_remaining: int = 24
