"""
Test script for running the Vivarium RPG locally.
The actual graph is defined in rpg_graph/agent.py
"""
from dotenv import load_dotenv

load_dotenv()

from rpg_graph.agent import graph
from rpg_graph.utils.state import welcome_message, welcome_image

# Re-export for backwards compatibility with langgraph.json
app = graph

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "game_1"}}

    result = graph.invoke({
        "messages": [welcome_message],
        "latest_image": welcome_image
    }, config)
    print(result["messages"][-1].content)
    print(f"\n--- Game State ---")
    print(f"Sleepiness: {result.get('sleepiness', 0)}")
    print(f"Magic Gifts: {result.get('gifts_found', 0)}/{result.get('total_gifts', 6)}")
    print(f"Location: {result.get('current_location', 'Unknown')}")
