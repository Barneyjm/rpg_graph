"""
Test script for running the Vivarium RPG locally.
The actual graph is defined in rpg_graph/agent.py
"""
from dotenv import load_dotenv

load_dotenv()

from rpg_graph.agent import graph
from rpg_graph.utils.state import welcome_message

# Re-export for backwards compatibility with langgraph.json
app = graph

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "game_1"}}

    result = graph.invoke({"messages": [welcome_message]}, config)
    print(result["messages"][-1].content)
    print(f"\n--- Game State ---")
    print(f"Fatigue: {result.get('fatigue', 0)}")
    print(f"Vivariums: {result.get('vivariums_found', 0)}/6")
    print(f"Region: {result.get('current_region', 'Unknown')}")
