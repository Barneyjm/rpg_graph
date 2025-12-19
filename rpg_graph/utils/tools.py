"""Tools for Santa's Workshop Adventure.

This module re-exports all game tools from their respective modules.
"""

# Re-export action tools
from rpg_graph.utils.action_tools import (
    roll_dice,
    draw_cards,
    take_action,
    discover_new_region,
    check_for_gift,
    hot_cocoa_break,
    ask_snow_globe,
)

# Re-export inventory tools
from rpg_graph.utils.inventory_tools import (
    check_inventory,
    pick_up_item,
    drop_item,
    use_item,
    search_for_items,
)

# Re-export image tools
from rpg_graph.utils.image_tools import generate_scene_image

# Export all tools as a list for the agent
tools = [
    # Action tools
    roll_dice,
    draw_cards,
    take_action,
    discover_new_region,
    check_for_gift,
    hot_cocoa_break,
    ask_snow_globe,
    # Inventory tools
    check_inventory,
    pick_up_item,
    drop_item,
    use_item,
    search_for_items,
    # Image tools
    generate_scene_image,
]
