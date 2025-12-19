"""Game constants and data for Santa's Workshop Adventure."""

# Card colors for the deck
COLORS = ['hearts', 'diamonds', 'spades', 'clubs']

# Full deck of cards
DECK = [{"value": value, "color": color}
        for value in range(1, 14) for color in COLORS]

# Region lookup table - maps dice rolls (d1d2) to location names
REGIONS = {
    "11": "Candy Cane Forest",
    "12": "Gingerbread Village",
    "13": "Snowflake Meadow",
    "14": "Reindeer Stables",
    "15": "The Giant Christmas Tree",
    "16": "Frozen Toy Workshop",
    "21": "Sugarplum Hills",
    "22": "Secret Elf Tunnels",
    "23": "Marshmallow Marsh",
    "24": "Peppermint Mountains",
    "25": "Hot Cocoa River",
    "26": "Frozen Cranberry Lake",
    "31": "Northern Lights Bay",
    "32": "Gingerbread Island",
    "33": "Snowy Plains",
    "34": "Crystal Ice Glacier",
    "35": "Enchanted Skating Pond",
    "36": "Cookie Dough Desert",
    "41": "Cozy Blanket Tundra",
    "42": "Tinsel Caves",
    "43": "Mistletoe Meadow",
    "44": "Owl's Winter Nest",
    "45": "Toy Town Square",
    "46": "Icicle Cliffs",
    "51": "Mrs. Claus's Gardens",
    "52": "Evergreen Jungle",
    "53": "Snowdrift Prairie",
    "54": "The Naughty List Wasteland",
    "55": "Busy Bee Honeycomb Bakery",
    "56": "Ribbon Canyon",
    "61": "Gift Wrapping Catacombs",
    "62": "Warm Hearth Mountain",
    "63": "Frosted Wetlands",
    "64": "Ancient Ornament Vault",
    "65": "Snowmelt Stream",
    "66": "Cozy Woodland Hollow",
}

# Magic gift names that can be found
GIFT_NAMES = [
    "Sparkling Snow Globe",
    "Golden Jingle Bell",
    "Magical Toy Train",
    "Enchanted Nutcracker",
    "Glowing Star Ornament",
    "Crystal Candy Cane",
]

# Item pools for search_for_items by card value
ITEM_POOLS = {
    "rare": ["hot_cocoa_thermos", "snow_globe", "gingerbread_man", "elf_lantern",
             "music_box", "sack_of_toys"],
    "medium": ["hot_cocoa_thermos", "snow_globe", "gingerbread_man", "elf_lantern", "toy_hammer"],
    "common": ["candy_cane", "jingle_bell", "snowflake_cookie", "holly_sprig", "ribbon"],
}
