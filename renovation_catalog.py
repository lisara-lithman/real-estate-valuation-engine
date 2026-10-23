"""
renovation_catalog.py
---------------------
Professional Renovation & Capital Improvement Catalog for the Enterprise Real Estate AI Engine.

Each package defines:
- title: Display name with icon
- description: Domain description of the renovation scope
- default_cost: Standard industry contractor cost estimate ($)
- feature_deltas: Precise feature updates in original domain units
"""

RENOVATION_CATALOG = {
    "🍳 Gourmet Kitchen Remodel": {
        "title": "🍳 Gourmet Kitchen Remodel",
        "description": "Full tear-out, premium quartz countertops, custom soft-close cabinetry, and chef-grade stainless appliances.",
        "default_cost": 15000,
        "min_cost": 5000,
        "max_cost": 40000,
        "updates": {
            "KitchenQual": {"mode": "set", "value": 5},          # 5 = 'Ex' (Excellent)
            "OverallQual": {"mode": "boost", "value": 1, "max": 10} # Quality tier upgrade capped at 10
        }
    },
    "🛁 Luxury Master Bath Overhaul": {
        "title": "🛁 Luxury Master Bath Overhaul",
        "description": "En-suite master bath addition/remodel with frameless glass dual-shower, heated floors, and double vanity.",
        "default_cost": 10000,
        "min_cost": 3000,
        "max_cost": 30000,
        "updates": {
            "FullBath": {"mode": "add", "value": 1},
            "TotalBath": {"mode": "add", "value": 1.0},
            "OverallQual": {"mode": "boost_if_below", "threshold": 8, "value": 1}
        }
    },
    "🛋️ Finished Basement Rec Room": {
        "title": "🛋️ Finished Basement Rec Room",
        "description": "Convert cold unfinished basement into high-grade insulated living quarters (entertainment/rec room).",
        "default_cost": 20000,
        "min_cost": 8000,
        "max_cost": 50000,
        "updates": {
            "BsmtQual": {"mode": "set", "value": 5},              # 5 = 'Ex'
            "BsmtFinType1": {"mode": "set", "value": 6},          # 6 = 'GLQ' (Good Living Quarters)
            "OverallQual": {"mode": "boost", "value": 1, "max": 10}
        }
    },
    "☀️ Outdoor Living Wood Deck": {
        "title": "☀️ Outdoor Living Wood Deck",
        "description": "Construct a 250 sq ft treated composite entertainment deck with custom railings and ambient lighting.",
        "default_cost": 8000,
        "min_cost": 2500,
        "max_cost": 25000,
        "updates": {
            "WoodDeckSF_raw": {"mode": "set", "value": 250}       # Adds 250 sq ft raw wood deck
        }
    }
}
