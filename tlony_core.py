"""
Common constants and helpers shared across the TLoNY YOLO scripts.
"""

from __future__ import annotations

from typing import Dict, Set

# Dataset metadata -----------------------------------------------------------------

DATASET_REPO_ID = "mehmetkeremturkcan/traffic-lights-of-new-york"
DATASET_FILENAME = "tlony.zip"


# Class naming ---------------------------------------------------------------------

TLONY_CLASSES = [
    "red",          # 0
    "green",        # 1
    "yellow",       # 2
    "red+yellow",   # 3
    "unknown",      # 4
    "pedred",       # 5
    "pedgreen",     # 6
    "pedunknown",   # 7
]

# Vehicular classes are the only ones that drive the state machine / HUD.
VEHICULAR_CLASSES: Set[str] = {"red", "green", "yellow", "red+yellow"}


def map_class_to_group(class_name: str) -> str:
    """
    Groups fine-grained TLoNY classes into high-level traffic-light states.
    Returns one of: RED, GREEN, YELLOW, OTHER.
    """
    if class_name in ("red", "red+yellow"):
        return "RED"
    if class_name == "green":
        return "GREEN"
    if class_name == "yellow":
        return "YELLOW"
    return "OTHER"


def interpret_condition(class_name: str) -> str:
    """
    Returns a human-readable description for TLoNY classes.
    """
    mapping: Dict[str, str] = {
        "red": "Semáforo vehicular en ROJO",
        "green": "Semáforo vehicular en VERDE",
        "yellow": "Semáforo vehicular en AMARILLO",
        "red+yellow": "Semáforo vehicular en ROJO+AMARILLO",
        "unknown": "Semáforo vehicular (color desconocido)",
        "pedred": "Semáforo PEATONAL en ROJO (No caminar)",
        "pedgreen": "Semáforo PEATONAL en VERDE (Caminar)",
        "pedunknown": "Semáforo PEATONAL (color desconocido)",
    }
    return mapping.get(class_name, class_name)
