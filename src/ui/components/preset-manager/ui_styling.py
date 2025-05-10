"""
UI Styling Components for Preset Management Interface

This module defines styling components, themes, animations, and responsive design
elements for the preset management UI. It provides a comprehensive set of styles
and utility functions to ensure a consistent, responsive, and aesthetically
pleasing user interface across desktop and mobile views.
"""

import json
import random
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


# Color definitions as RGBA tuples (R, G, B, A)
class Colors:
    # Primary palette
    PRIMARY = (56, 128, 255, 1.0)
    PRIMARY_LIGHT = (98, 155, 255, 1.0)
    PRIMARY_DARK = (28, 71, 166, 1.0)
    
    # Secondary palette
    SECONDARY = (255, 82, 82, 1.0)
    SECONDARY_LIGHT = (255, 120, 120, 1.0)
    SECONDARY_DARK = (200, 40, 40, 1.0)
    
    # Accent colors
    ACCENT_1 = (142, 45, 226, 1.0)  # Purple
    ACCENT_2 = (255, 179, 0, 1.0)   # Gold
    ACCENT_3 = (0, 184, 148, 1.0)   # Teal
    ACCENT_4 = (252, 92, 101, 1.0)  # Coral
    
    # Neutrals
    WHITE = (255, 255, 255, 1.0)
    BLACK = (0, 0, 0, 1.0)
    GRAY_100 = (243, 244, 246, 1.0)
    GRAY_200 = (229, 231, 235, 1.0)
    GRAY_300 = (209, 213, 219, 1.0)
    GRAY_400 = (156, 163, 175, 1.0)
    GRAY_500 = (107, 114, 128, 1.0)
    GRAY_600 = (75, 85, 99, 1.0)
    GRAY_700 = (55, 65, 81, 1.0)
    GRAY_800 = (31, 41, 55, 1.0)
    GRAY_900 = (17, 24, 39, 1.0)
    
    # Semantic colors
    SUCCESS = (0, 200, 83, 1.0)
    WARNING = (255, 145, 0, 1.0)
    ERROR = (255, 23, 68, 1.0)
    INFO = (0, 145, 234, 1.0)
    
    # Consciousness level colors - representing different brainwave states
    DELTA = (75, 0, 130, 1.0)     # Deep sleep - deepest consciousness
    THETA = (106, 90, 205, 1.0)   # Deep meditation
    ALPHA = (0, 128, 128, 1.0)    # Relaxed awareness
    BETA = (70, 130, 180, 1.0)    # Active thinking
    GAMMA = (220, 20, 60, 1.0)    # Peak consciousness and perception

    @staticmethod
    def rgba(color: Tuple[int, int, int, float]) -> str:
        """Convert a color tuple to rgba string."""
        return f"rgba({color[0]}, {color[1]}, {color[2]}, {color[3]})"
    
    @staticmethod
    def rgb(color: Tuple[int, int, int, float]) -> str:
        """Convert a color tuple to rgb string."""
        return f"rgb({color[0]}, {color[1]}, {color[2]})"
    
    @staticmethod
    def hex(color: Tuple[int, int, int, float]) -> str:
        """Convert a color tuple to hex string."""
        return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
    
    @staticmethod
    def with_alpha(color: Tuple[int, int, int, float], alpha: float) -> Tuple[int, int, int, float]:
        """Return a color with modified alpha value."""
        return (color[0], color[1], color[2], alpha)


class ThemeType(Enum):
    """Enum for available UI themes."""
    LIGHT = "light"
    DARK = "dark"
    HIGH_CONTRAST = "high_contrast"
    NEURAL = "neural"
    QUANTUM = "quantum"
    SACRED = "sacred"
    CONSCIOUSNESS = "consciousness"


class Themes:
    """UI theme definitions for preset management."""
    
    # Base themes
    LIGHT = {
        "name": "Light",
        "background": Colors.WHITE,
        "surface": Colors.GRAY_100,
        "card": Colors.WHITE,
        "text": {
            "primary": Colors.GRAY_900,
            "secondary": Colors.GRAY_700,
            "tertiary": Colors.GRAY_500,
            "on_primary": Colors.WHITE,
            "on_secondary": Colors.WHITE,
            "on_accent": Colors.WHITE,
        },
        "border": Colors.GRAY_300,
        "divider": Colors.GRAY_200,
        "primary": Colors.PRIMARY,
        "secondary": Colors.SECONDARY,
        "accent": Colors.ACCENT_1,
        "shadow": "rgba(0, 0, 0, 0.1)",
        "status": {
            "success": Colors.SUCCESS,
            "warning": Colors.WARNING,
            "error": Colors.ERROR,
            "info": Colors.INFO,
        }
    }
    
    DARK = {
        "name": "Dark",
        "background": Colors.GRAY_900,
        "surface": Colors.GRAY_800,
        "card": Colors.GRAY_800,
        "text": {
            "primary": Colors.WHITE,
            "secondary": Colors.GRAY_300,
            "tertiary": Colors.GRAY_400,
            "on_primary": Colors.WHITE,
            "on_secondary": Colors.WHITE,
            "on_accent": Colors.WHITE,
        },
        "border": Colors.GRAY_700,
        "divider": Colors.GRAY_700,
        "primary": Colors.PRIMARY,
        "secondary": Colors.SECONDARY,
        "accent": Colors.ACCENT_1,
        "shadow": "rgba(0, 0, 0, 0.3)",
        "status": {
            "success": Colors.SUCCESS,
            "warning": Colors.WARNING,
            "error": Colors.ERROR,
            "info": Colors.INFO,
        }
    }
    
    HIGH_CONTRAST = {
        "name": "High Contrast",
        "background": Colors.BLACK,
        "surface": Colors.GRAY_900,
        "card": Colors.GRAY_900,
        "text": {
            "primary": Colors.WHITE,
            "secondary": Colors.WHITE,
            "tertiary": Colors.GRAY_300,
            "on_primary": Colors.BLACK,
            "on_secondary": Colors.BLACK,
            "on_accent": Colors.BLACK,
        },
        "border": Colors.WHITE,
        "divider": Colors.WHITE,
        "primary": (255, 255, 0, 1.0),  # Yellow for high contrast
        "secondary": (0, 255, 255, 1.0),  # Cyan for high contrast
        "accent": (255, 0, 255, 1.0),  # Magenta for high contrast
        "shadow": "rgba(255, 255, 255, 0.3)",
        "status": {
            "success": (0, 255, 0, 1.0),
            "warning": (255, 255, 0, 1.0),
            "error": (255, 0, 0, 1.0),
            "info": (0, 255, 255, 1.0),
        }
    }
    
    # Specialized themes
    NEURAL = {
        "name": "Neural",
        "background": (25, 26, 45, 1.0),  # Deep blue-black
        "surface": (35, 36, 65, 1.0),
        "card": (45, 46, 85, 1.0),
        "text": {
            "primary": (220, 220, 255, 1.0),
            "secondary": (180, 180, 255, 1.0),
            "tertiary": (140, 140, 240, 1.0),
            "on_primary": Colors.WHITE,
            "on_secondary": Colors.WHITE,
            "on_accent": Colors.WHITE,
        },
        "border": (100, 100, 180, 1.0),
        "divider": (70, 70, 140, 1.0),
        "primary": (120, 99, 255, 1.0),  # Neural purple
        "secondary": (64, 186, 255, 1.0),  # Electric blue
        "accent": (255, 56, 100, 1.0),  # Neural pink
        "shadow": "rgba(76, 0, 255, 0.3)",
        "status": {
            "success": (0, 255, 170, 1.0),
            "warning": (255, 170, 0, 1.0),
            "error": (255, 0, 100, 1.0),
            "info": (0, 200, 255, 1.0),
        }
    }
    
    QUANTUM = {
        "name": "Quantum",
        "background": (10, 15, 30, 1.0),  # Deep space black
        "surface": (20, 25, 40, 1.0),
        "card": (30, 35, 50, 1.0),
        "text": {
            "primary": (210, 255, 255, 1.0),
            "secondary": (150, 230, 230, 1.0),
            "tertiary": (100, 200, 200, 1.0),
            "on_primary": Colors.BLACK,
            "on_secondary": Colors.BLACK,
            "on_accent": Colors.BLACK,
        },
        "border": (0, 200, 200, 1.0),
        "divider": (0, 150, 150, 1.0),
        "primary": (0, 255, 240, 1.0),  # Quantum cyan
        "secondary": (180, 255, 100, 1.0),  # Quantum green
        "accent": (255, 100, 255, 1.0),  # Quantum purple
        "shadow": "rgba(0, 255, 255, 0.2)",
        "status": {
            "success": (100, 255, 200, 1.0),
            "warning": (255, 230, 0, 1.0),
            "error": (255, 50, 120, 1.0),
            "info": (80, 200, 255, 1.0),
        }
    }
    
    SACRED = {
        "name": "Sacred Geometry",
        "background": (25, 15, 35, 1.0),  # Deep violet
        "surface": (35, 25, 45, 1.0),
        "card": (45, 35, 55, 1.0),
        "text": {
            "primary": (255, 250, 220, 1.0),  # Soft gold
            "secondary": (230, 210, 180, 1.0),
            "tertiary": (200, 180, 150, 1.0),
            "on_primary": Colors.BLACK,
            "on_secondary": Colors.BLACK,
            "on_accent": Colors.BLACK,
        },
        "border": (212, 175, 55, 1.0),  # Gold
        "divider": (150, 120, 40, 1.0),
        "primary": (212, 175, 55, 1.0),  # Gold
        "secondary": (176, 38, 255, 1.0),  # Purple
        "accent": (255, 89, 0, 1.0),  # Orange
        "shadow": "rgba(212, 175, 55, 0.2)",
        "status": {
            "success": (170, 210, 90, 1.0),
            "warning": (255, 190, 60, 1.0),
            "error": (230, 70, 90, 1.0),
            "info": (90, 160, 255, 1.0),
        }
    }
    
    CONSCIOUSNESS = {
        "name": "Consciousness",
        "background": (10, 10, 25, 1.0),  # Deep consciousness black
        "surface": (20, 20, 35, 1.0),
        "card": (30, 30, 45, 1.0),
        "text": {
            "primary": (230, 230, 255, 1.0),
            "secondary": (200, 200, 255, 1.0),
            "tertiary": (170, 170, 255, 1.0),
            "on_primary": Colors.BLACK,
            "on_secondary": Colors.BLACK,
            "on_accent": Colors.BLACK,
        },
        "border": (90, 60, 170, 1.0),
        "divider": (70, 40, 140, 1.0),
        "primary": Colors.DELTA,  # Deep consciousness purple
        "secondary": Colors.THETA,  # Meditation blue
        "accent": Colors.GAMMA,  # Peak consciousness red
        "shadow": "rgba(75, 0, 130, 0.3)",
        "status": {
            "success": Colors.ALPHA,
            "warning": Colors.BETA,
            "error": Colors.GAMMA,
            "info": Colors.THETA,
        }
    }
    
    @staticmethod
    def get_theme(theme_type: ThemeType) -> Dict:
        """Get a theme by its type."""
        theme_map = {
            ThemeType.LIGHT: Themes.LIGHT,
            ThemeType.DARK: Themes.DARK,
            ThemeType.HIGH_CONTRAST: Themes.HIGH_CONTRAST,
            ThemeType.NEURAL: Themes.NEURAL,
            ThemeType.QUANTUM: Themes.QUANTUM,
            ThemeType.SACRED: Themes.SACRED,
            ThemeType.CONSCIOUSNESS: Themes.CONSCIOUSNESS,
        }
        return theme_map.get(theme_type, Themes.LIGHT)
    
    @staticmethod
    def to_css_variables(theme: Dict) -> Dict[str, str]:
        """Convert a theme to CSS variables."""
        css_vars = {}
        
        # Process simple color properties
        for key in ["background", "surface", "card", "border", "divider", "primary", "secondary", "accent", "shadow"]:
            if key in theme:
                value = theme[key]
                if isinstance(value, tuple):
                    css_vars[f"--color-{key}"] = Colors.rgba(value)
                else:
                    css_vars[f"--color-{key}"] = value
        
        # Process nested text colors
        for text_key, text_value in theme.get("text", {}).items():
            if isinstance(text_value, tuple):
                css_vars[f"--color-text-{text

