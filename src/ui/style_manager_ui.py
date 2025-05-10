import json
import logging
import os
from collections import OrderedDict
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from neural_beat_architect.core.architect import StyleParameters
from neural_beat_architect.core.style_factory import StyleFactory
from neural_processing.quantum_sacred_enhancer import QuantumSacredEnhancer
from neural_processing.sacred_coherence import (apply_sacred_geometry,
                                                calculate_phi_ratio)

logger = logging.getLogger(__name__)

class StyleManagerUI:
    """
    A memory-efficient UI manager for handling style operations with one-click access
    to core functionality while minimizing resource usage.
    """
    
    # Quick transformation presets
    QUICK_TRANSFORMS = {
        "chill": {"intensity": -0.2, "tempo": -0.15, "complexity": -0.1, 
                  "consciousness_level": 3, "apply_phi": True},
        "energize": {"intensity": 0.25, "tempo": 0.2, "brightness": 0.15, 
                     "saturation": 0.1, "consciousness_level": 5},
        "deep": {"bass_emphasis": 0.3, "reverb": 0.2, "consciousness_level": 8, 
                 "apply_phi": True, "harmonic_richness": 0.15},
        "bright": {"brightness": 0.3, "treble_emphasis": 0.2, "clarity": 0.15, 
                   "consciousness_level": 5},
        "minimal": {"complexity": -0.3, "layers": -0.2, "percussion_complexity": -0.15},
        "complex": {"complexity": 0.3, "layers": 0.2, "percussion_complexity": 0.15,
                    "consciousness_level": 5, "apply_phi": True},
        "ethereal": {"reverb": 0.3, "delay": 0.2, "consciousness_level": 13, 
                     "apply_phi": True, "harmonic_richness": 0.2},
        "cosmic": {"consciousness_level": 21, "apply_phi": True, "harmonic_richness": 0.3,
                   "saturation": 0.2, "reverb": 0.25}
    }
    
    # Consciousness level Fibonacci sequence
    CONSCIOUSNESS_LEVELS = [1, 2, 3, 5, 8, 13, 21, 34]
    
    def __init__(self, style_factory: Optional[StyleFactory] = None, 
                 cache_size: int = 50, db_path: Optional[str] = None):
        """
        Initialize the StyleManagerUI with memory optimization.
        
        Args:
            style_factory: StyleFactory instance or None (will be created)
            cache_size: Maximum number of styles to cache in memory
            db_path: Path to style database file
        """
        self.style_factory = style_factory or StyleFactory()
        self.cache_size = cache_size
        self.current_style = None
        self.current_style_name = ""
        self._style_cache = OrderedDict()
        self._recent_styles: List[str] = []
        self._favorites: Set[str] = set()
        
        # Initialize quantum enhancer for advanced processing
        self.quantum_enhancer = QuantumSacredEnhancer()
        
        # Memory monitoring
        self._memory_usage = 0
        self._monitor_memory_usage()
    
    def _monitor_memory_usage(self):
        """Monitor memory usage to ensure efficient operation."""
        # In a real implementation, this would track actual memory usage
        # For now, we'll just track the number of cached styles
        self._memory_usage = len(self._style_cache)
        if self._memory_usage > self.cache_size * 0.9:
            logger.info(f"Memory usage high ({self._memory_usage}), optimizing cache")
            self._optimize_cache()
    
    def _optimize_cache(self):
        """Optimize the style cache by removing least recently used styles."""
        # Remove styles until we're below 80% of cache size
        target_size = int(self.cache_size * 0.8)
        while len(self._style_cache) > target_size:
            self._style_cache.popitem(last=False)  # Remove oldest item
        logger.info(f"Cache optimized, new size: {len(self._style_cache)}")
    
    @lru_cache(maxsize=10)
    def get_style_categories(self) -> List[str]:
        """Get available style categories (cached for efficiency)."""
        return self.style_factory.get_categories()
    
    def get_styles_in_category(self, category: str) -> List[str]:
        """Get available styles in a category."""
        return self.style_factory.get_styles_by_category(category)
    
    def get_recent_styles(self, limit: int = 10) -> List[str]:
        """Get recently used styles."""
        return self._recent_styles[:limit]
    
    def load_style(self, style_name: str) -> StyleParameters:
        """
        Load a style by name with memory-efficient caching.
        
        Args:
            style_name: Name of the style to load
            
        Returns:
            StyleParameters object
        """
        # Update recent styles list
        if style_name in self._recent_styles:
            self._recent_styles.remove(style_name)
        self._recent_styles.insert(0, style_name)
        
        # Check cache first
        if style_name in self._style_cache:
            self.current_style = self._style_cache[style_name]
            self.current_style_name = style_name
            # Move to end for LRU tracking
            self._style_cache.move_to_end(style_name)
            return self.current_style
        
        # Load from factory
        try:
            style = self.style_factory.get_style(style_name)
            
            # Update cache
            self._style_cache[style_name] = style
            if len(self._style_cache) > self.cache_size:
                self._style_cache.popitem(last=False)  # Remove oldest
                
            self.current_style = style
            self.current_style_name = style_name
            return style
            
        except Exception as e:
            logger.error(f"Error loading style {style_name}: {e}")
            raise
    
    def save_style(self, style_name: str, style: StyleParameters) -> bool:
        """
        Save a style to the database.
        
        Args:
            style_name: Name for the style
            style: StyleParameters object
            
        Returns:
            Success status
        """
        try:
            self.style_factory.save_style(style_name, style)
            
            # Update cache
            self._style_cache[style_name] = style
            
            # Update recent list
            if style_name in self._recent_styles:
                self._recent_styles.remove(style_name)
            self._recent_styles.insert(0, style_name)
            
            self._monitor_memory_usage()
            return True
        except Exception as e:
            logger.error(f"Error saving style {style_name}: {e}")
            return False
    
    def apply_quick_transform(self, transform_name: str) -> StyleParameters:
        """
        Apply a quick preset transformation to the current style.
        
        Args:
            transform_name: Name of the preset transformation
            
        Returns:
            Transformed StyleParameters object
        """
        if not self.current_style:
            raise ValueError("No current style loaded")
            
        if transform_name not in self.QUICK_TRANSFORMS:
            raise ValueError(f"Unknown transformation: {transform_name}")
            
        transform_params = self.QUICK_TRANSFORMS[transform_name]
        
        # Create a copy of the current style to modify
        new_style = self.style_factory.create_style(
            f"{self.current_style_name}_{transform_name}",
            base_style=self.current_style
        )
        
        # Apply the transformation parameters
        for param, value in transform_params.items():
            if param == "consciousness_level":
                self.set_consciousness_level(new_style, value)
            elif param == "apply_phi" and value:
                new_style = self.apply_phi_optimization(new_style)
            elif hasattr(new_style, param):
                current = getattr(new_style, param)
                if isinstance(current, (int, float)):
                    # Apply relative change for numeric values
                    setattr(new_style, param, max(0, min(1, current + value)))
        
        # Update the current style
        self.current_style = new_style
        self.current_style_name = f"{self.current_style_name}_{transform_name}"
        
        # Cache the result
        self._style_cache[self.current_style_name] = new_style
        self._monitor_memory_usage()
        
        return new_style
    
    def apply_phi_optimization(self, style: StyleParameters) -> StyleParameters:
        """
        Apply golden ratio (phi) optimization to style parameters.
        
        Args:
            style: StyleParameters object to optimize
            
        Returns:
            Optimized StyleParameters object
        """
        phi = calculate_phi_ratio()
        
        # Apply sacred geometry to key aesthetic parameters
        style.harmonic_ratio = apply_sacred_geometry(style.harmonic_ratio, phi)
        style.rhythm_complexity = apply_sacred_geometry(style.rhythm_complexity, phi)
        
        if hasattr(style, "reverb") and hasattr(style, "delay"):
            # Create phi relationship between reverb and delay
            avg = (style.reverb + style.delay) / 2
            style.reverb = avg * phi / (1 + phi)
            style.delay = avg / (1 + phi)
        
        # Apply to frequency distribution if available
        if hasattr(style, "bass_emphasis") and hasattr(style, "mid_emphasis") and hasattr(style, "treble_emphasis"):
            total = style.bass_emphasis + style.mid_emphasis + style.treble_emphasis
            style.bass_emphasis = total * phi / (1 + phi) / 2
            remaining = total - style.bass_emphasis
            style.mid_emphasis = remaining * phi / (1 + phi)
            style.treble_emphasis = remaining - style.mid_emphasis
        
        # Mark as phi-optimized
        style.phi_optimized = True
        
        return style
    
    def set_consciousness_level(self, style: StyleParameters, level: int) -> StyleParameters:
        """
        Adjust a style to target a specific consciousness level using
        Fibonacci-based sacred geometry principles.
        
        Args:
            style: StyleParameters object to modify
            level: Consciousness level (ideally a Fibonacci number)
            
        Returns:
            Modified StyleParameters object
        """
        # Ensure level is in our consciousness levels
        if level not in self.CONSCIOUSNESS_LEVELS:
            # Find closest Fibonacci number
            closest_level = min(self.CONSCIOUSNESS_LEVELS, key=lambda x: abs(x - level))
            logger.info(f"Adjusting consciousness level {level} to nearest Fibonacci: {closest_level}")
            level = closest_level
        
        # Store original level for reference
        original_level = getattr(style, "consciousness_level", 3)
        
        # Set the consciousness level
        style.consciousness_level = level
        
        # Apply quantum coherence enhancement based on level
        self.quantum_enhancer.enhance_coherence(style, level)
        
        # Adjust related parameters based on consciousness level
        style.mental_state_targeting = min(1.0, level / 21)
        style.harmonic_richness = min(1.0, level / 34 + 0.3)
        style.psychoacoustic_depth = min(1.0, level / 13)
        
        # Apply phi-based scaling to related parameters
        phi = calculate_phi_ratio()
        adjustment_factor = level / original_level
        
        style.reverb = min(1.0, style.reverb * (adjustment_factor * phi / 2))
        style.delay = min(1.0, style.delay * (adjustment_factor * phi / 2))
        
        return style
    
    def blend_styles(self, style_names: List[str], blend_name: str, 
                     weights: Optional[List[float]] = None,
                     phi_optimize: bool = True) -> StyleParameters:
        """
        Blend multiple styles together with optional phi-optimization.
        
        Args:
            style_names: List of style names to blend
            blend_name: Name for the resulting style
            weights: Optional weights for each style (will use phi-ratio if None)
            phi_optimize: Whether to apply phi-optimization to the result
            
        Returns:
            Blended StyleParameters object
        """
        # Load all styles
        styles = []
        for name in style_names:
            # Check cache first
            if name in self._style_cache:
                styles.append(self._style_cache[name])
            else:
                styles.append(self.style_factory.get_style(name))
        
        # Create blend
        blended_style = self.style_factory.blend_styles(
            styles, blend_name, weights=weights, use_phi=phi_optimize
        )
        
        # Update current style
        self.current_style = blended_style
        self.current_style_name = blend_name
        
        # Cache the result
        self._style_cache[blend_name] = blended_style
        self._monitor_memory_usage()
        
        return blended_style
    
    def get_parameter_suggestions(self, style: StyleParameters) -> Dict[str, float]:
        """
        Get suggestions for parameter improvements based on sacred geometry.
        
        Args:
            style: StyleParameters object to analyze
            
        Returns:
            Dictionary of parameter names and suggested values
        """
        suggestions = {}
        phi = calculate_phi_ratio()
        
        # Analyze key parameters for phi-ratio improvements
        for param_name in ["reverb", "delay", "saturation", "harmonic_richness"]:
            if hasattr(style, param_name):
                current_value = getattr(style, param_name)
                suggested_value = apply_sacred_geometry(current_value, phi)
                if abs(current_value - suggested_value) > 0.1:
                    suggestions[param_name] = suggested_value
        
        # Check consciousness level optimization
        current_level = getattr(style, "consciousness_level", 3)
        if current_level not in self.CONSCIOUSNESS_LEVELS:
            # Suggest nearest Fibonacci number
            suggested_level = min(self.CONSCIOUSNESS_LEVELS, key=lambda x: abs(x - current_level))
            suggestions["consciousness_level"] = suggested_level
        
        return suggestions
    
    def clear_cache(self):
        """Clear the style cache to free memory."""
        self._style_cache.clear()
        self._memory_usage = 0

import json
import logging
import os
import time
import tkinter as tk
from dataclasses import dataclass, field
from functools import lru_cache
from tkinter import messagebox, ttk
from typing import Any, Callable, Dict, List, Optional, Tuple

# Import StyleFactory to manage styles
from neural_beat_architect.core.style_factory import StyleFactory
from neural_processing.sacred_coherence import (apply_sacred_geometry,
                                                calculate_phi_ratio)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StyleManagerUI")


@dataclass
class UIState:
    """Maintains the current state of the UI to minimize redraws"""
    current_style_name: str = ""
    current_style: Optional[Dict[str, Any]] = None
    recently_used: List[str] = field(default_factory=list)
    favorites: List[str] = field(default_factory=list)
    consciousness_level: int = 3  # Default consciousness level
    parameters_expanded: bool = False
    advanced_mode: bool = False
    last_update_time: float = field(default_factory=time.time)
    phi_optimization_enabled: bool = True


class StyleManagerUI:
    """
    A memory-efficient UI for managing styles with phi-optimization and 
    consciousness level visualization.
    """
    # Constants for optimization
    MAX_RECENT_STYLES = 10
    STYLE_CACHE_SIZE = 20
    CONSCIOUSNESS_LEVELS = [1, 3, 5, 8, 13]  # Fibonacci sequence for consciousness
    
    def __init__(self, root=None, style_factory=None, config_path=None):
        """
        Initialize the StyleManagerUI with optional root window, style factory, and config path.
        
        Args:
            root: Tkinter root window or None for standalone
            style_factory: StyleFactory instance or None to create new
            config_path: Path to configuration file or None for default
        """
        self.root = root if root else tk.Tk()
        self.style_factory = style_factory if style_factory else StyleFactory()
        self.config_path = config_path if config_path else os.path.join(
            os.path.dirname(__file__), "../../config/style_manager_config.json"
        )
        
        # Initialize UI state
        self.state = UIState()
        
        # Initialize UI components dictionary to track created widgets
        self.components = {}
        
        # Memory management - track resource usage
        self.memory_usage = 0
        self.start_time = time.time()
        self.update_interval = 1.0  # seconds
        
        # Load configuration
        self._load_config()
        
        # Setup UI if root is provided
        if self.root is not None:
            self._setup_ui()
    
    def _load_config(self):
        """Load configuration from file, create default if not exists"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as config_file:
                    config = json.load(config_file)
                    
                # Apply config to state
                self.state.recently_used = config.get('recently_used', [])
                self.state.favorites = config.get('favorites', [])
                self.state.consciousness_level = config.get('consciousness_level', 3)
                self.state.phi_optimization_enabled = config.get('phi_optimization_enabled', True)
                self.state.advanced_mode = config.get('advanced_mode', False)
                
                logger.info(f"Configuration loaded from {self.config_path}")
            else:
                self._save_config()  # Create default config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    def _save_config(self):
        """Save current configuration to file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            config = {
                'recently_used': self.state.recently_used,
                'favorites': self.state.favorites,
                'consciousness_level': self.state.consciousness_level,
                'phi_optimization_enabled': self.state.phi_optimization_enabled,
                'advanced_mode': self.state.advanced_mode,
                'last_update': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(self.config_path, 'w') as config_file:
                json.dump(config, config_file, indent=4)
                
            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def _setup_ui(self):
        """Set up the main UI layout"""
        # Configure the root window
        self.root.title("Style Manager")
        self.root.geometry("900x600")
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a notebook for tabbed interface
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        styles_tab = ttk.Frame(notebook)
        transform_tab = ttk.Frame(notebook)
        consciousness_tab = ttk.Frame(notebook)
        advanced_tab = ttk.Frame(notebook)
        
        notebook.add(styles_tab, text="Styles")
        notebook.add(transform_tab, text="Transform")
        notebook.add(consciousness_tab, text="Consciousness")
        notebook.add(advanced_tab, text="Advanced")
        
        # Setup each tab
        self._setup_styles_tab(styles_tab)
        self._setup_transform_tab(transform_tab)
        self._setup_consciousness_tab(consciousness_tab)
        self._setup_advanced_tab(advanced_tab)
        
        # Status bar
        status_frame = ttk.Frame(main_frame, relief=tk.SUNKEN, padding=(2, 2))
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT)
        
        self.memory_var = tk.StringVar(value="Memory: 0 MB")
        memory_label = ttk.Label(status_frame, textvariable=self.memory_var)
        memory_label.pack(side=tk.RIGHT)
        
        # Schedule periodic UI updates
        self._schedule_update()
    
    def _setup_styles_tab(self, parent):
        """Set up the styles tab with style browsing and selection"""
        # Left side - style categories and list
        left_frame = ttk.Frame(parent, padding="5")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Categories header
        ttk.Label(left_frame, text="Categories", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        
        # Categories listbox
        categories_frame = ttk.Frame(left_frame)
        categories_frame.pack(fill=tk.X, pady=5)
        
        categories = self.style_factory.get_categories()
        category_var = tk.StringVar(value=categories)
        
        categories_lb = tk.Listbox(categories_frame, listvariable=category_var, height=5)
        categories_lb.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        categories_scrollbar = ttk.Scrollbar(categories_frame, orient="vertical", command=categories_lb.yview)
        categories_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        categories_lb.configure(yscrollcommand=categories_scrollbar.set)
        
        categories_lb.bind('<<ListboxSelect>>', self._on_category_selected)
        
        # Styles header
        ttk.Label(left_frame, text="Styles", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(10, 0))
        
        # Styles listbox
        styles_frame = ttk.Frame(left_frame)
        styles_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.styles_var = tk.StringVar()
        
        styles_lb = tk.Listbox(styles_frame, listvariable=self.styles_var, height=10)
        styles_lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        styles_scrollbar = ttk.Scrollbar(styles_frame, orient="vertical", command=styles_lb.yview)
        styles_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        styles_lb.configure(yscrollcommand=styles_scrollbar.set)
        
        styles_lb.bind('<<ListboxSelect>>', self._on_style_selected)
        
        # Search box
        search_frame = ttk.Frame(left_frame)
        search_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT)
        search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=search_var)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        search_var.trace_add("write", lambda *args: self._on_search(search_var.get()))
        
        # Right side - style details and preview
        right_frame = ttk.Frame(parent, padding="5")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Style preview frame
        preview_frame = ttk.LabelFrame(right_frame, text="Style Preview", padding="10")
        preview_frame.pack(fill=tk.BOTH, expand=True)
        
        # Style name and description
        self.style_name_var = tk.StringVar(value="No style selected")
        ttk.Label(preview_frame, textvariable=self.style_name_var, font=("Arial", 14, "bold")).pack(anchor=tk.W)
        
        self.style_desc_var = tk.StringVar(value="Select a style to view details")
        ttk.Label(preview_frame, textvariable=self.style_desc_var, wraplength=300).pack(anchor=tk.W, pady=5)
        
        # Preview canvas for visualization
        self.preview_canvas = tk.Canvas(preview_frame, bg="black", height=150)
        self.preview_canvas.pack(fill=tk.X, pady=10)
        
        # Quick action buttons
        actions_frame = ttk.Frame(preview_frame)
        actions_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(actions_frame, text="Load Style", command=self._on_load_style).pack(side=tk.LEFT, padx=2)
        ttk.Button(actions_frame, text="Preview", command=self._on_preview_style).pack(side=tk.LEFT, padx=2)
        ttk.Button(actions_frame, text="Add to Favorites", command=self._on_add_to_favorites).pack(side=tk.LEFT, padx=2)
        
        # Recently used styles
        recent_frame = ttk.LabelFrame(right_frame, text="Recently Used", padding="5")
        recent_frame.pack(fill=tk.X, pady=10)
        
        self.recent_styles_frame = ttk.Frame(recent_frame)
        self.recent_styles_frame.pack(fill=tk.X)
        
        self._update_recent_styles()
    
    def _setup_transform_tab(self, parent):
        """Set up the transform tab with quick transformations and phi-optimization"""
        # Quick transformation buttons
        quick_frame = ttk.LabelFrame(parent, text="Quick Transformations", padding="10")
        quick_frame.pack(fill=tk.X, padx=10, pady=10)
        
        transforms_frame = ttk.Frame(quick_frame)
        transforms_frame.pack(fill=tk.X)
        
        quick_transforms = [
            ("Energize", self._energize_transform),
            ("Chill", self._chill_transform),
            ("Deepen", self._deepen_transform),
            ("Brighten", self._brighten_transform),
            ("Intensify", self._intensify_transform),
            ("Soften", self._soften_transform),
            ("Randomize", self._randomize_transform)
        ]
        
        # Create a grid of transform buttons
        for i, (name, command) in enumerate(quick_transforms):
            ttk.Button(transforms_frame, text=name, command=command).grid(
                row=i//3, column=i%3, padx=5, pady=5, sticky="ew"
            )
        
        # Phi optimization frame
        phi_frame = ttk.LabelFrame(parent, text="Phi Optimization", padding="10")
        phi_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Phi-optimization toggle
        self.phi_var = tk.BooleanVar(value=self.state.phi_optimization_enabled)
        ttk.Checkbutton(
            phi_frame, 
            text="Enable Phi-Ratio Optimization",
            variable=self.phi_var,
            command=self._on_phi_toggle
        ).pack(anchor=tk.W)
        
        # Phi-optimization explanation
        ttk.Label(
            phi_frame, 
            text="Phi-optimization applies the golden ratio (1.618...) to parameter relationships, "
                 "creating more harmonically balanced and aesthetically pleasing results.",
            wraplength=600,
            justify=tk.LEFT
        ).pack(anchor=tk.W, pady=5)
        
        # Phi ratio visualization
        phi_canvas = tk.Canvas(phi_frame, width=600, height=80, bg="white")
        phi_canvas.pack(fill=tk.X, pady=5)
        
        # Draw golden spiral visualization
        self._draw_phi_visualization(phi_canvas)
        
        # Parameter adjustments with phi-optimization
        params_frame = ttk.LabelFrame(parent, text="Parameter Adjustments", padding="10")
        params_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Parameter categories in tabs
        param_notebook = ttk.Notebook(params_frame)
        param_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create parameter category tabs
        categories = ["Rhythm", "Harmony", "Timbre", "Dynamics", "Effects"]
        
        for category in categories:
            tab = ttk.Frame(param_notebook, padding="10")
            param_notebook.add(tab, text=category)
            
            # Add some parameter sliders for this category
            self._create_parameter_sliders(tab, category.lower())
    
    def _setup_consciousness_tab(self, parent):
        """Set up the consciousness level tab with visualization and adjustment"""
        # Consciousness

"""
StyleManagerUI - A simplified UI for style management with phi-optimization
and consciousness level visualization.
"""

import threading
import time
import tkinter as tk
from functools import lru_cache
from tkinter import messagebox, ttk
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.neural_beat_architect.core.architect import StyleParameters
from src.neural_beat_architect.core.style_factory import StyleFactory
from src.neural_processing.sacred_coherence import (apply_sacred_geometry,
                                                    calculate_phi_ratio)


class StyleManagerUI:
    """
    A memory-efficient UI for managing styles with phi-optimization and
    consciousness level visualization.
    """
    
    # Constants for UI configuration
    CONSCIOUSNESS_LEVELS = [1, 3, 5, 8, 13]  # Fibonacci sequence for consciousness levels
    QUICK_TRANSFORMS = {
        "Chill": {"intensity": -0.2, "density": -0.3, "energy": -0.25},
        "Energize": {"intensity": 0.3, "energy": 0.4, "tempo": 0.15},
        "Deep": {"consciousness_level": 2, "low_frequency_boost": 0.3},
        "Bright": {"high_frequency_boost": 0.3, "clarity": 0.25},
        "Sacred": {"sacred_ratio": 0.618, "harmonic_alignment": 0.5},
        "Random": {}  # Will be handled specially in the transform method
    }
    
    def __init__(self, master: tk.Tk, style_factory: StyleFactory):
        """
        Initialize the StyleManagerUI with a root window and StyleFactory.
        
        Args:
            master: The tkinter root window
            style_factory: The StyleFactory instance for style operations
        """
        self.master = master
        self.style_factory = style_factory
        
        # UI state variables
        self.current_style_name = tk.StringVar()
        self.current_style: Optional[StyleParameters] = None
        self.consciousness_level = tk.IntVar(value=3)  # Default to level 3
        self.parameter_vars: Dict[str, tk.Variable] = {}
        self.style_modified = False
        
        # Performance optimization
        self._style_cache = {}  # Simple cache for frequently accessed styles
        self._parameter_frames = {}  # Store parameter frames to avoid recreation
        
        # Initialize the UI
        self._setup_ui()
        
        # Load initial styles
        self._load_styles()

    def _setup_ui(self):
        """Set up the main UI components."""
        # Configure the main window
        self.master.title("Beat Production Beast - Style Manager")
        self.master.geometry("950x650")
        self.master.minsize(850, 600)
        
        # Create main frames
        self.main_frame = ttk.Frame(self.master, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Style selection
        self.left_frame = ttk.LabelFrame(self.main_frame, text="Styles", padding=5)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5)
        
        # Right panel - Style editing
        self.right_frame = ttk.LabelFrame(self.main_frame, text="Style Parameters", padding=5)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Style selection components
        self._setup_style_selection()
        
        # Quick transform buttons
        self._setup_quick_transforms()
        
        # Consciousness level slider
        self._setup_consciousness_controls()
        
        # Parameter editing components
        self._setup_parameter_editing()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.master, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Bind events
        self.master.bind("<Control-s>", lambda e: self._save_current_style())
        self.master.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_style_selection(self):
        """Set up the style selection components."""
        # Style categories
        categories_frame = ttk.Frame(self.left_frame)
        categories_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(categories_frame, text="Category:").pack(side=tk.LEFT, padx=5)
        self.category_var = tk.StringVar(value="All")
        self.category_combo = ttk.Combobox(categories_frame, textvariable=self.category_var)
        self.category_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.category_combo.bind("<<ComboboxSelected>>", lambda e: self._filter_styles())
        
        # Search box
        search_frame = ttk.Frame(self.left_frame)
        search_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT, padx=5)
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.search_var.trace_add("write", lambda *args: self._filter_styles())
        
        # Style list
        list_frame = ttk.Frame(self.left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.style_listbox = tk.Listbox(list_frame, height=15, selectmode=tk.SINGLE)
        self.style_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.style_listbox.bind("<<ListboxSelect>>", self._on_style_selected)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.style_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.style_listbox.config(yscrollcommand=scrollbar.set)
        
        # Style actions buttons
        actions_frame = ttk.Frame(self.left_frame)
        actions_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(actions_frame, text="New Style", command=self._create_new_style).pack(side=tk.LEFT, padx=2)
        ttk.Button(actions_frame, text="Duplicate", command=self._duplicate_style).pack(side=tk.LEFT, padx=2)
        ttk.Button(actions_frame, text="Delete", command=self._delete_style).pack(side=tk.LEFT, padx=2)
        ttk.Button(actions_frame, text="Save", command=self._save_current_style).pack(side=tk.LEFT, padx=2)

    def _setup_quick_transforms(self):
        """Set up the quick transform buttons."""
        transform_frame = ttk.LabelFrame(self.left_frame, text="Quick Transforms")
        transform_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create a grid of transform buttons
        row, col = 0, 0
        for name in self.QUICK_TRANSFORMS:
            ttk.Button(
                transform_frame, 
                text=name, 
                command=lambda n=name: self._apply_quick_transform(n)
            ).grid(row=row, column=col, padx=3, pady=3, sticky=tk.W)
            
            col += 1
            if col > 2:  # 3 buttons per row
                col = 0
                row += 1
                
        # Phi optimization toggle
        phi_frame = ttk.Frame(transform_frame)
        phi_frame.grid(row=row+1, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        self.phi_optimize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            phi_frame, 
            text="Use φ (Phi) optimization", 
            variable=self.phi_optimize_var
        ).pack(side=tk.LEFT)
        
        # Phi ratio display
        phi_value = calculate_phi_ratio()
        ttk.Label(
            phi_frame, 
            text=f"φ = {phi_value:.4f}", 
            foreground="purple"
        ).pack(side=tk.LEFT, padx=10)

    def _setup_consciousness_controls(self):
        """Set up controls for consciousness level."""
        consciousness_frame = ttk.LabelFrame(self.left_frame, text="Consciousness Level")
        consciousness_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Visualization canvas
        self.consciousness_canvas = tk.Canvas(consciousness_frame, height=60, background="black")
        self.consciousness_canvas.pack(fill=tk.X, padx=5, pady=5)
        
        # Slider for consciousness level
        slider_frame = ttk.Frame(consciousness_frame)
        slider_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(slider_frame, text="Level:").pack(side=tk.LEFT)
        
        consciousness_slider = ttk.Scale(
            slider_frame,
            from_=1,
            to=5,
            orient=tk.HORIZONTAL,
            variable=self.consciousness_level,
            command=self._update_consciousness_visualization
        )
        consciousness_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Level display
        self.level_label = ttk.Label(slider_frame, text="3")
        self.level_label.pack(side=tk.LEFT, padx=5)
        
        # Initialize visualization
        self._update_consciousness_visualization()

    def _setup_parameter_editing(self):
        """Set up the parameter editing section."""
        # Create notebook for parameter categories
        self.param_notebook = ttk.Notebook(self.right_frame)
        self.param_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add tabs for different parameter categories
        self.basic_frame = ttk.Frame(self.param_notebook)
        self.rhythm_frame = ttk.Frame(self.param_notebook)
        self.harmony_frame = ttk.Frame(self.param_notebook)
        self.special_frame = ttk.Frame(self.param_notebook)
        
        self.param_notebook.add(self.basic_frame, text="Basic")
        self.param_notebook.add(self.rhythm_frame, text="Rhythm")
        self.param_notebook.add(self.harmony_frame, text="Harmony")
        self.param_notebook.add(self.special_frame, text="Special")
        
        # Store frame references for parameter generation
        self._parameter_frames = {
            "basic": self.basic_frame,
            "rhythm": self.rhythm_frame,
            "harmony": self.harmony_frame,
            "special": self.special_frame
        }
        
        # Action buttons
        button_frame = ttk.Frame(self.right_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="Apply Changes", command=self._apply_parameter_changes).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset", command=self._reset_parameters).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Apply φ Optimization", 
                 command=self._apply_phi_optimization).pack(side=tk.LEFT, padx=5)

    def _load_styles(self):
        """Load styles from the StyleFactory."""
        self.status_var.set("Loading styles...")
        
        # This would be done in a background thread in a production app
        try:
            # Get categories for the dropdown
            categories = self.style_factory.get_style_categories()
            self.category_combo['values'] = ["All"] + sorted(categories)
            
            # Populate the style list
            self._filter_styles()
            
            # Select the first style if available
            if self.style_listbox.size() > 0:
                self.style_listbox.selection_set(0)
                self._on_style_selected(None)
                
            self.status_var.set("Styles loaded successfully")
        except Exception as e:
            self.status_var.set(f"Error loading styles: {str(e)}")
            messagebox.showerror("Error", f"Failed to load styles: {str(e)}")

    def _filter_styles(self):
        """Filter styles based on category and search text."""
        category = self.category_var.get()
        search_text = self.search_var.get().lower()
        
        # Clear the current list
        self.style_listbox.delete(0, tk.END)
        
        try:
            # Get styles from factory - this would have pagination in a real app
            if category == "All":
                styles = self.style_factory.get_all_styles()
            else:
                styles = self.style_factory.get_styles_by_category(category)
            
            # Filter by search text if needed
            if search_text:
                styles = [s for s in styles if search_text in s.lower()]
            
            # Add to listbox
            for style_name in sorted(styles):
                self.style_listbox.insert(tk.END, style_name)
                
        except Exception as e:
            self.status_var.set(f"Error filtering styles: {str(e)}")

    def _on_style_selected(self, event):
        """Handle style selection from the listbox."""
        if not self.style_listbox.curselection():
            return
            
        # Check if current style has unsaved changes
        if self.style_modified:
            save = messagebox.askyesnocancel(
                "Unsaved Changes", 
                "Current style has unsaved changes. Save before switching?"
            )
            if save is None:  # Cancel was pressed
                return
            if save:
                self._save_current_style()
        
        # Get the selected style name
        index = self.style_listbox.curselection()[0]
        style_name = self.style_listbox.get(index)
        
        # Update the current style
        self.current_style_name.set(style_name)
        
        # Load the style (use cache if available)
        if style_name in self._style_cache:
            self.current_style = self._style_cache[style_name]
        else:
            try:
                self.current

import json
import os
import threading
import time
import tkinter as tk
from collections import deque
from functools import lru_cache
from tkinter import messagebox, ttk
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.neural_beat_architect.core.architect import StyleParameters
# Import our style factory and related modules
from src.neural_beat_architect.core.style_factory import StyleFactory
from src.neural_processing.sacred_coherence import (apply_sacred_geometry,
                                                    calculate_phi_ratio)


class StyleManagerUI:
    """
    Memory-efficient UI for style management with one-click transformations.
    Provides intuitive access to the StyleFactory functionality.
    """
    
    def __init__(self, root, style_factory=None):
        """
        Initialize the StyleManagerUI.
        
        Args:
            root: The tkinter root window
            style_factory: The StyleFactory instance to use (or None to create a new one)
        """
        self.root = root
        self.style_factory = style_factory or StyleFactory()
        
        # UI state variables
        self.current_style = None
        self.selected_category = tk.StringVar()
        self.selected_style = tk.StringVar()
        self.consciousness_level = tk.IntVar(value=5)
        self.phi_optimization = tk.BooleanVar(value=True)
        self.show_advanced = tk.BooleanVar(value=False)
        
        # Memory management
        self._style_cache = {}  # LRU cache for styles
        self._max_cache_size = 20
        self._recent_styles = deque(maxlen=10)
        self._current_memory_usage = 0
        self._memory_threshold = 100 * 1024 * 1024  # 100 MB
        
        # Style parameters shown in UI
        self.parameters = {}
        self.parameter_vars = {}
        
        # Initialize the UI
        self._setup_ui()
        
        # Load initial styles
        self._lazy_load_styles()
    
    def _setup_ui(self):
        """Set up the main UI components."""
        self.root.title("Style Manager - BeatProductionBeast")
        self.root.geometry("1000x700")
        
        # Create main frame with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create two main sections
        left_frame = ttk.Frame(main_frame, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Left side: Style browser
        self._setup_style_browser(left_frame)
        
        # Right side: Style editor
        self._setup_style_editor(right_frame)
        
        # Status bar
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Memory usage indicator
        self.memory_var = tk.StringVar(value="Memory: 0 MB")
        memory_label = ttk.Label(status_frame, textvariable=self.memory_var)
        memory_label.pack(side=tk.RIGHT, padx=10, pady=5)
        
        # Start memory monitoring
        self._start_memory_monitoring()
    
    def _setup_style_browser(self, parent):
        """Set up the style browser section."""
        # Style categories
        category_frame = ttk.LabelFrame(parent, text="Categories")
        category_frame.pack(fill=tk.X, pady=(0, 10))
        
        categories = ["All", "Electronic", "Hip-Hop", "Ambient", "Experimental", 
                     "House", "Techno", "Lo-Fi", "Bass", "Cinematic", "World"]
        
        for category in categories:
            rb = ttk.Radiobutton(
                category_frame, 
                text=category,
                variable=self.selected_category,
                value=category,
                command=self._filter_styles_by_category
            )
            rb.pack(anchor=tk.W, padx=5, pady=2)
        
        self.selected_category.set("All")
        
        # Style list
        style_frame = ttk.LabelFrame(parent, text="Styles")
        style_frame.pack(fill=tk.BOTH, expand=True)
        
        # Search box
        search_frame = ttk.Frame(style_frame)
        search_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.search_var = tk.StringVar()
        self.search_var.trace("w", lambda *args: self._filter_styles_by_search())
        
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Style listbox
        self.style_listbox = tk.Listbox(style_frame, height=20)
        self.style_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.style_listbox.bind("<<ListboxSelect>>", self._on_style_selected)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(self.style_listbox, orient="vertical", 
                                command=self.style_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.style_listbox.config(yscrollcommand=scrollbar.set)
        
        # Recent styles
        recent_frame = ttk.LabelFrame(parent, text="Recent Styles")
        recent_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.recent_listbox = tk.Listbox(recent_frame, height=5)
        self.recent_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.recent_listbox.bind("<<ListboxSelect>>", self._on_recent_selected)
    
    def _setup_style_editor(self, parent):
        """Set up the style editor section."""
        # Top controls
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Current style name and save
        name_frame = ttk.Frame(control_frame)
        name_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Label(name_frame, text="Style Name:").pack(side=tk.LEFT, padx=(0, 5))
        self.style_name_var = tk.StringVar()
        name_entry = ttk.Entry(name_frame, textvariable=self.style_name_var, width=20)
        name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        save_btn = ttk.Button(name_frame, text="Save Style", command=self.save_style)
        save_btn.pack(side=tk.LEFT)
        
        # Quick transform buttons
        transform_frame = ttk.LabelFrame(parent, text="Quick Transformations")
        transform_frame.pack(fill=tk.X, pady=(0, 10))
        
        transforms = [
            ("Chill", {"energy": -0.2, "tempo": -0.1, "rhythm_complexity": -0.15}),
            ("Energize", {"energy": 0.2, "tempo": 0.15, "rhythm_complexity": 0.1}),
            ("Deep", {"bass_presence": 0.2, "sub_bass": 0.2, "consciousness_level": 2}),
            ("Bright", {"brightness": 0.2, "treble_presence": 0.15, "clarity": 0.1}),
            ("Complex", {"rhythm_complexity": 0.2, "melodic_complexity": 0.2, "harmony_complexity": 0.15}),
            ("Minimal", {"rhythm_complexity": -0.2, "melodic_complexity": -0.2, "space": 0.2}),
            ("Phi-Optimized", {"apply_phi": True})
        ]
        
        btn_frame1 = ttk.Frame(transform_frame)
        btn_frame1.pack(fill=tk.X, padx=5, pady=5)
        
        btn_frame2 = ttk.Frame(transform_frame)
        btn_frame2.pack(fill=tk.X, padx=5, pady=5)
        
        for i, (name, params) in enumerate(transforms):
            frame = btn_frame1 if i < 4 else btn_frame2
            btn = ttk.Button(
                frame, 
                text=name,
                command=lambda p=params: self.apply_quick_transform(p)
            )
            btn.pack(side=tk.LEFT, padx=5, expand=True)
        
        # Consciousness level
        consciousness_frame = ttk.LabelFrame(parent, text="Consciousness Level")
        consciousness_frame.pack(fill=tk.X, pady=(0, 10))
        
        scale_frame = ttk.Frame(consciousness_frame)
        scale_frame.pack(fill=tk.X, padx=10, pady=10)
        
        consciousness_scale = ttk.Scale(
            scale_frame,
            from_=1,
            to=13,
            orient="horizontal",
            variable=self.consciousness_level,
            command=self._update_consciousness_visualization
        )
        consciousness_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        self.consciousness_level_label = ttk.Label(scale_frame, text="5")
        self.consciousness_level_label.pack(side=tk.LEFT, padx=(0, 5))
        
        # Consciousness visualization canvas
        self.consciousness_canvas = tk.Canvas(consciousness_frame, height=80, bg="black")
        self.consciousness_canvas.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Parameter editor with notebook
        param_frame = ttk.LabelFrame(parent, text="Style Parameters")
        param_frame.pack(fill=tk.BOTH, expand=True)
        
        # Notebook for parameter categories
        self.param_notebook = ttk.Notebook(param_frame)
        self.param_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Parameter categories
        param_categories = {
            "Basic": ["tempo", "energy", "intensity", "mood", "brightness"],
            "Rhythm": ["rhythm_complexity", "percussion_density", "swing", "groove", "syncopation"],
            "Melody": ["melodic_complexity", "scale", "key", "melody_presence"],
            "Harmony": ["harmony_complexity", "chord_progression", "dissonance", "tonality"],
            "Texture": ["space", "density", "layering", "atmosphere", "texture_complexity"],
            "Frequencies": ["bass_presence", "mid_presence", "treble_presence", "sub_bass", "clarity"],
            "Consciousness": ["consciousness_level", "meditation_focus", "brain_entrainment", "sacred_geometry"]
        }
        
        # Create a tab for each parameter category
        for category, params in param_categories.items():
            tab = ttk.Frame(self.param_notebook)
            self.param_notebook.add(tab, text=category)
            
            # Create parameters in this category
            for i, param in enumerate(params):
                frame = ttk.Frame(tab)
                frame.pack(fill=tk.X, padx=10, pady=2)
                
                # Label
                label = ttk.Label(frame, text=param.replace("_", " ").title())
                label.pack(side=tk.LEFT, width=120, anchor=tk.W)
                
                # Value entry/slider
                self.parameter_vars[param] = tk.DoubleVar(value=0.5)
                
                scale = ttk.Scale(
                    frame,
                    from_=0.0,
                    to=1.0,
                    orient="horizontal",
                    variable=self.parameter_vars[param],
                    command=lambda val, p=param: self._update_parameter(p, val)
                )
                scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
                
                value_label = ttk.Label(frame, width=5, text="0.5")
                value_label.pack(side=tk.LEFT)
                
                # Store references
                self.parameters[param] = {
                    "var": self.parameter_vars[param],
                    "scale": scale,
                    "label": value_label
                }
        
        # Advanced options checkbox
        advanced_check = ttk.Checkbutton(
            param_frame,
            text="Show Advanced Options",
            variable=self.show_advanced,
            command=self._toggle_advanced_options
        )
        advanced_check.pack(anchor=tk.W, padx=10, pady=5)
        
        # Bottom buttons
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        apply_btn = ttk.Button(
            button_frame, 
            text="Apply Style", 
            command=self.apply_style
        )
        apply_btn.pack(side=tk.RIGHT, padx=5)
        
        blend_btn = ttk.Button(
            button_frame, 
            text="Blend Styles", 
            command=self._show_blend_dialog
        )
        blend_btn.pack(side=tk.RIGHT, padx=5)
        
        reset_btn = ttk.Button(
            button_frame, 
            text="Reset", 
            command=self._reset_parameters
        )
        reset_btn.pack(side=tk.RIGHT, padx=5)
    
    def load_style(self, style_name: str) -> Optional[StyleParameters]:
        """
        Load a style by name from the StyleFactory.
        
        Args:
            style_name: The name of the style to load
            
        Returns:
            The loaded style or None if not found
        """
        # Try to get from cache first
        if style_name in self._style_cache:
            self._recent_styles.append(style_name)
            self._update_recent_styles_list()
            return self._style_cache[style_name]
        
        # Load from the style factory
        try:
            style = self.style_factory.get_style(style_name)
            
            # Add to cache with memory management
            self._add_to_cache(style_name, style)
            
            # Add to recent styles
            self._recent_styles.append(style_name)
            self._update_recent_styles_list()
            
            self.status_var.set(f"Loaded style: {

"""
StyleManagerUI - Advanced UI for style management with memory optimization
Implements one-click style transformations with sacred geometry enhancements,
consciousness level visualization, and comprehensive style parameter management.
"""

import os
import json
import time
import random
import logging
from typing import Dict, List, Optional, Tuple, Callable, Any, Set, Union
from functools import lru_cache
from dataclasses import asdict, is_dataclass
from threading import Thread
from datetime import datetime

from src.neural_beat_architect.core.style_factory import StyleFactory
from src.neural_beat_architect.core.architect import StyleParameters
from src.neural_processing.sacred_coherence import calculate_phi_ratio, apply_sacred_geometry
from src.neural_processing.quantum_sacred_enhancer import enhance_consciousness, generate_sacred_pattern

# Configure logging
logger = logging.getLogger(__name__)

class MemoryOptimizer:
    """Memory usage optimization and tracking for UI components."""
    
    def __init__(self, threshold_mb: int = 100):
        self.threshold_mb = threshold_mb
        self.checkpoints: Dict[str, float] = {}
        self._monitoring_active = False
        self._monitor_thread = None
    
    def checkpoint(self, label: str) -> float:
        """Record memory usage at a checkpoint."""
        import psutil
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
        self.checkpoints[label] = memory_usage
        
        if memory_usage > self.threshold_mb:
            logger.warning(f"Memory usage high ({memory_usage:.2f}MB) at {label}")
        
        return memory_usage
    
    def start_monitoring(self, interval_seconds: int = 30):
        """Start background memory monitoring."""
        if self._monitoring_active:
            return
            
        self._monitoring_active = True
        
        def _monitor_memory():
            while self._monitoring_active:
                self.checkpoint(f"Periodic check {datetime.now().strftime('%H:%M:%S')}")
                time.sleep(interval_seconds)
        
        self._monitor_thread = Thread(target=_monitor_memory, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop background memory monitoring."""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1)
            self._monitor_thread = None
    
    def optimize(self, force_gc: bool = False):
        """Optimize memory usage."""
        if force_gc:
            import gc
            gc.collect()
        
        # Record memory after optimization
        return self.checkpoint("After optimization")

class StyleCache:
    """Memory-efficient cache for style objects with LRU eviction."""
    
    def __init__(self, max_size: int = 20):
        self.max_size = max_size
        self._cache: Dict[str, Tuple[StyleParameters, float]] = {}  # name -> (style, timestamp)
    
    def get(self, style_name: str) -> Optional[StyleParameters]:
        """Get a style from cache, updating its access time."""
        if style_name not in self._cache:
            return None
            
        style, _ = self._cache[style_name]
        self._cache[style_name] = (style, time.time())  # Update timestamp
        return style
    
    def put(self, style: StyleParameters) -> None:
        """Add a style to cache with eviction if needed."""
        # Evict least recently used if cache is full
        if len(self._cache) >= self.max_size:
            oldest_name = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_name]
        
        self._cache[style.style_name] = (style, time.time())
    
    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
    
    def get_all_names(self) -> List[str]:
        """Get all cached style names."""
        return list(self._cache.keys())

class ConsciousnessVisualizer:
    """Visualizes consciousness levels with sacred geometry patterns."""
    
    # Fibonacci sequence for consciousness levels
    FIBONACCI = [1, 2, 3, 5, 8, 13, 21]
    
    # Consciousness level descriptions
    LEVEL_DESCRIPTIONS = {
        1: "Basic awareness - fundamental beat recognition",
        2: "Dual awareness - rhythm and melody separation",
        3: "Expanded awareness - emotional resonance with music",
        5: "Harmonic awareness - recognition of sacred proportions",
        8: "Integrated awareness - unified perception of musical elements",
        13: "Transcendent awareness - accessing collective musical consciousness",
        21: "Cosmic awareness - complete harmonic integration with universal frequencies"
    }
    
    def __init__(self):
        self.current_level = 1
        self.pattern_cache = {}  # Cache generated patterns for memory efficiency
    
    def generate_visualization(self, level: int) -> Dict[str, Any]:
        """Generate visualization data for a consciousness level."""
        if level in self.pattern_cache:
            return self.pattern_cache[level]
            
        # Find nearest Fibonacci number for optimal sacred geometry
        fibonacci_level = min(self.FIBONACCI, key=lambda x: abs(x - level))
        
        # Generate sacred pattern based on consciousness level
        pattern = generate_sacred_pattern(fibonacci_level)
        
        # Description with deeper insight for higher levels
        description = self.LEVEL_DESCRIPTIONS.get(fibonacci_level, 
                        f"Level {level} - Custom consciousness configuration")
        
        # Frequency bands activated at this consciousness level
        frequency_bands = self._calculate_frequency_bands(fibonacci_level)
        
        result = {
            "level": level,
            "fibonacci_level": fibonacci_level,
            "pattern": pattern,
            "description": description,
            "frequency_bands": frequency_bands,
            "phi_ratio": calculate_phi_ratio(),
            "color_scheme": self._generate_color_scheme(fibonacci_level)
        }
        
        # Cache result for memory efficiency
        self.pattern_cache[level] = result
        
        return result
    
    def _calculate_frequency_bands(self, level: int) -> Dict[str, float]:
        """Calculate active frequency bands for consciousness level."""
        phi = calculate_phi_ratio()
        
        return {
            "delta": max(0.1, min(1.0, level / (21 * phi))),  # 0.5-4Hz
            "theta": max(0.1, min(1.0, level / (13 * phi))),  # 4-8Hz
            "alpha": max(0.1, min(1.0, level / (8 * phi))),   # 8-13Hz
            "beta": max(0.1, min(1.0, level / (5 * phi))),    # 13-30Hz
            "gamma": max(0.1, min(1.0, level / (3 * phi)))    # 30-100Hz
        }
    
    def _generate_color_scheme(self, level: int) -> Dict[str, str]:
        """Generate sacred geometry-aligned color scheme for visualization."""
        # Colors optimized for consciousness level visualization
        base_hues = {
            1: 240,  # Blue - basic
            2: 260,  # Indigo - dual
            3: 280,  # Purple - expanded
            5: 300,  # Magenta - harmonic
            8: 320,  # Pink - integrated
            13: 340, # Red - transcendent
            21: 360  # Full spectrum - cosmic
        }
        
        hue = base_hues.get(level, 280)
        
        # Generate colors with phi-based relationships
        phi = calculate_phi_ratio()
        
        return {
            "primary": f"hsl({hue}, 80%, 50%)",
            "secondary": f"hsl({(hue + 360/phi) % 360}, 70%, 60%)",
            "tertiary": f"hsl({(hue + 2*360/phi) % 360}, 90%, 40%)",
            "background": f"hsl({hue}, 20%, 10%)",
            "highlight": f"hsl({(hue + 180) % 360}, 100%, 75%)"
        }

class StyleManagerUI:
    """
    Advanced UI for style management with memory optimization.
    
    Features:
    - One-click style transformations with sacred geometry
    - Memory-efficient operation with smart caching
    - Consciousness level visualization
    - Comprehensive parameter management
    - Style blending with phi-optimized transformations
    """
    
    # Quick transform presets for one-click operations
    QUICK_TRANSFORMS = {
        "Chill": {
            "intensity": 0.6, 
            "tempo_modifier": 0.85, 
            "harmonic_density": 0.5, 
            "consciousness_level": 3, 
            "ambient_depth": 0.7
        },
        "Energize": {
            "intensity": 0.85, 
            "tempo_modifier": 1.2, 
            "harmonic_density": 0.7, 
            "consciousness_level": 5, 
            "clarity": 0.8
        },
        "Deep": {
            "intensity": 0.7, 
            "bass_boost": 0.8, 
            "resonance": 0.75, 
            "consciousness_level": 8, 
            "brainwave_entrainment": 0.6
        },
        "Bright": {
            "intensity": 0.75, 
            "high_freq_boost": 0.7, 
            "clarity": 0.8, 
            "consciousness_level": 5, 
            "spatial_depth": 0.6
        },
        "Sacred": {
            "phi_optimized": True, 
            "frequency_modulation": 0.7, 
            "consciousness_level": 13, 
            "harmonic_ratio": 1.618, 
            "mental_state_targeting": "transcendent"
        },
        "Quantum": {
            "phi_optimized": True,
            "consciousness_level": 21,
            "frequency_modulation": 0.9,
            "quantum_harmonic_layering": 0.8,
            "mental_state_targeting": "cosmic",
            "brainwave_entrainment": 0.9
        },
        "Random": {}  # Will be filled with random params on click
    }
    
    # Parameter groups for organized display
    PARAMETER_GROUPS = {
        "basic": [
            "style_name", "category", "tempo_modifier", "intensity", 
            "swing", "pattern_complexity", "consciousness_level"
        ],
        "rhythm": [
            "groove_type", "syncopation", "rhythm_complexity", "time_signature",
            "rhythmic_variation", "accent_strength", "polyrhythm_amount"
        ],
        "harmonic": [
            "harmonic_density", "harmonic_complexity", "key_signature", 
            "chord_complexity", "dissonance", "harmonic_variation",
            "chord_voicing_spread", "modal_interchange_amount"
        ],
        "textural": [
            "texture_density", "stereo_width", "spatial_depth", "clarity",
            "granularity", "roughness", "ambient_depth", "texture_evolution"
        ],
        "consciousness": [
            "consciousness_level", "frequency_modulation", "phi_optimized",
            "brainwave_entrainment", "mental_state_targeting", "harmonic_ratio",
            "sacred_frequency_alignment", "quantum_harmonic_layering"
        ]
    }
    
    def __init__(self, style_factory: Optional[StyleFactory] = None):
        """
        Initialize the StyleManagerUI.
        
        Args:
            style_factory: StyleFactory instance for style operations
        """
        # Initialize with dependency injection or create new instance
        self.style_factory = style_factory or StyleFactory()
        
        # Memory optimization
        self.memory_optimizer = MemoryOptimizer(threshold_mb=150)
        self.style_cache = StyleCache(max_size=30)
        
        # UI state
        self.current_style: Optional[StyleParameters] = None
        self.recent_styles: List[str] = []
        self.favorite_styles: Set[str] = set()
        self.expanded_groups: Set[str] = {"basic"}  # Track which parameter groups are expanded
        self.ui_mode = "standard"  # or "advanced", "minimal"
        
        # Component management
        self.components = {}
        self.lazy_loaded = set()
        
        # Initialize visualization components
        self.consciousness_visualizer = ConsciousnessVisualizer()
        
        # Setup and initialization
        self._initialize_component_registry()
        self._load_user_preferences()
        self._load_recent_and_favorite_styles()
        
        # Start monitoring memory usage
        self.memory_optimizer.start_monitoring(interval_seconds=60)
        
        logger.info("StyleManagerUI initialized")
    
    def _initialize_component_registry(self):
        """Initialize the component registry for lazy loading."""
        # Register UI components that will be lazy-loaded
        self.components = {
            # Core components that are always loaded
            "style_browser": {"loaded": False, "memory_impact": "low"},
            "parameter_panel": {"loaded": False, "memory_impact": "medium"},
            "transform_panel": {"loaded": False, "memory_impact": "low"},
            
            # Advanced components that are lazy-loaded
            "consciousness_display": {"loaded": False, "memory_impact": "high"},
            "sacred_geometry_viewer": {"loaded": False, "memory_impact": "high"},
            "style_analyzer": {"loaded": False, "memory_impact": "very_high"},
            "parameter_visualizer": {"loaded": False, "memory_impact": "high"},
            "style_comparison_tool": {"loaded": False, "memory_impact": "high"},
            "style_evolution_tracker": {"loaded": False, "memory_impact": "medium"},
            "batch_processor": {"loaded": False, "memory_impact": "very_high"},
        }
    
    def _load_component(self, component_name: str) -> bool:
        """
        Lazily load a UI component when needed.
        
        Args:
            component_name: Name of the component to load
            
        Returns:
            True if component was loaded successfully
        """
        if component_name not in self.components:
            logger.warning(f"Unknown component: {component_name}")
            return False
            
        if self.components[component_name]["loaded"]:
            return True
            
        # Check memory before loading
        current_memory = self.memory_optimizer.checkpoint(f"Before loading {component_name}")
        memory_impact = self.components[component_name]["memory_impact"]
        
        # Only load high-impact components if memory usage is reasonable
        if memory_impact in ["high", "very_high"] and current_memory > 200:
            logger.warning(f"Delaying loading of {component_name} due to current memory usage")
            return False
            
        # Simulate loading the component
        logger.info(f"Loading component: {component_name}")
        self.components[component_name]["loaded"] = True
        self.lazy_loaded.add(component_name)
        
        # Checkpoint memory after loading
        self.memory_optimizer.checkpoint(f"After loading

import os
import json
from typing import Dict, List, Optional, Callable, Any
from functools import lru_cache
import tkinter as tk
from tkinter import ttk, messagebox

from src.neural_beat_architect.core.style_factory import StyleFactory
from src.neural_beat_architect.core.architect import StyleParameters
from src.neural_processing.sacred_coherence import calculate_phi_ratio


class StyleManagerUI:
    """
    Memory-efficient UI for style management with one-click operations.
    Integrates with StyleFactory to provide easy access to style manipulation functions.
    """
    
    def __init__(self, root: tk.Tk, style_factory: StyleFactory = None):
        """
        Initialize the Style Manager UI.
        
        Args:
            root: The tkinter root window
            style_factory: Optional StyleFactory instance (will create one if None)
        """
        self.root = root
        self.style_factory = style_factory or StyleFactory()
        
        # UI state variables
        self.current_style: Optional[StyleParameters] = None
        self.selected_style_name: str = ""
        self.preview_active: bool = False
        self._parameter_vars: Dict[str, tk.Variable] = {}
        
        # Memory optimization - lazy loaded properties
        self._styles_cache: Dict[str, StyleParameters] = {}
        self._presets_cache: Dict[str, Dict[str, Any]] = {}
        
        # UI elements
        self.main_frame = ttk.Frame(self.root)
        self.styles_list = None
        self.parameter_frame = None
        self.quick_buttons_frame = None
        self.preview_button = None
        self.apply_button = None
        
        # Setup UI
        self._setup_ui()
        
        # Load initial data (with memory optimization)
        self._lazy_load_styles()
    
    def _setup_ui(self):
        """Set up the main UI layout with memory optimization in mind."""
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create a paned window for flexible resizing
        paned = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Style selection
        left_frame = ttk.Frame(paned, padding=5)
        paned.add(left_frame, weight=1)
        
        ttk.Label(left_frame, text="Available Styles").pack(anchor=tk.W)
        
        # Style search for quick filtering
        search_frame = ttk.Frame(left_frame)
        search_frame.pack(fill=tk.X, pady=5)
        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        self.search_var.trace("w", self._filter_styles)
        ttk.Entry(search_frame, textvariable=self.search_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Styles list with scrollbar
        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.styles_list = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, exportselection=False)
        self.styles_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.styles_list.yview)
        
        self.styles_list.bind('<<ListboxSelect>>', self._on_style_selected)
        
        # Recent styles (memory-efficient - only shows last 5)
        ttk.Label(left_frame, text="Recent Styles:").pack(anchor=tk.W, pady=(10, 5))
        self.recent_list = tk.Listbox(left_frame, height=5)
        self.recent_list.pack(fill=tk.X)
        self.recent_list.bind('<<ListboxSelect>>', self._on_recent_selected)
        
        # Right panel - Style parameters and operations
        right_frame = ttk.Frame(paned, padding=5)
        paned.add(right_frame, weight=2)
        
        # Style name and info
        info_frame = ttk.Frame(right_frame)
        info_frame.pack(fill=tk.X, pady=5)
        
        self.style_name_var = tk.StringVar(value="No style selected")
        ttk.Label(info_frame, textvariable=self.style_name_var, font=("", 12, "bold")).pack(side=tk.LEFT)
        
        # Quick transform buttons frame
        self.quick_buttons_frame = ttk.LabelFrame(right_frame, text="Quick Transforms")
        self.quick_buttons_frame.pack(fill=tk.X, pady=5)
        
        # Create quick transform buttons (in a grid for space efficiency)
        transforms = [
            ("Chill", "chill"), ("Energize", "energize"), 
            ("Deep", "deep"), ("Bright", "bright"),
            ("Phi-Optimized", "phi"), ("Random", "random")
        ]
        
        for i, (label, transform) in enumerate(transforms):
            row, col = divmod(i, 3)
            ttk.Button(
                self.quick_buttons_frame, 
                text=label,
                command=lambda t=transform: self._quick_transform(t)
            ).grid(row=row, column=col, padx=5, pady=5, sticky=tk.W+tk.E)
        
        # Parameters scrollable frame (loaded only when a style is selected - memory efficient)
        params_label_frame = ttk.LabelFrame(right_frame, text="Style Parameters")
        params_label_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create a canvas with scrollbar for parameters
        canvas = tk.Canvas(params_label_frame)
        scrollbar = ttk.Scrollbar(params_label_frame, orient=tk.VERTICAL, command=canvas.yview)
        self.parameter_frame = ttk.Frame(canvas)
        
        self.parameter_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.parameter_frame, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Action buttons at the bottom
        button_frame = ttk.Frame(right_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.preview_button = ttk.Button(
            button_frame, 
            text="Preview", 
            command=self._preview_current_style,
            state=tk.DISABLED
        )
        self.preview_button.pack(side=tk.LEFT, padx=5)
        
        self.apply_button = ttk.Button(
            button_frame, 
            text="Apply Style", 
            command=self._apply_current_style,
            state=tk.DISABLED
        )
        self.apply_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, 
            text="Save As New", 
            command=self._save_as_new_style
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, 
            text="Export", 
            command=self._export_style
        ).pack(side=tk.RIGHT, padx=5)
    
    @lru_cache(maxsize=16)  # Memory optimization - cache recent styles
    def _get_style(self, style_name: str) -> StyleParameters:
        """Get a style with memory-efficient caching."""
        if style_name not in self._styles_cache:
            self._styles_cache[style_name] = self.style_factory.get_style(style_name)
        return self._styles_cache[style_name]
    
    def _lazy_load_styles(self):
        """Load styles list with lazy loading for better memory usage."""
        # Clear the current list
        self.styles_list.delete(0, tk.END)
        
        # Get only the style names first (not the full objects)
        style_names = self.style_factory.get_all_style_names()
        
        # Sort alphabetically for ease of use
        style_names.sort()
        
        # Add to the listbox
        for name in style_names:
            self.styles_list.insert(tk.END, name)
    
    def _filter_styles(self, *args):
        """Filter the styles list based on search input."""
        search_term = self.search_var.get().lower()
        self.styles_list.delete(0, tk.END)
        
        style_names = self.style_factory.get_all_style_names()
        for name in style_names:
            if search_term in name.lower():
                self.styles_list.insert(tk.END, name)
    
    def _on_style_selected(self, event):
        """Handle style selection from the main list."""
        selection = self.styles_list.curselection()
        if not selection:
            return
        
        style_name = self.styles_list.get(selection[0])
        self._load_style(style_name)
        
        # Add to recent (avoiding duplicates)
        recent_styles = [self.recent_list.get(i) for i in range(self.recent_list.size())]
        if style_name in recent_styles:
            recent_styles.remove(style_name)
        recent_styles.insert(0, style_name)
        recent_styles = recent_styles[:5]  # Keep only 5 most recent
        
        # Update recent list
        self.recent_list.delete(0, tk.END)
        for name in recent_styles:
            self.recent_list.insert(tk.END, name)
    
    def _on_recent_selected(self, event):
        """Handle selection from recent styles list."""
        selection = self.recent_list.curselection()
        if not selection:
            return
        
        style_name = self.recent_list.get(selection[0])
        self._load_style(style_name)
    
    def _load_style(self, style_name: str):
        """Load a style and update the UI."""
        self.selected_style_name = style_name
        self.style_name_var.set(style_name)
        
        # Get style (cached)
        self.current_style = self._get_style(style_name)
        
        # Enable buttons
        self.preview_button.config(state=tk.NORMAL)
        self.apply_button.config(state=tk.NORMAL)
        
        # Clear parameter frame
        for widget in self.parameter_frame.winfo_children():
            widget.destroy()
        
        self._parameter_vars.clear()
        
        # Dynamically build parameter UI based on style parameters
        self._build_parameter_ui()
    
    def _build_parameter_ui(self):
        """Build the parameter sliders and controls dynamically."""
        # This dictionary categorizes parameters for better organization
        parameter_categories = {
            "Basic": ["bpm", "swing", "intensity", "complexity", "variation"],
            "Rhythm": ["kick_pattern", "snare_pattern", "hat_pattern", "percussion_complexity"],
            "Harmony": ["key", "scale", "chord_progression", "harmonic_rhythm"],
            "Consciousness": ["consciousness_level", "mental_state_targeting", "frequency_modulation"],
            "Advanced": ["sacred_geometry_ratio", "quantum_coherence"]
        }
        
        # Create notebook for categorized tabs
        notebook = ttk.Notebook(self.parameter_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tab for each category
        category_frames = {}
        for category in parameter_categories:
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=category)
            category_frames[category] = frame
        
        # Add an "All Parameters" tab
        all_params_frame = ttk.Frame(notebook)
        notebook.add(all_params_frame, text="All Parameters")
        
        # Get all parameters from the current style
        all_params = vars(self.current_style)
        
        # Process categorized parameters
        for category, param_names in parameter_categories.items():
            frame = category_frames[category]
            row = 0
            
            for param_name in param_names:
                if param_name in all_params:
                    self._add_parameter_control(frame, param_name, all_params[param_name], row)
                    row += 1
        
        # Process all parameters for the "All Parameters" tab
        row = 0
        for param_name, value in all_params.items():
            # Skip private attributes
            if param_name.startswith('_'):
                continue
                
            self._add_parameter_control(all_params_frame, param_name, value, row)
            row += 1
    
    def _add_parameter_control(self, parent, param_name, value, row):
        """Add a control for a specific parameter to the UI."""
        # Create a label with the parameter name
        ttk.Label(parent, text=param_name.replace('_', ' ').title()).grid(
            row=row, column=0, sticky=tk.W, padx=5, pady=2
        )
        
        # Create appropriate control based on parameter type
        if isinstance(value, (int, float)):
            # For numeric values, use a slider
            var = tk.DoubleVar(value=float(value))
            
            # Define slider range based on parameter
            min_val, max_val = 0, 100
            if 'bpm' in param_name:
                min_val, max_val = 60, 200
            elif 'level' in param_name:
                min_val, max_val = 1, 13
            elif 'ratio' in param_name:
                min_val, max_val = 0.1, 3.0
            elif 'probability' in param_name or 'intensity' in param_name:
                min_val, max_val = 0.0, 1.0
            
            slider = ttk.Scale(
                parent, 
                from_=min_val, 
                to=max_val, 
                variable=var,
                command=lambda v, name=param_name: self._update_parameter(name, float(v))
            )
            slider.grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
            
            # Add a numeric entry for precise control
            entry = ttk.Entry(parent, textvariable=var, width=8)
            entry.grid(row=row, column=2, padx=5, pady=2)
            entry

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import json
import os
from functools import lru_cache, partial
from typing import Dict, List, Optional, Tuple, Any
import time
import numpy as np
from PIL import Image, ImageTk

# Project imports
from src.neural_beat_architect.core.style_factory import StyleFactory
from src.neural_beat_architect.core.architect import StyleParameters, NeuralBeatArchitect
from src.neural_processing.sacred_coherence import calculate_phi_ratio, apply_sacred_geometry
from src.neural_processing.neural_enhancer import NeuralEnhancer
from src.neural_processing.quantum_sacred_enhancer import QuantumSacredEnhancer
from src.beat_generation.cli import AVAILABLE_STYLES
from src.audio_engine.frequency_modulator import FrequencyModulator

class StyleManagerUI:
    """
    Memory-efficient UI for managing beat styles with one-click access to StyleFactory 
    functionality and integration with all BeatProductionBeast components.
    
    Features:
    - Lazy loading of style data to minimize memory usage
    - One-click style application and preset transformations
    - Memory-optimized style previews with visualization
    - Quick access to common transformations with sacred geometry optimization
    - Expandable advanced options panel for detailed control
    - Integration with beat generation, neural processing, and audio engine
    - Style analysis and recommendation system
    """
    
    def __init__(self, root: Optional[tk.Tk] = None, parent_frame: Optional[tk.Frame] = None):
        """Initialize the StyleManagerUI with optional parent elements."""
        # Core components
        self.style_factory = StyleFactory()
        self.neural_architect = NeuralBeatArchitect()
        self.neural_enhancer = NeuralEnhancer()
        self.quantum_enhancer = QuantumSacredEnhancer()
        self.frequency_modulator = FrequencyModulator()
        
        # State variables
        self._current_style = None
        self._is_advanced_visible = False
        self._loaded_styles = {}  # Cache for loaded styles
        self._visualization_active = False
        self._batch_styles = []  # For batch processing
        self._processing_thread = None
        self._consciousness_level = 5  # Default consciousness level
        
        # Create root if not provided
        if root is None and parent_frame is None:
            self.root = tk.Tk()
            self.root.title("BeatProductionBeast Style Manager")
            self.root.geometry("950x700")
            icon_path = os.path.join(os.path.dirname(__file__), "../../resources/icons/app_icon.png")
            if os.path.exists(icon_path):
                icon = ImageTk.PhotoImage(Image.open(icon_path))
                self.root.iconphoto(True, icon)
            self.parent = self.root
        else:
            self.root = root
            self.parent = parent_frame or root
            
        # Create style
        self._create_style()
        
        # Create main elements
        self._create_ui_elements()
        
        # Set up the layout
        self._setup_layout()
        
        # Lazy load initial style categories
        self._lazy_load_style_categories()
        
        # Load recent styles history
        self._load_recent_styles()
    
    def _create_style(self):
        """Create custom styles for UI elements"""
        style = ttk.Style()
        
        # Configure button styles
        style.configure('Primary.TButton', 
                        background='#3584e4', 
                        foreground='white', 
                        padding=5, 
                        font=('Arial', 10, 'bold'))
        
        style.configure('Action.TButton', 
                        background='#26a269', 
                        foreground='white', 
                        padding=5)
        
        style.configure('Accent.TButton', 
                        background='#c061cb', 
                        foreground='white', 
                        padding=5)
        
        # Configure frame styles
        style.configure('Card.TFrame', 
                        background='#f6f5f4', 
                        relief='raised', 
                        borderwidth=1)
        
        # Configure label styles
        style.configure('Header.TLabel', 
                        font=('Arial', 12, 'bold'), 
                        padding=5)
        
        style.configure('Status.TLabel', 
                        font=('Arial', 9), 
                        padding=3, 
                        background='#e6e6e6')
    
    def _create_ui_elements(self):
        """Create all UI elements but defer heavy data loading."""
        # Create notebook for tabbed interface
        self.notebook = ttk.Notebook(self.parent)
        
        # Create main tabs
        self.style_tab = ttk.Frame(self.notebook)
        self.batch_tab = ttk.Frame(self.notebook)
        self.analysis_tab = ttk.Frame(self.notebook)
        self.visualization_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.style_tab, text="Style Editor")
        self.notebook.add(self.batch_tab, text="Batch Processing")
        self.notebook.add(self.analysis_tab, text="Style Analysis")
        self.notebook.add(self.visualization_tab, text="Visualization")
        
        # ====== Style Editor Tab ======
        # Create main frame with scrolling capability
        self.main_frame = ttk.Frame(self.style_tab)
        
        # Style categories section with memory-efficient scrolled view
        self.category_frame = ttk.LabelFrame(self.main_frame, text="Style Categories")
        self.category_canvas = tk.Canvas(self.category_frame, height=120)
        self.category_scrollbar = ttk.Scrollbar(self.category_frame, orient="horizontal", 
                                                command=self.category_canvas.xview)
        self.category_canvas.configure(xscrollcommand=self.category_scrollbar.set)
        
        self.category_content = ttk.Frame(self.category_canvas)
        self.category_content_id = self.category_canvas.create_window((0, 0), window=self.category_content, anchor="nw")
        self.category_buttons = []  # Will be populated on demand
        
        # Recent styles section
        self.recent_frame = ttk.LabelFrame(self.main_frame, text="Recent Styles")
        self.recent_styles_var = tk.StringVar()
        self.recent_styles_list = ttk.Combobox(self.recent_frame, textvariable=self.recent_styles_var, state="readonly")
        self.recent_styles_list.bind("<<ComboboxSelected>>", self._on_recent_selected)
        self.load_recent_button = ttk.Button(self.recent_frame, text="Load", command=self._load_selected_recent)
        
        # Current style section
        self.style_frame = ttk.LabelFrame(self.main_frame, text="Current Style")
        self.style_label = ttk.Label(self.style_frame, text="No style selected", style="Header.TLabel")
        self.preview_button = ttk.Button(self.style_frame, text="Preview", style="Action.TButton",
                                         command=self._preview_current_style)
        self.apply_button = ttk.Button(self.style_frame, text="Apply to Beat", style="Primary.TButton",
                                       command=self._apply_current_style)
        self.save_button = ttk.Button(self.style_frame, text="Save Style", 
                                      command=self._save_current_style)
        
        # Quick actions section with expandable panel
        self.quick_actions_frame = ttk.LabelFrame(self.main_frame, text="One-Click Transformations")
        
        # Create buttons for quick transformations using partial to avoid memory overhead
        self.quick_buttons = []
        quick_actions = [
            ("Energize", partial(self._quick_transform, "energize"), "Action.TButton"),
            ("Chill", partial(self._quick_transform, "chill"), "Action.TButton"),
            ("Transcend", partial(self._quick_transform, "transcend"), "Accent.TButton"),
            ("Deep Focus", partial(self._quick_transform, "deep_focus"), "Action.TButton"),
            ("Sacred Phi", partial(self._quick_transform, "phi_optimize"), "Accent.TButton"),
            ("Random", partial(self._quick_transform, "random"), "Primary.TButton"),
        ]
        
        for label, command, style in quick_actions:
            btn = ttk.Button(self.quick_actions_frame, text=label, command=command, style=style)
            self.quick_buttons.append(btn)
        
        # Advanced options toggle
        self.toggle_advanced_button = ttk.Button(
            self.main_frame, 
            text="Show Advanced Options ▼", 
            command=self._toggle_advanced_options
        )
        
        # Advanced options panel (initially hidden)
        self.advanced_frame = ttk.LabelFrame(self.main_frame, text="Advanced Options")
        
        # Consciousness level control with visualization
        self.consciousness_frame = ttk.Frame(self.advanced_frame)
        self.consciousness_label = ttk.Label(self.consciousness_frame, 
                                            text="Consciousness Level: 5 (Heart Chakra)")
        
        # Using Fibonacci sequence for consciousness levels
        fibonacci_levels = [1, 2, 3, 5, 8, 13]
        self.consciousness_scale = ttk.Scale(
            self.consciousness_frame, 
            from_=0, 
            to=len(fibonacci_levels)-1, 
            orient=tk.HORIZONTAL,
            command=lambda v: self._update_consciousness_level(fibonacci_levels[int(float(v))])
        )
        self.consciousness_scale.set(2)  # Default value (level 3)
        
        # Consciousness visualization canvas
        self.consciousness_canvas = tk.Canvas(self.consciousness_frame, height=30, width=300, 
                                             bg="#f0f0f0", highlightthickness=0)
        
        # Create sliders for common parameters
        self.param_sliders = {}
        common_params = [
            ("tempo", "Tempo (BPM)", 60, 180, 120),
            ("intensity", "Intensity", 0, 100, 50),
            ("complexity", "Complexity", 0, 100, 50),
            ("ambience", "Ambience", 0, 100, 20),
            ("harmony", "Harmonic Richness", 0, 100, 50),
            ("swing", "Swing", 0, 100, 25),
            ("resonance", "Resonance", 0, 100, 40),
            ("phase_shift", "Phase Shift", 0, 100, 0),
        ]
        
        self.params_frame = ttk.Frame(self.advanced_frame)
        
        for i, (param_id, label, min_val, max_val, default) in enumerate(common_params):
            row = i // 2
            col_start = (i % 2) * 3
            
            lbl = ttk.Label(self.params_frame, text=label)
            lbl.grid(row=row, column=col_start, sticky="w", padx=5, pady=2)
            
            slider = ttk.Scale(
                self.params_frame, 
                from_=min_val, 
                to=max_val, 
                orient=tk.HORIZONTAL,
                command=partial(self._update_parameter, param_id)
            )
            slider.set(default)
            slider.grid(row=row, column=col_start+1, sticky="ew", padx=5, pady=2)
            
            value_label = ttk.Label(self.params_frame, text=str(default))
            value_label.grid(row=row, column=col_start+2, sticky="w", padx=5, pady=2)
            
            self.param_sliders[param_id] = (slider, value_label)
        
        # Blend styles section
        self.blend_frame = ttk.LabelFrame(self.advanced_frame, text="Blend Styles")
        self.blend_style1 = ttk.Combobox(self.blend_frame, state="readonly")
        self.blend_style2 = ttk.Combobox(self.blend_frame, state="readonly")
        self.blend_ratio_scale = ttk.Scale(
            self.blend_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL
        )
        self.blend_ratio_label = ttk.Label(self.blend_frame, text="Blend Ratio: 50%")
        self.blend_ratio_scale.set(50)
        self.blend_ratio_scale.config(command=self._update_blend_ratio)
        
        self.phi_var = tk.BooleanVar(value=True)
        self.phi_check = ttk.Checkbutton(
            self.blend_frame,
            text="Use Phi-Optimization",
            variable=self.phi_var
        )
        
        self.blend_button = ttk.Button(
            self.blend_frame, 
            text="Blend Styles", 
            style="Primary.TButton",
            command=self._blend_styles
        )
        
        # Neural enhancement section
        self.neural_frame = ttk.LabelFrame(self.advanced_frame, text="Neural Enhancement")
        
        self.sacred_var = tk.BooleanVar(value=True)
        self.sacred_check = ttk.Checkbutton(
            self.neural_frame,
            text="Apply Sacred Geometry",
            variable=self.sacred_var
        )
        
        self.quantum_var = tk.BooleanVar(value=False)
        self.quantum_check = ttk.Checkbutton(
            self.neural_frame,
            text="Quantum Coherence Enhancement",
            variable=self.quantum_var
        )
        
        self.frequency_var = tk.BooleanVar(value=True)
        self.frequency_check = ttk.Checkbutton(
            self.neural_frame,
            text="Frequency Modulation",
            variable=self.frequency_var
        )
        
        self.enhance_button = ttk.Button(
            self.neural_frame,
            text="Apply Neural Enhancement",
            style="Accent.TButton",
            command=self._apply_neural_enhancement
        )
        
        # ====== Batch Processing Tab ======
        self.batch_main_frame = ttk.Frame(self.batch_tab)
        
        # Batch processing controls
        self.batch_controls_frame = ttk.LabelFrame(self.batch_main_frame, text="Batch Processing")
        
        # Style selection for batch processing
        self.batch_styles_frame = ttk.Frame(self.batch_controls_frame)
        self.batch_styles_list = tk.Listbox(self.batch_styles_frame, selectmode=tk.MULTIPLE, height=6)
        self.batch_styles_list = tk.Listbox(self.batch_styles_frame, selectmode=tk.MULTIPLE, height=6)
        self.batch_styles_scrollbar = ttk.Scrollbar(self.batch_styles_frame, orient=tk.VERTICAL,
                                                    command=self.batch_styles_list.yview)
        self.batch_styles_list.config(yscrollcommand=self.batch_styles_scrollbar.set)
        
        self.batch_add_button = ttk.Button(self.batch_controls_frame, text="Add Style",
                                           command=self._add_to_batch)
        self.batch_remove_button = ttk.Button(self.batch_controls_frame, text="Remove Selected",
                                             command=self._remove_from_batch)
        self.batch_process_button = ttk.Button(self.batch_controls_frame, text="Process Batch",
                                              command=self._process_batch,
                                              style="Primary.TButton")
        
        # Memory usage monitor
        self.memory_frame = ttk.LabelFrame(self.batch_main_frame, text="Memory Usage")
        self.memory_label = ttk.Label(self.memory_frame, text="Memory optimized: 0 MB used")
        self.memory_bar = ttk.Progressbar(self.memory_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.memory_bar['value'] = 10  # Initial value
        
        # ====== Style Analysis Tab ======
        self.analysis_main_frame = ttk.Frame(self.analysis_tab)
        
        # Style analysis controls
        self.analysis_controls_frame = ttk.LabelFrame(self.analysis_main_frame, text="Style Analysis")
        
        # Style selection for analysis
        self.analysis_style_var = tk.StringVar()
        self.analysis_style_combo = ttk.Combobox(self.analysis_controls_frame, 
                                                textvariable=self.analysis_style_var,
                                                state="readonly")
        self.analyze_button = ttk.Button(self.analysis_controls_frame, text="Analyze Style",
                                        command=self._analyze_style)
        
        # Analysis results area
        self.analysis_results_frame = ttk.LabelFrame(self.analysis_main_frame, text="Analysis Results")
        self.analysis_text = tk.Text(self.analysis_results_frame, height=10, width=50, wrap=tk.WORD)
        self.analysis_text.config(state=tk.DISABLED)
        self.analysis_scrollbar = ttk.Scrollbar(self.analysis_results_frame, orient=tk.VERTICAL,
                                               command=self.analysis_text.yview)
        self.analysis_text.config(yscrollcommand=self.analysis_scrollbar.set)
        
        # Style recommendation area
        self.recommendation_frame = ttk.LabelFrame(self.analysis_main_frame, text="Recommendations")
        self.recommendation_list = tk.Listbox(self.recommendation_frame, height=6)
        self.recommendation_scrollbar = ttk.Scrollbar(self.recommendation_frame, orient=tk.VERTICAL,
                                                      command=self.recommendation_list.yview)
        self.recommendation_list.config(yscrollcommand=self.recommendation_scrollbar.set)
        self.recommendation_list.bind("<Double-1>", self._on_recommendation_selected)
        
        # ====== Visualization Tab ======
        self.visualization_main_frame = ttk.Frame(self.visualization_tab)
        
        # Visualization controls
        self.visualization_controls_frame = ttk.LabelFrame(self.visualization_main_frame, 
                                                          text="Style Visualization")
        
        # Visualization canvas
        self.visualization_canvas_frame = ttk.Frame(self.visualization_main_frame)
        self.visualization_canvas = tk.Canvas(self.visualization_canvas_frame, 
                                             background="black", height=300, width=500)
        
        # Visualization options
        self.visualization_options_frame = ttk.Frame(self.visualization_controls_frame)
        
        # Visualization type selector
        self.visualization_type_var = tk.StringVar(value="sacred_geometry")
        self.visualization_type_label = ttk.Label(self.visualization_options_frame, 
                                                 text="Visualization Type:")
        self.visualization_type_combo = ttk.Combobox(self.visualization_options_frame, 
                                                    textvariable=self.visualization_type_var,
                                                    values=["sacred_geometry", "spectrum", "consciousness_field"],
                                                    state="readonly")
        self.visualization_type_combo.bind("<<ComboboxSelected>>", self._update_visualization)
        
        # Start/stop visualization
        self.visualization_running_var = tk.BooleanVar(value=False)
        self.visualization_toggle_button = ttk.Button(self.visualization_controls_frame,
                                                     text="Start Visualization",
                                                     command=self._toggle_visualization)
        
        # Status bar
        self.status_bar = ttk.Label(self.parent, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
    
    def _setup_layout(self):
        """Organize all UI elements in a clean, efficient layout"""
        # Set up main notebook tabs
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # ====== Style Editor Tab ======
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Style categories
        self.category_frame.pack(fill=tk.X, padx=5, pady=5)
        self.category_canvas.pack(fill=tk.X, expand=True, side=tk.TOP)
        self.category_scrollbar.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Recent styles
        self.recent_frame.pack(fill=tk.X, padx=5, pady=5)
        self.recent_styles_list.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        self.load_recent_button.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Current style
        self.style_frame.pack(fill=tk.X, padx=5, pady=5)
        self.style_label.pack(fill=tk.X, padx=5, pady=5)
        self.preview_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.apply_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.save_button.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Quick actions
        self.quick_actions_frame.pack(fill=tk.X, padx=5, pady=5)
        for i, btn in enumerate(self.quick_buttons):
            btn.pack(side=tk.LEFT, padx=5, pady=5, expand=True)
        
        # Advanced options toggle
        self.toggle_advanced_button.pack(fill=tk.X, padx=5, pady=5)
        
        # Advanced frame layout (initially hidden)
        # Consciousness level section
        self.consciousness_frame.pack(fill=tk.X, padx=5, pady=5)
        self.consciousness_label.pack(side=tk.TOP, anchor=tk.W, padx=5, pady=2)
        self.consciousness_scale.pack(fill=tk.X, padx=5, pady=2)
        self.consciousness_canvas.pack(fill=tk.X, padx=5, pady=2)
        
        # Parameters section
        self.params_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Blend section
        self.blend_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(self.blend_frame, text="Style 1:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.blend_style1.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Label(self.blend_frame, text="Style 2:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.blend_style2.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=2)
        self.blend_ratio_label.grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.blend_ratio_scale.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=2)
        self.phi_check.grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        self.blend_button.grid(row=4, column=0, columnspan=2, sticky=tk.E, padx=5, pady=5)
        
        # Neural enhancement section
        self.neural_frame.pack(fill=tk.X, padx=5, pady=5)
        self.sacred_check.pack(anchor=tk.W, padx=5, pady=2)
        self.quantum_check.pack(anchor=tk.W, padx=5, pady=2)
        self.frequency_check.pack(anchor=tk.W, padx=5, pady=2)
        self.enhance_button.pack(anchor=tk.E, padx=5, pady=5)
        
        # ====== Batch Processing Tab ======
        self.batch_main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Batch controls
        self.batch_controls_frame.pack(fill=tk.X, padx=5, pady=5)
        self.batch_styles_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.batch_styles_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.batch_styles_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Batch buttons
        button_frame = ttk.Frame(self.batch_controls_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        self.batch_add_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.batch_remove_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.batch_process_button.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Memory usage monitor
        self.memory_frame.pack(fill=tk.X, padx=5, pady=5)
        self.memory_label.pack(side=tk.TOP, padx=5, pady=2)
        self.memory_bar.pack(fill=tk.X, padx=5, pady=2)
        
        # ====== Style Analysis Tab ======
        self.analysis_main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Analysis controls
        self.analysis_controls_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(self.analysis_controls_frame, text="Style:").pack(side=tk.LEFT, padx=5, pady=5)
        self.analysis_style_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        self.analyze_button.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Analysis results
        self.analysis_results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.analysis_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.analysis_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Recommendations
        self.recommendation_frame.pack(fill=tk.X, padx=5, pady=5)
        self.recommendation_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.recommendation_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # ====== Visualization Tab ======
        self.visualization_main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Visualization controls
        self.visualization_controls_frame.pack(fill=tk.X, padx=5, pady=5)
        

class StyleManagerUI:
    """
    Memory-efficient UI integration for StyleFactory operations.
    
    Provides one-click access to style operations with lazy loading of resources.
    Includes presets for quick transformations, expandable advanced options,
    visual feedback for consciousness levels, and supports all style manipulation functions.
    """
    
    def __init__(self, master: tk.Widget, style_factory: Optional[StyleFactory] = None):
        """
        Initialize the StyleManagerUI with lazy loading capabilities.
        
        Args:
            master: The tkinter parent widget
            style_factory: Optional StyleFactory instance. If None, created on demand.
        """
        self.master = master
        self._style_factory = None
        self._style_factory_ref = None if style_factory is None else weakref.ref(style_factory)
        
        # Cache for style parameters (loaded on demand)
        self._style_cache = {}
        self._preset_cache = {}
        
        # UI state
        self.selected_style = tk.StringVar()
        self.consciousness_level = tk.IntVar(value=5)  # Default level 5
        self.advanced_mode = tk.BooleanVar(value=False)
        
        # Lazy-loaded resources
        self._resource_cache = {}
        self._thread_pool = []
        
        # Create the main frame
        self.main_frame = ttk.Frame(self.master)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Build the basic UI structure
        self._build_ui()

    @property
    def style_factory(self) -> StyleFactory:
        """
        Lazy load the StyleFactory to conserve memory.
        
        Returns:
            StyleFactory instance
        """
        if self._style_factory_ref is not None:
            factory = self._style_factory_ref()
            if factory is not None:
                return factory
        
        if self._style_factory is None:
            self._style_factory = StyleFactory()
            # Load only essential styles at first
            self._style_factory.load_database(load_all=False)
        
        return self._style_factory
    
    def _build_ui(self):
        """Build the core UI components with lazy loading features"""
        # Top area with style selection and one-click presets
        self._build_top_controls()
        
        # Middle area with consciousness level indicator and quick actions
        self._build_middle_controls()
        
        # Bottom area with expandable advanced options
        self._build_advanced_controls()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.main_frame, textvariable=self.status_var)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))
    
    def _build_top_controls(self):
        """Build top controls with style selection and one-click presets"""
        top_frame = ttk.Frame(self.main_frame)
        top_frame.pack(fill=tk.X, expand=False, pady=(0, 10))
        
        # Style selection with lazy loading of style names
        ttk.Label(top_frame, text="Style:").pack(side=tk.LEFT, padx=(0, 5))
        
        # Combobox with placeholder until loaded
        self.style_combo = ttk.Combobox(top_frame, textvariable=self.selected_style, 
                                        values=["Loading styles..."])
        self.style_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.style_combo.bind("<<ComboboxSelected>>", self._on_style_selected)
        
        # Load styles in background to keep UI responsive
        self._start_background_task(self._load_style_names)
        
        # Preset buttons frame
        preset_frame = ttk.LabelFrame(self.main_frame, text="Quick Presets")
        preset_frame.pack(fill=tk.X, expand=False, pady=(0, 10))
        
        # Grid for preset buttons (loads on demand)
        self.preset_grid = ttk.Frame(preset_frame)
        self.preset_grid.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Lazy load presets
        self._start_background_task(self._load_presets)
    
    def _build_middle_controls(self):
        """Build middle controls with consciousness level slider and visual indicator"""
        mid_frame = ttk.Frame(self.main_frame)
        mid_frame.pack(fill=tk.X, expand=False, pady=(0, 10))
        
        # Consciousness level controls
        ttk.Label(mid_frame, text="Consciousness Level:").pack(side=tk.LEFT, padx=(0, 5))
        
        # Visual indicator for consciousness level (uses canvas for efficiency)
        self.consciousness_canvas = tk.Canvas(mid_frame, width=50, height=20, 
                                             bg="white", highlightthickness=1)
        self.consciousness_canvas.pack(side=tk.LEFT, padx=(0, 5))
        self._update_consciousness_indicator()
        
        # Slider for consciousness level - Fibonacci numbers (1,3,5,8,13)
        self.consciousness_slider = ttk.Scale(mid_frame, from_=1, to=13, 
                                             variable=self.consciousness_level,
                                             command=self._on_consciousness_changed)
        self.consciousness_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Action buttons for common operations
        actions_frame = ttk.Frame(self.main_frame)
        actions_frame.pack(fill=tk.X, expand=False, pady=(0, 10))
        
        ttk.Button(actions_frame, text="Create", command=self._on_create_clicked).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions_frame, text="Transform", command=self._on_transform_clicked).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions_frame, text="Blend", command=self._on_blend_clicked).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions_frame, text="Apply", command=self._on_apply_clicked).pack(side=tk.LEFT, padx=5)
    
    def _build_advanced_controls(self):
        """Build advanced controls area that expands/collapses to save memory"""
        # Frame with toggle
        advanced_toggle_frame = ttk.Frame(self.main_frame)
        advanced_toggle_frame.pack(fill=tk.X)
        
        ttk.Checkbutton(advanced_toggle_frame, text="Advanced Options", 
                       variable=self.advanced_mode,
                       command=self._toggle_advanced_options).pack(side=tk.LEFT)
        
        # Advanced options container (hidden by default)
        self.advanced_frame = ttk.LabelFrame(self.main_frame, text="Advanced Style Parameters")
        # Not packed initially - will be shown when toggled
        
        # Parameters will be loaded on demand when advanced mode is activated
        self.param_frames = {}  # Will hold parameter widgets
    
    def _toggle_advanced_options(self):
        """Toggle visibility of advanced options with memory-efficient loading"""
        if self.advanced_mode.get():
            # Show advanced options and load parameters if needed
            if not self.param_frames:
                self._load_advanced_parameters()
            self.advanced_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        else:
            # Hide advanced options to save memory
            self.advanced_frame.pack_forget()
    
    def _load_advanced_parameters(self):
        """Load advanced parameters on demand to save memory"""
        # Clear existing frames
        for widget in self.advanced_frame.winfo_children():
            widget.destroy()
        
        self.param_frames = {}
        
        # Get style parameters from current selection
        style_name = self.selected_style.get()
        if not style_name or style_name == "Loading styles...":
            return
        
        # Get style parameters (with memory-efficient caching)
        style_params = self._get_style_params(style_name)
        if not style_params:
            return
        
        # Create parameter groups (notebook for efficiency)
        param_notebook = ttk.Notebook(self.advanced_frame)
        param_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create parameter groups
        groups = {
            "Basic": ["bpm", "swing", "intensity", "complexity"],
            "Rhythm": ["pattern_complexity", "rhythm_variation", "syncopation"],
            "Harmony": ["key", "scale", "chord_complexity", "tension"],
            "Sacred": ["consciousness_level", "sacred_ratio", "phi_alignment"]
        }
        
        # Create a tab for each group
        for group_name, param_list in groups.items():
            group_frame = ttk.Frame(param_notebook)
            param_notebook.add(group_frame, text=group_name)
            
            # Create widgets for each parameter in the group
            row = 0
            for param in param_list:
                if hasattr(style_params, param):
                    # Create frame for parameter
                    param_frame = ttk.Frame(group_frame)
                    param_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=2)
                    
                    # Label
                    ttk.Label(param_frame, text=f"{param}:").grid(row=0, column=0, sticky="w")
                    
                    # Value controls (appropriate for parameter type)
                    value = getattr(style_params, param)
                    
                    if isinstance(value, (int, float)):
                        var = tk.DoubleVar(value=value)
                        slider = ttk.Scale(param_frame, from_=0, to=100, variable=var)
                        slider.grid(row=0, column=1, sticky="ew", padx=5)
                        
                        # Store reference
                        self.param_frames[param] = (var, slider)
                    elif isinstance(value, str):
                        var = tk.StringVar(value=value)
                        entry = ttk.Entry(param_frame, textvariable=var)
                        entry.grid(row=0, column=1, sticky="ew", padx=5)
                        
                        # Store reference
                        self.param_frames[param] = (var, entry)
                    elif isinstance(value, bool):
                        var = tk.BooleanVar(value=value)
                        check = ttk.Checkbutton(param_frame, variable=var)
                        check.grid(row=0, column=1, sticky="w", padx=5)
                        
                        # Store reference
                        self.param_frames[param] = (var, check)
                    
                    row += 1
        
        # Add Apply button for advanced parameters
        apply_frame = ttk.Frame(self.advanced_frame)
        apply_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(apply_frame, text="Apply Advanced Settings", 
                  command=self._apply_advanced_settings).pack(side=tk.RIGHT)
    
    def _start_background_task(self, task_func: Callable, *args, **kwargs):
        """Start a background task with memory management"""
        # Clean up completed threads
        self._thread_pool = [t for t in self._thread_pool if t.is_alive()]
        
        # Create new thread
        thread = threading.Thread(target=task_func, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        self._thread_pool.append(thread)
    
    def _load_style_names(self):
        """Load style names in background to keep UI responsive"""
        try:
            # Simulate network or disk loading delay
            time.sleep(0.5)
            
            # Get all style names from factory
            style_names = self.style_factory.get_all_style_names()
            
            # Update UI on main thread
            self.master.after(0, lambda: self._update_style_combo(style_names))
        except Exception as e:
            # Update UI with error
            self.master.after(0, lambda: self._show_error(f"Error loading styles: {str(e)}"))
    
    def _update_style_combo(self, style_names: List[str]):
        """Update style combobox with loaded names"""
        self.style_combo['values'] = style_names
        if style_names:
            self.selected_style.set(style_names[0])
            self._on_style_selected(None)
    
    def _load_presets(self):
        """Load style presets in background"""
        try:
            # Define presets for one-click operations
            presets = [
                {"name": "Energize", "icon": "⚡", "transforms": {"intensity": 1.5, "bpm": 1.2}},
                {"name": "Chill", "icon": "❄️", "transforms": {"intensity": 0.7, "bpm": 0.8}},
                {"name": "Deep", "icon": "🌊", "transforms": {"consciousness_level": 8, "complexity": 1.3}},
                {"name": "Sacred", "icon": "✨", "transforms": {"phi_alignment": 1.0, "sacred_ratio": 1.618}},
                {"name": "Evolve", "icon": "🔄", "transforms": {"consciousness_level": "+3", "evolution": 1.0}},
                {"name": "Blend", "icon": "🔀", "action": "_show_blend_dialog"},
                {"name": "Transform", "icon": "🔮", "action": "_show_transform_dialog"},
                {"name": "Random", "icon": "🎲", "action": "_generate_random_style"}
            ]
            
            # Cache presets
            self._preset_cache = {p["name"]: p for p in presets}
            
            # Update UI on main thread
            self.master.after(0, lambda: self._build_preset_buttons(presets))
        except Exception as e:
            # Update UI with error
            self.master.after(0, lambda: self._show_error(f"Error loading presets: {str(e)}"))
    
    def _build_preset_buttons(self, presets: List[Dict]):
        """Build preset buttons in the grid"""
        # Clear existing buttons
        for widget in self.preset_grid.winfo_children():
            widget.destroy()
        
        # Create button grid - 4 columns
        cols = 

