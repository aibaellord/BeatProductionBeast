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

