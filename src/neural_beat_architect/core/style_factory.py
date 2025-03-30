import json
import os
import math
import random
import copy
from dataclasses import asdict, fields
from typing import Dict, List, Optional, Tuple, Union, Any

# Assuming StyleParameters is defined in architect.py
from .architect import StyleParameters


class StyleFactory:
    """
    Factory class for creating and managing music styles.
    Provides methods for loading, saving, creating, blending, and transforming styles.
    Supports a large database of styles across various musical genres.
    
    The StyleFactory enables:
    - Loading and saving style databases from/to JSON
    - Creating new styles with customized parameters
    - Blending multiple styles with golden ratio (phi) optimization
    - Transforming styles based on consciousness parameters
    - Retrieving styles by name, category, or other filters
    """

    def __init__(self, database_path: str = None):
        """
        Initialize the StyleFactory with an optional path to a style database.
        
        Args:
            database_path: Path to the JSON file containing style definitions
        """
        self.database_path = database_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
            "data", "style_database.json"
        )
        self.style_database: Dict[str, StyleParameters] = {}
        self.style_metadata: Dict[str, Dict] = {}
        
        # Try to load the database if it exists
        if os.path.exists(self.database_path):
            self.load_database()
        else:
            self._initialize_default_styles()
            self._initialize_metadata()
            self.save_database()

    def _initialize_default_styles(self) -> None:
        """
        Initialize a set of default styles if no database exists.
        These styles serve as a fallback and reference implementation.
        """
        # Add a small set of default styles
        self.style_database = {
            "hip-hop": StyleParameters(
                name="hip-hop",
                bpm_range=(85, 95),
                key_signature="C minor",
                instruments=["drums", "bass", "synth", "samples"],
                pattern_complexity=0.6,
                rhythm_signature="4/4",
                swing_amount=0.3,
                intensity=0.7,
                harmony_complexity=0.5,
                bass_prominence=0.8,
                percussion_density=0.7,
                atmospheric_elements=0.4,
                consciousness_level=5,
                frequency_profile="mid-heavy",
                sacred_geometry_enabled=True,
                mental_state_targeting="focused",
                quantum_entanglement_level=0.4
            ),
            "trap": StyleParameters(
                name="trap",
                bpm_range=(130, 160),
                key_signature="G minor",
                instruments=["808s", "hi-hats", "snares", "synth"],
                pattern_complexity=0.5,
                rhythm_signature="4/4",
                swing_amount=0.2,
                intensity=0.8,
                harmony_complexity=0.3,
                bass_prominence=0.9,
                percussion_density=0.8,
                atmospheric_elements=0.3,
                consciousness_level=4,
                frequency_profile="low-heavy",
                sacred_geometry_enabled=True,
                mental_state_targeting="energetic",
                quantum_entanglement_level=0.5
            ),
            "lo-fi": StyleParameters(
                name="lo-fi",
                bpm_range=(70, 90),
                key_signature="F major",
                instruments=["drums", "piano", "bass", "vinyl_noise"],
                pattern_complexity=0.4,
                rhythm_signature="4/4",
                swing_amount=0.4,
                intensity=0.3,
                harmony_complexity=0.7,
                bass_prominence=0.6,
                percussion_density=0.5,
                atmospheric_elements=0.8,
                consciousness_level=7,
                frequency_profile="warm-mids",
                sacred_geometry_enabled=True,
                mental_state_targeting="relaxed",
                quantum_entanglement_level=0.6
            ),
            "ambient": StyleParameters(
                name="ambient",
                bpm_range=(60, 85),
                key_signature="D minor",
                instruments=["pads", "synths", "field_recordings", "textures"],
                pattern_complexity=0.3,
                rhythm_signature="free-time",
                swing_amount=0.1,
                intensity=0.2,
                harmony_complexity=0.8,
                bass_prominence=0.5,
                percussion_density=0.2,
                atmospheric_elements=0.9,
                consciousness_level=8,
                frequency_profile="full-spectrum",
                sacred_geometry_enabled=True,
                mental_state_targeting="meditative",
                quantum_entanglement_level=0.8
            ),
            "techno": StyleParameters(
                name="techno",
                bpm_range=(125, 145),
                key_signature="A minor",
                instruments=["kick", "synth", "hi-hats", "claps"],
                pattern_complexity=0.7,
                rhythm_signature="4/4",
                swing_amount=0.1,
                intensity=0.9,
                harmony_complexity=0.4,
                bass_prominence=0.7,
                percussion_density=0.9,
                atmospheric_elements=0.5,
                consciousness_level=6,
                frequency_profile="transient-heavy",
                sacred_geometry_enabled=True,
                mental_state_targeting="hypnotic",
                quantum_entanglement_level=0.7
            )
        }

    def _initialize_metadata(self) -> None:
        """
        Initialize metadata for styles in the database.
        Metadata includes category, description, moods, related styles,
        creation date, version, and tags.
        """
        for style_name, style in self.style_database.items():
            self.style_metadata[style_name] = {
                "category": self._determine_category(style),
                "description": self._generate_description(style),
                "moods": self._determine_moods(style),
                "related_styles": self._find_related_styles(style_name),
                "creation_date": "2023-08-15",
                "version": "1.0",
                "tags": self._generate_tags(style),
                "consciousness_alignment": self._calculate_consciousness_alignment(style),
                "recommended_use_cases": self._determine_use_cases(style),
                "sacred_geometry_structures": self._identify_sacred_structures(style) if style.sacred_geometry_enabled else []
            }

    def _determine_category(self, style: StyleParameters) -> str:
        """
        Determine the category for a style based on its parameters.
        
        Args:
            style: The style to categorize
            
        Returns:
            The category as a string
        """
        # Sophisticated categorization logic based on multiple parameters
        if style.bpm_range[0] < 70:
            base_category = "Downtempo"
        elif style.bpm_range[0] >= 120:
            base_category = "Uptempo"
        else:
            base_category = "Midtempo"
            
        # Refine category based on other parameters
        if style.atmospheric_elements > 0.7:
            return f"Atmospheric {base_category}"
        elif style.percussion_density > 0.8:
            return f"Rhythmic {base_category}"
        elif style.harmony_complexity > 0.7:
            return f"Harmonic {base_category}"
        elif style.bass_prominence > 0.8:
            return f"Bass-driven {base_category}"
        
        return base_category

    def _generate_description(self, style: StyleParameters) -> str:
        """
        Generate a description for a style based on its parameters.
        
        Args:
            style: The style to describe
            
        Returns:
            A description as a string
        """
        # Generate detailed, informative description
        description = f"{style.name.title()} style featuring {style.bpm_range[0]}-{style.bpm_range[1]} BPM, "
        
        # Add key information
        description += f"typically in {style.key_signature}, "
        
        # Add rhythm information
        description += f"with {style.rhythm_signature} rhythm"
        if style.swing_amount > 0.3:
            description += f" and {int(style.swing_amount * 100)}% swing. "
        else:
            description += ". "
            
        # Add instrumentation
        description += f"Key instruments include {', '.join(style.instruments[:3])}"
        if len(style.instruments) > 3:
            description += f" and {len(style.instruments) - 3} others. "
        else:
            description += ". "
            
        # Add consciousness and sacred geometry information
        description += f"Optimized for consciousness level {style.consciousness_level}, "
        description += f"targeting {style.mental_state_targeting} mental states"
        
        if style.sacred_geometry_enabled:
            description += ", with sacred geometry principles applied for enhanced coherence."
        else:
            description += "."
            
        return description

    def _determine_moods(self, style: StyleParameters) -> List[str]:
        """
        Determine appropriate moods for a style based on its parameters.
        
        Args:
            style: The style to determine moods for
            
        Returns:
            A list of mood strings
        """
        moods = []
        
        # Determine moods based on comprehensive parameter analysis
        # Intensity-based moods
        if style.intensity > 0.8:
            moods.extend(["energetic", "intense", "powerful"])
        elif style.intensity > 0.5:
            moods.extend(["dynamic", "engaging", "active"])
        elif style.intensity > 0.3:
            moods.extend(["moderate", "balanced", "composed"])
        else:
            moods.extend(["calm", "serene", "gentle"])
            
        # Harmony-based moods
        if "minor" in style.key_signature.lower():
            if style.harmony_complexity > 0.7:
                moods.extend(["introspective", "sophisticated", "thoughtful"])
            else:
                moods.extend(["melancholic", "serious", "reflective"])
        else:
            if style.harmony_complexity > 0.7:
                moods.extend(["uplifting", "complex", "bright"])
            else:
                moods.extend(["cheerful", "straightforward", "light"])
                
        # Atmospheric elements
        if style.atmospheric_elements > 0.7:
            moods.extend(["dreamy", "spacious", "atmospheric"])
            
        # Consciousness level
        if style.consciousness_level > 7:
            moods.extend(["transcendent", "enlightened", "awakened"])
        elif style.consciousness_level > 5:
            moods.extend(["conscious", "aware", "mindful"])
            
        # Remove duplicates and limit to 7 moods
        return list(set(moods))[:7]

    def _find_related_styles(self, style_name: str) -> List[str]:
        """
        Find styles that are related to the given style based on parameters.
        
        Args:
            style_name: The name of the style to find related styles for
            
        Returns:
            A list of related style names
        """
        # Comprehensive style relationships
        related_map = {
            "hip-hop": ["trap", "lo-fi", "boom-bap", "jazz-hop", "instrumental-hip-hop"],
            "trap": ["hip-hop", "drill", "cloud-rap", "mumble-rap", "phonk"],
            "lo-fi": ["hip-hop", "jazz-hop", "chillhop", "ambient", "downtempo"],
            "house": ["techno", "deep-house", "electro", "progressive-house", "tech-house"],
            "techno": ["house", "minimal", "industrial", "acid", "hard-techno"],
            "ambient": ["chillout", "lo-fi", "downtempo", "drone", "space-music"],
            "drum-and-bass": ["jungle", "breakbeat", "liquid", "neurofunk", "drumstep"],
            "dubstep": ["future-bass", "trap", "riddim", "brostep", "tearout"],
            "jazz": ["jazz-hop", "fusion", "nu-jazz", "lounge", "bossa-nova"],
            "classical": ["neo-classical", "contemporary-classical", "cinematic", "orchestral", "chamber"],
            "rock": ["indie-rock", "alternative", "post-rock", "psychedelic", "progressive"],
            "metal": ["progressive-metal", "djent", "black-metal", "doom-metal", "industrial-metal"],
            "folk": ["acoustic", "singer-songwriter", "celtic", "traditional", "americana"],
            "reggae": ["dub", "dancehall", "raggamuffin", "ska", "rocksteady"],
            "funk": ["disco", "soul", "r&b", "boogie", "electro-funk"],
            "world": ["afrobeat", "latin", "balkan", "indian-classical", "arabic"]
        }
        
        # Look for direct mappings in the related map
        direct_relations = related_map.get(style_name, [])
        
        # If the style isn't in our map, try to find related styles by parameter similarity
        if not direct_relations and style_name in self.style_database:
            current_style = self.style_database[style_name]
            related = []
            
            # Compare with other styles to find similarities
            for other_name, other_style in self.style_database.items():
                if other_name != style_name:
                    # Calculate similarity (simple implementation)
                    similarity = 0
                    
                    # BPM similarity
                    bpm_diff = abs(current_style.bpm_range[0] - other_style.bpm_range[0])
                    if bpm_diff < 20:
                        similarity += 1
                        
                    # Key signature similarity
                    if current_style.key_signature == other_style.key_signature:
                        similarity += 1
                        
                    # Instrumentation similarity
                    common_instruments = set(current_style.instruments) & set(other_style.instruments)
                    if len(common_instruments) >= 2:
                        similarity += 1
                        
                    # Consciousness level similarity
                    if abs(current_style.consciousness_level - other_style.consciousness_level) <= 1:
                        similarity += 1
                        
                    # If there's significant similarity, add to related styles
                    if similarity >= 2:
                        related.append(other_name)
                        
            return related[:5]  # Limit to 5 related styles
            
        return direct_relations

    def _generate_tags(self, style: StyleParameters) -> List[str]:
        """
        Generate appropriate tags for a style based on its parameters.
        
        Args:
            style: The style to generate tags for
            
        Returns:
            A list of tag strings
        """
        tags = [style.

import os
import json
import math
import random
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import asdict
from pathlib import Path
import numpy as np

from src.neural_beat_architect.core.architect import StyleParameters
from src.neural_processing.sacred_coherence import calculate_phi_ratio, apply_sacred_geometry

logger = logging.getLogger(__name__)

class StyleFactory:
    """
    A comprehensive factory class for creating, managing, and manipulating
    beat production styles. Supports over 100 styles with capabilities for
    loading, saving, blending, and transforming styles.
    """

    def __init__(self, database_path: Optional[str] = None):
        """
        Initialize the StyleFactory with a database of styles.
        
        Args:
            database_path: Path to the JSON style database. If None, uses default path.
        """
        self.styles: Dict[str, StyleParameters] = {}
        self.categories: Dict[str, List[str]] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        
        # Default path is in the data directory
        if database_path is None:
            database_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "data",
                "style_database.json"
            )
        
        self.database_path = database_path
        self._initialize_metadata()
        
        # Load the database if it exists
        if os.path.exists(database_path):
            self.load_database(database_path)
        else:
            logger.warning(f"Style database not found at {database_path}. Starting with empty database.")
            self._initialize_default_styles()

    def _initialize_metadata(self):
        """Initialize the metadata structure for styles categorization and filtering."""
        self.metadata = {
            "version": "1.0.0",
            "total_styles": 0,
            "categories": {},
            "influences": {},
            "eras": {},
            "mood_mappings": {},
            "consciousness_levels": {},
            "style_properties": {}
        }

    def _initialize_default_styles(self):
        """Initialize a set of default styles if no database is found."""
        # Basic electronic styles
        self.create_style(
            name="deep_house",
            display_name="Deep House",
            category="electronic",
            bpm=124,
            swing=0.2,
            complexity=0.7,
            intensity=0.65,
            texture_density=0.6,
            harmonic_complexity=0.5,
            rhythmic_density=0.65,
            bass_presence=0.8,
            brightness=0.6,
            warmth=0.7,
            mental_state_targeting=True,
            consciousness_level=7,
            frequency_optimization=True,
            sacred_geometry_enabled=True,
            description="Deep, soulful electronic music with jazz and soul influences."
        )
        
        self.create_style(
            name="ambient_chill",
            display_name="Ambient Chill",
            category="ambient",
            bpm=85,
            swing=0.1,
            complexity=0.4,
            intensity=0.3,
            texture_density=0.7,
            harmonic_complexity=0.6,
            rhythmic_density=0.3,
            bass_presence=0.5,
            brightness=0.5,
            warmth=0.8,
            mental_state_targeting=True,
            consciousness_level=8,
            frequency_optimization=True,
            sacred_geometry_enabled=True,
            description="Calm atmospheric sounds with minimal beats for deep relaxation."
        )
        
        self.create_style(
            name="trap_soul",
            display_name="Trap Soul",
            category="hip-hop",
            bpm=75,
            swing=0.4,
            complexity=0.6,
            intensity=0.7,
            texture_density=0.5,
            harmonic_complexity=0.6,
            rhythmic_density=0.75,
            bass_presence=0.9,
            brightness=0.4,
            warmth=0.6,
            mental_state_targeting=True,
            consciousness_level=6,
            frequency_optimization=True,
            sacred_geometry_enabled=True,
            description="Blend of trap beats with soul and R&B influences."
        )

    def load_database(self, path: Optional[str] = None) -> bool:
        """
        Load styles from a JSON database file.
        
        Args:
            path: Path to the JSON database file. If None, uses the default path.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            file_path = path or self.database_path
            
            if not os.path.exists(file_path):
                logger.error(f"Database file not found: {file_path}")
                return False
                
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Load metadata
            if "metadata" in data:
                self.metadata = data["metadata"]
            
            # Load styles
            self.styles = {}
            self.categories = {}
            
            for style_name, style_data in data.get("styles", {}).items():
                # Convert dictionary back to StyleParameters object
                style = StyleParameters(**style_data)
                self.styles[style_name] = style
                
                # Update categories
                category = style_data.get("category", "uncategorized")
                if category not in self.categories:
                    self.categories[category] = []
                self.categories[category].append(style_name)
            
            logger.info(f"Loaded {len(self.styles)} styles from database.")
            return True
            
        except Exception as e:
            logger.error(f"Error loading style database: {str(e)}")
            return False

    def save_database(self, path: Optional[str] = None) -> bool:
        """
        Save the current styles to a JSON database file.
        
        Args:
            path: Path to save the JSON database. If None, uses the default path.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            file_path = path or self.database_path
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Update metadata
            self.metadata["total_styles"] = len(self.styles)
            self.metadata["categories"] = {category: len(styles) for category, styles in self.categories.items()}
            
            # Prepare data for saving
            data = {
                "metadata": self.metadata,
                "styles": {name: asdict(style) for name, style in self.styles.items()}
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(self.styles)} styles to database at {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving style database: {str(e)}")
            return False

    def create_style(self, name: str, **kwargs) -> StyleParameters:
        """
        Create a new style with the given parameters.
        
        Args:
            name: Unique identifier for the style
            **kwargs: Parameters for the StyleParameters class
            
        Returns:
            The created StyleParameters object
        """
        # Ensure name is valid
        if not name or not isinstance(name, str):
            raise ValueError("Style name must be a non-empty string")
        
        # Create the style
        style = StyleParameters(**kwargs)
        
        # Add to styles dictionary
        self.styles[name] = style
        
        # Add to categories
        category = kwargs.get("category", "uncategorized")
        if category not in self.categories:
            self.categories[category] = []
        if name not in self.categories[category]:
            self.categories[category].append(name)
        
        # Add metadata
        self.metadata["style_properties"][name] = {
            "display_name": kwargs.get("display_name", name),
            "description": kwargs.get("description", ""),
            "category": category,
            "consciousness_level": kwargs.get("consciousness_level", 5),
            "creation_date": kwargs.get("creation_date", ""),
            "influences": kwargs.get("influences", [])
        }
        
        return style

    def get_style(self, name: str) -> Optional[StyleParameters]:
        """
        Get a style by name.
        
        Args:
            name: Name of the style to retrieve
            
        Returns:
            StyleParameters object if found, None otherwise
        """
        return self.styles.get(name)

    def get_styles_by_category(self, category: str) -> List[Tuple[str, StyleParameters]]:
        """
        Get all styles in a specific category.
        
        Args:
            category: Category to filter by
            
        Returns:
            List of (name, style) tuples for the given category
        """
        if category not in self.categories:
            return []
            
        return [(name, self.styles[name]) for name in self.categories[category] 
                if name in self.styles]

    def get_all_styles(self) -> Dict[str, StyleParameters]:
        """
        Get all available styles.
        
        Returns:
            Dictionary of all styles
        """
        return self.styles

    def get_all_categories(self) -> List[str]:
        """
        Get all available categories.
        
        Returns:
            List of category names
        """
        return list(self.categories.keys())

    def blend_styles(self, 
                    style1_name: str, 
                    style2_name: str, 
                    blend_ratio: float = 0.5, 
                    use_phi_ratio: bool = True,
                    new_style_name: Optional[str] = None) -> Optional[Tuple[str, StyleParameters]]:
        """
        Blend two styles together, optionally using phi ratio for optimal blending.
        
        Args:
            style1_name: Name of the first style
            style2_name: Name of the second style
            blend_ratio: Ratio of style2 to include (0.0-1.0)
            use_phi_ratio: If True, uses golden ratio for blending
            new_style_name: Name for the new blended style. If None, generates a name.
            
        Returns:
            Tuple of (name, StyleParameters) for the new blended style, or None if failed
        """
        # Get the styles
        style1 = self.get_style(style1_name)
        style2 = self.get_style(style2_name)
        
        if not style1 or not style2:
            missing = style1_name if not style1 else style2_name
            logger.error(f"Style not found: {missing}")
            return None
        
        # If requested, use golden ratio for blending
        if use_phi_ratio:
            phi = calculate_phi_ratio()
            blend_ratio = phi - 1  # ~0.618
        
        # Ensure blend_ratio is in valid range
        blend_ratio = max(0.0, min(1.0, blend_ratio))
        
        # Create a new style by blending parameters
        style1_dict = asdict(style1)
        style2_dict = asdict(style2)
        
        # Prepare the new style parameters
        new_style_params = {}
        
        # Blend numeric parameters
        for key, value in style1_dict.items():
            if isinstance(value, (int, float)) and key in style2_dict:
                # Linear interpolation
                new_value = value * (1 - blend_ratio) + style2_dict[key] * blend_ratio
                
                # Round integers
                if isinstance(value, int):
                    new_value = round(new_value)
                    
                new_style_params[key] = new_value
            elif isinstance(value, bool) and key in style2_dict:
                # For boolean values, use the value from style1 if blend_ratio < 0.5
                # Otherwise use the value from style2
                new_style_params[key] = style2_dict[key] if blend_ratio >= 0.5 else value
            else:
                # For non-numeric parameters, keep the value from style1
                new_style_params[key] = value
        
        # Generate a name if not provided
        if not new_style_name:
            # Get display names if available
            style1_display = self.metadata["style_properties"].get(style1_name, {}).get("display_name", style1_name)
            style2_display = self.metadata["style_properties"].get(style2_name, {}).get("display_name", style2_name)
            
            new_style_name = f"blend_{style1_name}_{style2_name}_{int(blend_ratio * 100)}"
            display_name = f"{style1_display} & {style2_display} Fusion"
        else:
            display_name = new_style_name.replace('_', ' ').title()
        
        # Set descriptive properties
        new_style_params["display_name"] = display_name
        new_style_params["category"] = style1_dict.get("category", "blended")
        
        # Generate description
        style1_desc = self.metadata["style_properties"].get(style1_name, {}).get("description", "")
        style2_desc = self.metadata["style_properties"].get(style2_name, {}).get("description", "")
        new_style_params["description"] = f"A blend of {style1_dict.get('display_name', style1_name)} and {style2_dict.get('display_name', style2_name)}"
        
        if style1_desc and style2_desc:
            new_style_params["description"] += f". Combines elements of {style1_desc} with {style2_desc}"
        
        # Combine influences
        influences1 = self.metadata["style_properties"].get(style1_name, {}).get("influences", [])
        influences2 = self.metadata["style_properties"].get(style2_name, {}).get("influences", [])
        new_style_params["influences"] = list(set(influences1 + influences2 + [style1_name, style2_name]))
        
        # Create the new style
        blended_style = self.create_style(name=new_style_name, **new_style_params)
        
        return (new_style_name, blended_style)

    def transform_style(self, 
                       style_name: str, 
                       transformation_params: Dict[str, float],
                       new_style_name: Optional[str] = None,
                       consciousness_level: Optional[int] = None,
                       sacred_geometry_enhance: bool = True) -> Optional[Tuple[str, StyleParameters]]:
        """
        Transform a style by modifying specific parameters with advanced consciousness level
        and sacred geometry enhancements.
        
        Args:
            style_name: Name of the style to transform
            transformation_params: Dictionary of parameters to transform and their new values
            new_style_name: Name for the new transformed style. If None, generates a name.
            consciousness_level: Optional consciousness level to apply (1, 3, 5, 8, or 13)
            sacred_geometry_enhance: Whether to apply sacred geometry principles to transformations
            
        Returns:
            Tuple of (name, StyleParameters) for the new transformed style, or None if failed
        """
        # Get the style
        style = self.get_style(style_name)
        
        if not style:
            logger.error(f"Style not found: {style_name}")
            return None
            
        # Create a copy of the original style as a dictionary
        style_dict = asdict(style)
        
        # Track transformation description for metadata
        transformation_description = []
        
        # Apply the transformations
        for param, value in transformation_params.items():
            if param in style_dict:
                # Store original value for description
                original_value = style_dict[param]
                
                # Apply the transformation
                if sacred_geometry_enhance and isinstance(value, float):
                    # Apply golden ratio (phi) for musically pleasing transformations
                    phi = self.sacred_geometry_ratios["phi"] if hasattr(self, "sacred_geometry_ratios") else 1.618
                    
                    # Determine if we're increasing or decreasing the parameter
                    if value > original_value:
                        # Increasing - use phi ratio
                        style_dict[param] = original_value + (value - original_value) * (1/phi)
                    else:
                        # Decreasing - use phi ratio
                        style_dict[param] = original_value - (original_value - value) * (1/phi)
                else:
                    # Direct parameter change
                    style_dict[param] = value
                
                # Record the transformation for description
                transformation_description.append(f"{param}: {original_value:.2f} → {style_dict[param]:.2f}")
            else:
                logger.warning(f"Parameter '{param}' not found in style parameters.")
        
        # Apply consciousness level transformations if specified
        if consciousness_level is not None:
            # Validate consciousness level - must be in Fibonacci sequence
            valid_levels = [1, 3, 5, 8, 13]
            if consciousness_level not in valid_levels:
                closest_level = min(valid_levels, key=lambda x: abs(x - consciousness_level))
                logger.warning(f"Invalid consciousness level {consciousness_level}. Using closest valid level: {closest_level}")
                consciousness_level = closest_level
            
            # Get consciousness parameters for the specified level
            if hasattr(self, "consciousness_parameters") and consciousness_level in self.consciousness_parameters:
                consciousness_params = self.consciousness_parameters[consciousness_level]
                
                # Apply consciousness parameters to the style
                for param, value in consciousness_params.items():
                    if param in style_dict:
                        # Apply consciousness adjustment with golden ratio weighting
                        style_dict[param] = style_dict[param] * 0.7 + value * 0.3
                
                # Update the consciousness level
                style_dict["consciousness_level"] = consciousness_level
                transformation_description.append(f"consciousness: {style.consciousness_level} → {consciousness_level}")
            else:
                # Simple implementation if consciousness_parameters not available
                style_dict["consciousness_level"] = consciousness_level
                transformation_description.append(f"consciousness: {style.consciousness_level} → {consciousness_level}")
        
        # Apply sacred geometry enhancements to rhythmic elements
        if sacred_geometry_enhance:
            # Get phi (golden ratio) - fallback to hardcoded if not available in class
            phi = self.sacred_geometry_ratios["phi"] if hasattr(self, "sacred_geometry_ratios") else 1.618
            
            # Enhance rhythmic parameters based on sacred geometry principles
            if "pattern_complexity" in style_dict:
                # Optimize pattern complexity using phi-based scaling
                style_dict["pattern_complexity"] = (style_dict["pattern_complexity"] * phi) % 1.0
                
            if "swing_amount" in style_dict:
                # Apply golden ratio to swing for more natural feel
                style_dict["swing_amount"] = min(1.0, style_dict["swing_amount"] * (1/phi) + (1 - 1/phi) * 0.382)
                
            if "bpm" in style_dict:
                # Adjust BPM to align with phi if it's far from a phi-aligned value
                base_bpm = style_dict["bpm"]
                phi_aligned_bpm = round(base_bpm / phi) * phi
                
                # Only adjust if we're more than 5% away from a phi-aligned value
                if abs(base_bpm - phi_aligned_bpm) / base_bpm > 0.05:
                    style_dict["bpm"] = phi_aligned_bpm
                    transformation_description.append(f"bpm phi-aligned: {base_bpm} → {phi_aligned_bpm:.1f}")
            
            # Set sacred geometry flag for the style
            style_dict["sacred_geometry_enabled"] = True
        
        # Generate a name if not provided
        if not new_style_name:
            # Create a descriptive name based on transformations
            transformation_type = next(iter(transformation_params.keys()), "transformed")
            new_style_name = f"{style_name}_{transformation_type}_variant"
            
            # Add consciousness marker if applicable
            if consciousness_level is not None:
                new_style_name += f"_c{consciousness_level}"
        
        # Create the new style with transformed parameters
        try:
            # Convert style_dict values to appropriate types if needed
            # (handling edge cases where values might be incorrectly typed)
            for key, value in style_dict.items():
                # Ensure float values are within valid ranges
                if isinstance(value, float) and key not in ["bpm"]:
                    style_dict[key] = max(0.0, min(1.0, value))
                    
            # Create the new style 
            display_name = new_style_name.replace('_', ' ').title()
            style_dict["display_name"] = display_name
            
            # Generate description of transformation
            base_description = self.metadata["style_properties"].get(style_name, {}).get("description", "")
            transformation_summary = ", ".join(transformation_description)
            style_dict["description"] = f"Transformed from {style_name}. {base_description} Modifications: {transformation_summary}"
            
            # Create the style
            new_style = self.create_style(name=new_style_name, **style_dict)
            
            # Add relationship to original style
            self.metadata["style_properties"][new_style_name]["derived_from"] = style_name
            
            # Track evolution history
            if hasattr(self, "style_evolution_history") and style_name in self.style_evolution_history:
                self.style_evolution_history[new_style_name] = self.style_evolution_history[style_name] + [style_name]
            
            return (new_style_name, new_style)
            
        except Exception as e:
            logger.error(f"Error creating transformed style: {str(e)}")
            return None
        

import json
import os
import random
import copy
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from pathlib import Path

from .architect import StyleParameters

class StyleFactory:
    """
    Factory class for creating, loading, saving, and manipulating beat styles.
    
    This class manages a database of styles that can be loaded from JSON files,
    created programmatically, blended together, or transformed based on consciousness
    parameters to produce advanced musical styles with sacred geometry properties.
    """
    
    def __init__(self, database_path: Optional[str] = None):
        """
        Initialize the StyleFactory with an optional database path.
        
        Args:
            database_path: Path to a JSON file containing style definitions.
                          If None, initializes with empty database.
        """
        self.style_database: Dict[str, StyleParameters] = {}
        self.style_metadata: Dict[str, Dict[str, Any]] = {}
        self.style_evolution_history: Dict[str, List[str]] = {}
        
        self._initialize_metadata()
        
        if database_path:
            db_path = Path(database_path)
            if db_path.exists():
                self.load_database(database_path)
            else:
                print(f"Database file {database_path} not found. Initializing empty database.")
    
    def _initialize_metadata(self) -> None:
        """
        Initialize style metadata including categories, family trees, and consciousness mappings.
        This metadata is used for organizing styles, tracking derivations, and enhancing
        styles with consciousness parameters.
        """
        # Main style categories
        self.style_categories = {
            "electronic": [
                "house", "techno", "trance", "drum_and_bass", "breakbeat", 
                "dubstep", "garage", "ambient", "downtempo", "idm", "glitch",
                "synthwave", "electro", "edm", "future_bass"
            ],
            "hip_hop": [
                "boom_bap", "trap", "lo_fi", "drill", "grime", "cloud_rap",
                "phonk", "mumble", "abstract", "instrumental", "hyphy"
            ],
            "experimental": [
                "glitch", "noise", "generative", "algorithmic", "quantum",
                "consciousness", "neural", "sacred_geometry", "abstract",
                "granular", "textural", "binaural", "microtonal"
            ],
            "fusion": [
                "jazz_fusion", "world_fusion", "electronic_fusion", "quantum_fusion",
                "tribal_tech", "cosmic_jazz", "neuro_funk", "ethno_dub"
            ],
            "ambient": [
                "dark_ambient", "drone", "space_ambient", "binaural", 
                "meditation", "environmental", "generative_ambient", "soundscape"
            ]
        }
        
        # Cross-style influence mapping (which styles influence others)
        self.style_influences = {
            "trap": ["hip_hop", "dubstep", "808", "southern"],
            "lo_fi": ["hip_hop", "jazz", "ambient", "cassette", "vinyl"],
            "techno": ["electronic", "industrial", "minimalist", "detroit", "berlin"],
            "drum_and_bass": ["breakbeat", "electronic", "jungle", "liquid"],
            "ambient": ["electronic", "atmospheric", "textural", "environmental"],
            "house": ["disco", "electronic", "garage", "deep", "soulful"],
            "experimental": ["avant_garde", "abstract", "noise", "academic"],
            "future_bass": ["trap", "pop", "edm", "synth", "bright"],
            "synthwave": ["80s", "retro", "soundtrack", "cinematic"],
            "neuro_funk": ["drum_and_bass", "techno", "glitch", "sound_design"]
        }
        
        # Consciousness-level mapping with detailed parameters 
        # Based on Fibonacci sequence (1,3,5,8,13) reflecting different states of consciousness
        self.consciousness_parameters = {
            1: {  # Base material consciousness
                "complexity": 0.2, 
                "harmony_richness": 0.1, 
                "frequency_shift": 0.0,
                "sacred_geometry_influence": 0.1,
                "quantum_coherence": 0.0,
                "neural_synchronization": 0.2,
                "fractal_dimension": 1.2
            },
            3: {  # Expanded awareness
                "complexity": 0.3, 
                "harmony_richness": 0.3, 
                "frequency_shift": 0.2,
                "sacred_geometry_influence": 0.3,
                "quantum_coherence": 0.2,
                "neural_synchronization": 0.3,
                "fractal_dimension": 1.5
            },
            5: {  # Creative consciousness
                "complexity": 0.5, 
                "harmony_richness": 0.5, 
                "frequency_shift": 0.4,
                "sacred_geometry_influence": 0.5,
                "quantum_coherence": 0.4,
                "neural_synchronization": 0.5,
                "fractal_dimension": 1.7
            },
            8: {  # Harmonic consciousness
                "complexity": 0.8, 
                "harmony_richness": 0.7, 
                "frequency_shift": 0.6,
                "sacred_geometry_influence": 0.7,
                "quantum_coherence": 0.6,
                "neural_synchronization": 0.8,
                "fractal_dimension": 1.9
            },
            13: { # Transcendent consciousness
                "complexity": 1.0, 
                "harmony_richness": 0.9, 
                "frequency_shift": 0.8,
                "sacred_geometry_influence": 1.0,
                "quantum_coherence": 0.9,
                "neural_synchronization": 1.0,
                "fractal_dimension": 2.1
            }
        }
        
        # Sacred geometry ratios for enhanced consciousness integration
        self.sacred_geometry_ratios = {
            "phi": (1 + 5 ** 0.5) / 2,  # Golden ratio
            "sqrt2": 2 ** 0.5,          # Square root of 2
            "sqrt3": 3 ** 0.5,          # Square root of 3
            "sqrt5": 5 ** 0.5,          # Square root of 5
            "pi_phi": np.pi / ((1 + 5 ** 0.5) / 2)  # Pi/Phi ratio
        }
        
        # Frequency bands associated with brain states
        self.neural_frequency_bands = {
            "delta": (0.5, 4),    # Deep sleep, healing
            "theta": (4, 8),      # Meditation, creativity
            "alpha": (8, 12),     # Relaxed awareness
            "beta": (12, 30),     # Active thinking, focus
            "gamma": (30, 100)    # Higher cognitive processing
        }
    
    def load_database(self, database_path: str) -> None:
        """
        Load style database from a JSON file.
        
        Args:
            database_path: Path to the JSON file containing style definitions.
        """
        try:
            with open(database_path, 'r') as f:
                style_data = json.load(f)
            
            # Load styles
            loaded_styles = 0
            for style_name, style_params in style_data.get('styles', {}).items():
                # Process nested objects if necessary
                if 'frequency_ratios' in style_params and isinstance(style_params['frequency_ratios'], list):
                    style_params['frequency_ratios'] = np.array(style_params['frequency_ratios'])
                
                # Create StyleParameters object
                self.style_database[style_name] = StyleParameters(**style_params)
                loaded_styles += 1
            
            # Load metadata if present
            if 'metadata' in style_data:
                for style_name, metadata in style_data['metadata'].items():
                    self.style_metadata[style_name] = metadata
            
            # Load evolution history if present
            if 'evolution_history' in style_data:
                self.style_evolution_history = style_data['evolution_history']
                    
            print(f"Loaded {loaded_styles} styles from {database_path}")
            
        except Exception as e:
            print(f"Error loading style database: {e}")
            raise
    
    def create_style(self, name: str, **kwargs) -> StyleParameters:
        """
        Create a new style with the given parameters.
        
        Args:
            name: Name of the style
            **kwargs: Parameters for the style
        
        Returns:
            The created StyleParameters object
        """
        # Generate default values for any missing parameters
        default_params = {
            "bpm": 120,
            "swing": 0.0,
            "pattern_complexity": 0.5,
            "harmonic_complexity": 0.5,
            "intensity": 0.5,
            "frequency_shift": 0.0,
            "consciousness_level": 5,
            "frequency_ratios": np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        }
        
        # Merge defaults with provided kwargs
        for key, value in default_params.items():
            if key not in kwargs:
                kwargs[key] = value
        
        # Create the style
        style = StyleParameters(**kwargs)
        self.style_database[name] = style
        
        # Initialize metadata for this style
        self.style_metadata[name] = {
            "created": True,
            "created_timestamp": self._get_timestamp(),
            "base_style": kwargs.get("base_style", None),
            "tags": kwargs.get("tags", []),
            "category": kwargs.get("category", "uncategorized"),
            "consciousness_level": kwargs.get("consciousness_level", 5),
            "description": kwargs.get("description", f"Custom style: {name}")
        }
        
        # Initialize evolution history
        self.style_evolution_history[name] = []
        
        return style
    
    def export_style(self, style_name: str, file_path: Optional[str] = None) -> Optional[Dict]:
        """
        Export a style to JSON.
        
        Args:
            style_name: Name of the style to export
            file_path: Path to save the JSON file. If None, returns the data as a dict.
        
        Returns:
            Dictionary representation of the style if file_path is None, else None
        """
        if style_name not in self.style_database:
            print(f"Style '{style_name}' not found in database")
            return None
        
        style = self.style_database[style_name]
        style_dict = asdict(style)
        
        # Handle numpy arrays for JSON serialization
        if 'frequency_ratios' in style_dict and isinstance(style_dict['frequency_ratios'], np.ndarray):
            style_dict['frequency_ratios'] = style_dict['frequency_ratios'].tolist()
        
        # Add metadata
        export_data = {
            "styles": {style_name: style_dict},
            "metadata": {style_name: self.style_metadata.get(style_name, {})},
            "evolution_history": {style_name: self.style_evolution_history.get(style_name, [])}
        }
        
        if file_path:
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            print(f"Style '{style_name}' exported to {file_path}")
            return None
        else:
            return export_data
    
    def export_database(self, file_path: str) -> None:
        """
        Export the entire style database to a JSON file.
        
        Args:
            file_path: Path to save the JSON file
        """
        export_data = {
            "styles": {}, 
            "metadata": self.style_metadata,
            "evolution_history": self.style_evolution_history
        }
        
        for name, style in self.style_database.items():
            style_dict = asdict(style)
            
            # Handle numpy arrays for JSON serialization
            if 'frequency_ratios' in style_dict and isinstance(style_dict['frequency_ratios'], np.ndarray):
                style_dict['frequency_ratios'] = style_dict['frequency_ratios'].tolist()
            
            export_data["styles"][name] = style_dict
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Style database with {len(self.style_database)} styles exported to {file_path}")
    
    def blend_styles(self, style1_name: str, style2_name: str, 
                    blend_ratio: float = 0.5, new_name: Optional[str] = None,
                    enhance_with_sacred_geometry: bool = False) -> StyleParameters:
        """
        Blend two styles together using the specified ratio with optional sacred geometry enhancement.
        
        Args:
            style1_name: Name of the first style
            style2_name: Name of the second style
            blend_ratio: How much of style1 vs style2 (0.5 = equal blend)
            new_name: Name for the blended style. If None, auto-generated.
            enhance_with_sacred_geometry: Whether to enhance the blend with sacred geometry ratios
        
        Returns:
            The blended StyleParameters object
        """
        if style1_name not in self.style_database:
            raise ValueError(f"Style '{style1_name}' not found in database")
        if style2_name not in self.style_database:
            raise ValueError(f"Style '{style2_name}' not found in database")
        
        style1 = self.style_database[style1_name]
        style2 = self.style_database[style2_name]
        
        # Create dictionary representations for blending
        style1_dict = asdict(style1)
        style2_dict = asdict(style2)
        blended_dict = {}
        
        # Blend numeric parameters
        for param, value in style1_dict.items():
            if isinstance(value, (int, float)) and param in style2_dict:
                blended_dict[param] = value * blend_ratio + style2_dict[param] * (1 - blend_ratio)
            elif isinstance(value, np.ndarray) and param in style2_dict:
                # Handle numpy arrays like frequency_ratios
                if len(value) == len(style2_dict[param]):
                    blended_dict[param] = value * blend_ratio + style2_dict[param] * (1 - blend_ratio)
                else:
                    # If arrays have different dimensions, combine them intelligently
                    combined = np.unique(np.concatenate([value, style2_dict[param]]))
                    # Apply weighting to combined array
                    style1_influence = np.zeros_like(combined)
                    style2_influence = np.zeros_like(combined)
                    
                    for i, freq in enumerate(combined):
                        if freq in value:
                            style1_influence[i] = 1.0
                        if freq in style2_dict[param]:
                            style2_influence[i] = 1.0

"""
Style Factory Module for Neural Beat Architect

Provides comprehensive tooling for creating, managing, and customizing musical styles.
This module enables the creation, loading, merging, and transformation of musical
styles with consciousness-enhancing capabilities.

Key features:
- Load styles from style_database.json
- Create new custom styles
- Blend multiple styles with phi-optimized weighting
- Transform styles with consciousness enhancement
- Generate new styles based on era, culture, or vibe
"""

import os
import json
import logging
import random
import copy
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any, Set
from dataclasses import asdict, is_dataclass
from pathlib import Path

from .architect import StyleParameters

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio - key to consciousness-optimized blending
DEFAULT_DATABASE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                    "data", "style_database.json")

class StyleFactory:
    """
    Factory class for creating and managing musical styles with consciousness enhancement.
    
    This class provides methods for loading styles from database, creating new styles,
    blending multiple styles, and transforming styles with consciousness-optimized algorithms.
    """
    
    def __init__(self, 
                database_path: Optional[str] = None,
                load_database: bool = True,
                consciousness_level: float = 0.8,
                enable_style_evolution: bool = True):
        """
        Initialize the StyleFactory with optional database loading.
        
        Args:
            database_path: Path to style database JSON file
            load_database: Whether to load database on initialization
            consciousness_level: Base level of consciousness enhancement (0.0-1.0)
            enable_style_evolution: Allow styles to evolve through usage
        """
        self.database_path = database_path or DEFAULT_DATABASE_PATH
        self.consciousness_level = consciousness_level
        self.enable_style_evolution = enable_style_evolution
        self.style_database = {}
        self.style_categories = set()
        self.style_eras = set()
        self.style_cultures = set()
        self.style_moods = set()
        
        # Load database if requested
        if load_database:
            self.load_database()
            
        # Initialize default style categories if database is empty
        if not self.style_categories:
            self._initialize_metadata()
    
    def load_database(self, database_path: Optional[str] = None) -> bool:
        """
        Load style database from JSON file.
        
        Args:
            database_path: Path to database file (overrides initialization path)
            
        Returns:
            Success status of database loading
        """
        #

