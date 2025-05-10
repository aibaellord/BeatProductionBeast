"""
Preset Management Module for AutomatedBeatCopycat

This module provides the Preset model and related utilities for storing, 
validating, and managing user-created presets. Presets allow users to save
and reuse their favorite configuration settings for beat generation.
"""

import datetime
import json
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union


# Constants for preset categories
class Genre(str, Enum):
    """Supported music genres for preset categorization"""
    HIP_HOP = "hip_hop"
    TRAP = "trap"
    LOFI = "lofi"
    AMBIENT = "ambient"
    ELECTRONIC = "electronic"
    EXPERIMENTAL = "experimental"
    POP = "pop"
    RNB = "rnb"
    JAZZ = "jazz"
    CLASSICAL = "classical"
    ROCK = "rock"
    DUBSTEP = "dubstep"
    HOUSE = "house"
    TECHNO = "techno"
    CUSTOM = "custom"

class Mood(str, Enum):
    """Emotional moods for preset categorization"""
    ENERGETIC = "energetic"
    RELAXED = "relaxed"
    DREAMY = "dreamy"
    DARK = "dark"
    UPLIFTING = "uplifting"
    MELANCHOLIC = "melancholic"
    AGGRESSIVE = "aggressive"
    PEACEFUL = "peaceful"
    FOCUSED = "focused"
    NOSTALGIC = "nostalgic"
    HYPNOTIC = "hypnotic"
    MEDITATIVE = "meditative"
    JOYFUL = "joyful"
    INTENSE = "intense"
    ETHEREAL = "ethereal"

class ConsciousnessLevel(str, Enum):
    """Consciousness levels for neural entrainment"""
    DELTA = "delta"        # Deep sleep (0.5-4 Hz)
    THETA = "theta"        # Meditation, creativity (4-8 Hz)
    ALPHA = "alpha"        # Relaxation, light meditation (8-13 Hz)
    BETA = "beta"          # Active thinking, focus (13-30 Hz)
    GAMMA = "gamma"        # Higher mental activity, insight (30-100 Hz)
    LAMBDA = "lambda"      # Advanced states (100-200 Hz)
    EPSILON = "epsilon"    # Transcendental (>200 Hz)
    CUSTOM = "custom"      # User-defined frequencies

@dataclass
class PresetTags:
    """Tags associated with a preset for categorization and filtering"""
    genres: Set[Genre] = field(default_factory=set)
    moods: Set[Mood] = field(default_factory=set)
    consciousness_levels: Set[ConsciousnessLevel] = field(default_factory=set)
    custom_tags: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, List[str]]:
        """Convert tags to a dictionary for serialization"""
        return {
            "genres": [genre.value for genre in self.genres],
            "moods": [mood.value for mood in self.moods],
            "consciousness_levels": [level.value for level in self.consciousness_levels],
            "custom_tags": list(self.custom_tags)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, List[str]]) -> 'PresetTags':
        """Create a PresetTags instance from a dictionary"""
        return cls(
            genres={Genre(g) for g in data.get("genres", []) if g in [e.value for e in Genre]},
            moods={Mood(m) for m in data.get("moods", []) if m in [e.value for e in Mood]},
            consciousness_levels={
                ConsciousnessLevel(c) for c in data.get("consciousness_levels", [])
                if c in [e.value for e in ConsciousnessLevel]
            },
            custom_tags=set(data.get("custom_tags", []))
        )
    
    def add_tag(self, tag_type: str, tag_value: str) -> bool:
        """Add a tag to the appropriate category"""
        if tag_type == "genre" and tag_value in [e.value for e in Genre]:
            self.genres.add(Genre(tag_value))
            return True
        elif tag_type == "mood" and tag_value in [e.value for e in Mood]:
            self.moods.add(Mood(tag_value))
            return True
        elif tag_type == "consciousness_level" and tag_value in [e.value for e in ConsciousnessLevel]:
            self.consciousness_levels.add(ConsciousnessLevel(tag_value))
            return True
        elif tag_type == "custom":
            self.custom_tags.add(tag_value)
            return True
        return False
    
    def remove_tag(self, tag_type: str, tag_value: str) -> bool:
        """Remove a tag from the appropriate category"""
        if tag_type == "genre" and Genre(tag_value) in self.genres:
            self.genres.remove(Genre(tag_value))
            return True
        elif tag_type == "mood" and Mood(tag_value) in self.moods:
            self.moods.remove(Mood(tag_value))
            return True
        elif tag_type == "consciousness_level" and ConsciousnessLevel(tag_value) in self.consciousness_levels:
            self.consciousness_levels.remove(ConsciousnessLevel(tag_value))
            return True
        elif tag_type == "custom" and tag_value in self.custom_tags:
            self.custom_tags.remove(tag_value)
            return True
        return False

class ValidationError(Exception):
    """Exception raised for preset validation errors"""
    pass

@dataclass
class Preset:
    """
    Preset class for storing and managing beat generation configurations
    
    Attributes:
        name: Descriptive name for the preset
        config_data: Dictionary containing all configuration parameters
        tags: Categories and tags for organizing and searching presets
        user_id: ID of the user who created/owns the preset
        id: Unique identifier for the preset (auto-generated if not provided)
        description: Optional description of what the preset does
        created_at: Timestamp when the preset was created
        updated_at: Timestamp when the preset was last updated
        is_public: Whether the preset is publicly available to other users
    """
    name: str
    config_data: Dict[str, Any]
    tags: PresetTags
    user_id: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: Optional[str] = None
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    is_public: bool = False
    
    def __post_init__(self):
        """Validate the preset after initialization"""
        self.validate()
    
    def validate(self) -> bool:
        """
        Validate the preset configuration data
        
        Raises:
            ValidationError: If the preset has invalid or missing data
        """
        # Validate name
        if not self.name or not isinstance(self.name, str):
            raise ValidationError("Preset name must be a non-empty string")
        
        # Validate config_data
        if not isinstance(self.config_data, dict):
            raise ValidationError("Configuration data must be a dictionary")
        
        # Validate required configuration keys
        required_keys = ["sacred_geometry", "consciousness_modulation", "neural_optimization"]
        missing_keys = [key for key in required_keys if key not in self.config_data]
        if missing_keys:
            raise ValidationError(f"Configuration data missing required keys: {', '.join(missing_keys)}")
            
        # Validate complex configurations
        self._validate_sacred_geometry()
        self._validate_consciousness_modulation()
        self._validate_neural_optimization()
        
        return True
    
    def _validate_sacred_geometry(self):
        """Validate sacred geometry configuration section"""
        sg_config = self.config_data.get("sacred_geometry", {})
        
        # Check if enabled
        if not isinstance(sg_config.get("enabled"), bool):
            raise ValidationError("sacred_geometry.enabled must be a boolean")
            
        # Check pattern type if enabled
        if sg_config.get("enabled"):
            valid_patterns = ["fibonacci", "golden_ratio", "flower_of_life", "platonic_solid", "custom"]
            pattern = sg_config.get("pattern")
            if not pattern or pattern not in valid_patterns:
                raise ValidationError(f"Invalid sacred geometry pattern. Must be one of: {', '.join(valid_patterns)}")
    
    def _validate_consciousness_modulation(self):
        """Validate consciousness modulation configuration section"""
        cm_config = self.config_data.get("consciousness_modulation", {})
        
        # Check if enabled
        if not isinstance(cm_config.get("enabled"), bool):
            raise ValidationError("consciousness_modulation.enabled must be a boolean")
            
        # Check level if enabled
        if cm_config.get("enabled"):
            valid_levels = [level.value for level in ConsciousnessLevel]
            level = cm_config.get("level")
            if not level or level not in valid_levels:
                raise ValidationError(f"Invalid consciousness level. Must be one of: {', '.join(valid_levels)}")
                
            # Validate frequency range if using custom level
            if level == "custom":
                freq_range = cm_config.get("frequency_range")
                if not freq_range or not isinstance(freq_range, dict):
                    raise ValidationError("Custom consciousness level requires a frequency_range dict")
                if "min" not in freq_range or "max" not in freq_range:
                    raise ValidationError("frequency_range must contain min and max values")
                if not (0 <= freq_range["min"] < freq_range["max"] <= 1000):
                    raise ValidationError("frequency_range values must be between 0 and 1000 Hz with min < max")
    
    def _validate_neural_optimization(self):
        """Validate neural optimization configuration section"""
        no_config = self.config_data.get("neural_optimization", {})
        
        # Check if enabled
        if not isinstance(no_config.get("enabled"), bool):
            raise ValidationError("neural_optimization.enabled must be a boolean")
            
        # Check target if enabled
        if no_config.get("enabled"):
            valid_targets = ["dopamine", "serotonin", "focus", "relaxation", "creativity", "custom"]
            target = no_config.get("target")
            if not target or target not in valid_targets:
                raise ValidationError(f"Invalid neural target. Must be one of: {', '.join(valid_targets)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the preset to a dictionary for serialization"""
        preset_dict = asdict(self)
        # Convert custom objects to serializable format
        preset_dict["tags"] = self.tags.to_dict()
        preset_dict["created_at"] = self.created_at.isoformat()
        preset_dict["updated_at"] = self.updated_at.isoformat()
        return preset_dict
    
    def to_json(self) -> str:
        """Serialize the preset to a JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Preset':
        """Create a Preset instance from a dictionary"""
        # Handle datetime conversion
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.datetime.fromisoformat(data["created_at"])
        if "updated_at" in data and isinstance(data["updated_at"], str):
            data["updated_at"] = datetime.datetime.fromisoformat(data["updated_at"])
            
        # Handle tags conversion
        if "tags" in data and isinstance(data["tags"], dict):
            data["tags"] = PresetTags.from_dict(data["tags"])
            
        # Create preset instance
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Preset':
        """Create a Preset instance from a JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def update(self, **kwargs) -> None:
        """
        Update preset attributes
        
        Args:
            **kwargs: Attributes to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key) and key != "id":  # Prevent changing the ID
                setattr(self, key, value)
                
        # Update the updated_at timestamp
        self.updated_at = datetime.datetime.now()
        
        # Revalidate after update
        self.validate()
    
    def merge_config(self, new_config: Dict[str, Any]) -> None:
        """
        Merge new configuration data with existing config
        
        Args:
            new_config: New configuration dictionary to merge
        """
        # Recursively merge dictionaries
        def merge_dicts(d1, d2):
            for k, v in d2.items():
                if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                    merge_dicts(d1[k], v)
                else:
                    d1[k] = v
        
        merge_dicts(self.config_data, new_config)
        self.updated_at = datetime.datetime.now()
        self.validate()
        
    def create_variation(self, variation_name: str, config_changes: Dict[str, Any]) -> 'Preset':
        """
        Create a variation of this preset with modified configuration
        
        Args:
            variation_name: Name for the new preset variation
            config_changes: Dictionary with configuration changes to apply
            
        Returns:
            A new Preset instance with the applied changes
        """
        # Create a deep copy of the config data
        new_config = json.loads(json.dumps(self.config_data))
        
        # Apply changes
        def apply_changes(base_dict, changes_dict):
            for k, v in changes_dict.items():
                if k in base_dict and isinstance(base_dict[k], dict) and isinstance(v, dict):
                    apply_changes(base_dict[k], v)
                else:
                    base_dict[k] = v
        
        apply_changes(new_config, config_changes)
        
        # Create new preset with variation name
        return Preset(
            name=f"{self.name} - {variation_name}",
            config_data=new_config,
            tags=PresetTags.from_dict(self.tags.to_dict()),  # Create a copy of tags
            user_id=self.user_id,
            description=f"Variation of '{self.name}': {variation_name}"
        )

class PresetManager:
    """
    Manager class for handling preset operations
    
    This class provides utility methods for working with collections of presets,
    including searching, filtering, and organizing presets.
    """
    
    @staticmethod
    def filter_presets(presets: List[Preset], **filters) -> List[Preset]:

