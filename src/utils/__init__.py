"""
Utility functions for the BeatProductionBeast project.

This module contains various helper functions and utilities that are used
across different components of the application.
"""

# Configuration handling
from .config_utils import get_default_config, load_config, save_config
# Data processing utilities
from .data_utils import (convert_from_tensor, convert_to_tensor,
                         denormalize_data, normalize_data)
# File handling utilities
from .file_utils import (ensure_directory_exists, get_project_path,
                         load_audio_file, save_audio_file)
# Logging and debugging utilities
from .logging_utils import log_error, log_info, log_warning, setup_logger
# Common mathematical operations
from .math_utils import (apply_window, fourier_transform,
                         inverse_fourier_transform, smooth_curve)
# Time and performance utilities
from .performance_utils import measure_execution_time, profile_function, timer
# Sacred Geometry utilities
from .sacred_geometry_core import SacredGeometryCore
from .sacred_geometry_patterns import SacredGeometryPatterns

__all__ = [
    # File utilities
    "load_audio_file",
    "save_audio_file",
    "get_project_path",
    "ensure_directory_exists",
    # Data utilities
    "normalize_data",
    "denormalize_data",
    "convert_to_tensor",
    "convert_from_tensor",
    # Logging utilities
    "setup_logger",
    "log_error",
    "log_warning",
    "log_info",
    # Config utilities
    "load_config",
    "save_config",
    "get_default_config",
    # Performance utilities
    "timer",
    "measure_execution_time",
    "profile_function",
    # Math utilities
    "smooth_curve",
    "apply_window",
    "fourier_transform",
    "inverse_fourier_transform",
    # Sacred Geometry
    "SacredGeometryCore",
    "SacredGeometryPatterns",
]
