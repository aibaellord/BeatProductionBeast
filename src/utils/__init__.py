"""
Utility functions for the BeatProductionBeast project.

This module contains various helper functions and utilities that are used
across different components of the application.
"""

# File handling utilities
from .file_utils import (
    load_audio_file,
    save_audio_file,
    get_project_path,
    ensure_directory_exists,
)

# Data processing utilities
from .data_utils import (
    normalize_data,
    denormalize_data,
    convert_to_tensor,
    convert_from_tensor,
)

# Logging and debugging utilities
from .logging_utils import (
    setup_logger,
    log_error,
    log_warning,
    log_info,
)

# Configuration handling
from .config_utils import (
    load_config,
    save_config,
    get_default_config,
)

# Time and performance utilities
from .performance_utils import (
    timer,
    measure_execution_time,
    profile_function,
)

# Common mathematical operations
from .math_utils import (
    smooth_curve,
    apply_window,
    fourier_transform,
    inverse_fourier_transform,
)

# Sacred Geometry utilities
from .sacred_geometry_core import SacredGeometryCore
from .sacred_geometry_patterns import SacredGeometryPatterns

__all__ = [
    # File utilities
    'load_audio_file',
    'save_audio_file',
    'get_project_path',
    'ensure_directory_exists',
    
    # Data utilities
    'normalize_data',
    'denormalize_data',
    'convert_to_tensor',
    'convert_from_tensor',
    
    # Logging utilities
    'setup_logger',
    'log_error',
    'log_warning',
    'log_info',
    
    # Config utilities
    'load_config',
    'save_config',
    'get_default_config',
    
    # Performance utilities
    'timer',
    'measure_execution_time',
    'profile_function',
    
    # Math utilities
    'smooth_curve',
    'apply_window',
    'fourier_transform',
    'inverse_fourier_transform',
    
    # Sacred Geometry
    'SacredGeometryCore',
    'SacredGeometryPatterns',
]

