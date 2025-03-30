#!/usr/bin/env python
"""
Command-line interface for the beat generation module of BeatProductionBeast.

This module provides a focused interface for generating beats with various
styles, patterns, and export options.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

# Import beat generation functionality
# These imports will need to be updated based on actual implementation
try:
    from .. import beat_generation  # Adjust based on actual module structure
except ImportError:
    # For development/direct execution
    import beat_generation

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("beatbeast-generate")

# Define available beat styles
AVAILABLE_STYLES = [
    "hip-hop",
    "trap",
    "electronic",
    "house",
    "techno",
    "rock",
    "jazz",
    "lo-fi",
    "drill",
    "reggaeton",
]

# Define available export formats
EXPORT_FORMATS = ["wav", "mp3", "midi", "json"]

def parse_arguments(args: List[str]) -> argparse.Namespace:
    """
    Parse command line arguments for beat generation.
    
    Args:
        args: Command line arguments
        
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Generate beats with customizable styles and patterns",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        required=True,
        help="Output file path (extension determines format unless --format is specified)"
    )
    
    # Beat style options
    style_group = parser.add_argument_group("Beat Style Options")
    style_group.add_argument(
        "-s", "--style",
        type=str,
        choices=AVAILABLE_STYLES,
        default="hip-hop",
        help="Beat style to generate"
    )
    style_group.add_argument(
        "--bpm",
        type=int,
        default=90,
        help="Beats per minute (tempo)"
    )
    style_group.add_argument(
        "--variation",
        type=float,
        default=0.3,
        help="Amount of variation to introduce (0.0-1.0)"
    )
    
    # Pattern options
    pattern_group = parser.add_argument_group("Pattern Options")
    pattern_group.add_argument(
        "--pattern-file",
        type=str,
        help="Load a specific pattern from a file"
    )
    pattern_group.add_argument(
        "--pattern-complexity",
        type=float,
        default=0.5,
        help="Complexity of the generated pattern (0.0-1.0)"
    )
    pattern_group.add_argument(
        "--bars",
        type=int,
        default=4,
        help="Number of bars to generate"
    )
    pattern_group.add_argument(
        "--swing",
        type=float,
        default=0.0,
        help="Swing amount to apply (0.0-1.0)"
    )
    
    # Export options
    export_group = parser.add_argument_group("Export Options")
    export_group.add_argument(
        "--format",
        type=str,
        choices=EXPORT_FORMATS,
        help="Export format (overrides file extension)"
    )
    export_group.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="Sample rate for audio exports"
    )
    export_group.add_argument(
        "--bit-depth",
        type=int,
        default=16,
        choices=[16, 24, 32],
        help="Bit depth for audio exports"
    )
    
    # Advanced options
    advanced_group = parser.add_argument_group("Advanced Options")
    advanced_group.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible generation"
    )
    advanced_group.add_argument(
        "--layers",
        type=str,
        nargs="+",
        choices=["kick", "snare", "hihat", "percussion", "bass", "melody", "all"],
        default=["all"],
        help="Specific layers to generate"
    )
    advanced_group.add_argument(
        "--intensity",
        type=float,
        default=0.7,
        help="Overall intensity of the beat (0.0-1.0)"
    )
    advanced_group.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args(args)

def determine_export_format(args: argparse.Namespace) -> str:
    """
    Determine the export format based on the output file extension or format argument.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Export format string
    """
    if args.format:
        return args.format
    
    # Extract extension from output path
    output_path = Path(args.output)
    extension = output_path.suffix.lower().lstrip(".")
    
    if extension in EXPORT_FORMATS:
        return extension
    
    # Default to WAV if extension is not recognized
    logger.warning(f"Unrecognized extension '{extension}', defaulting to 'wav'")
    return "wav"

def validate_arguments(args: argparse.Namespace) -> None:
    """
    Validate the parsed arguments for consistency and correctness.
    
    Args:
        args: Parsed command-line arguments
        
    Raises:
        ValueError: If arguments are invalid
    """
    # Validate BPM range
    if args.bpm < 40 or args.bpm > 300:
        raise ValueError(f"BPM must be between 40 and 300, got {args.bpm}")
    
    # Validate value ranges
    if not 0 <= args.variation <= 1:
        raise ValueError(f"Variation must be between 0 and 1, got {args.variation}")
    
    if not 0 <= args.pattern_complexity <= 1:
        raise ValueError(f"Pattern complexity must be between 0 and 1, got {args.pattern_complexity}")
    
    if not 0 <= args.swing <= 1:
        raise ValueError(f"Swing must be between 0 and 1, got {args.swing}")
    
    if not 0 <= args.intensity <= 1:
        raise ValueError(f"Intensity must be between 0 and 1, got {args.intensity}")
    
    # Validate output directory exists
    output_dir = Path(args.output).parent
    if not output_dir.exists():
        raise ValueError(f"Output directory does not exist: {output_dir}")
    
    # Validate pattern file if provided
    if args.pattern_file and not Path(args.pattern_file).exists():
        raise ValueError(f"Pattern file does not exist: {args.pattern_file}")

def prepare_generation_parameters(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Prepare parameters for beat generation based on parsed arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Dictionary of parameters for beat generation
    """
    params = {
        "style": args.style,
        "bpm": args.bpm,
        "variation": args.variation,
        "pattern_complexity": args.pattern_complexity,
        "bars": args.bars,
        "swing": args.swing,
        "intensity": args.intensity,
    }
    
    # Add optional parameters if provided
    if args.seed is not None:
        params["seed"] = args.seed
    
    if args.layers != ["all"]:
        params["layers"] = args.layers
    
    if args.pattern_file:
        params["pattern_file"] = args.pattern_file
    
    return params

def generate_beat(params: Dict[str, Any]) -> Any:
    """
    Generate a beat using the provided parameters.
    
    Args:
        params: Dictionary of generation parameters
        
    Returns:
        Generated beat object (implementation-specific)
    """
    # This function would call into the actual beat generation implementation
    # For now, we'll just log the parameters and return a placeholder
    logger.info(f"Generating beat with parameters: {params}")
    
    # Placeholder for actual beat generation
    # In a real implementation, this would call the actual generation logic
    # return beat_generation.generate(**params)
    
    # For demonstration purposes only
    class PlaceholderBeat:
        def __init__(self, params):
            self.params = params
            
        def __str__(self):
            return f"Beat({self.params['style']}, {self.params['bpm']} BPM)"
            
    return PlaceholderBeat(params)

def export_beat(beat: Any, output_path: str, export_format: str, args: argparse.Namespace) -> None:
    """
    Export the generated beat to the specified format and path.
    
    Args:
        beat: The generated beat object
        output_path: Path to export the beat to
        export_format: Format to export the beat in
        args: Original parsed arguments for additional export options
    """
    # This function would call into the actual export implementation
    # For now, we'll just log the export details
    logger.info(f"Exporting beat to {output_path} in {export_format} format")
    logger.info(f"Export settings: Sample rate={args.sample_rate}, Bit depth={args.bit_depth}")
    
    # Placeholder for actual export logic
    # In a real implementation, this would call the actual export functions
    # if export_format == "wav":
    #     beat_generation.export.to_wav(beat, output_path, sample_rate=args.sample_rate, bit_depth=args.bit_depth)
    # elif export_format == "mp3":
    #     beat_generation.export.to_mp3(beat, output_path, sample_rate=args.sample_rate)
    # elif export_format == "midi":
    #     beat_generation.export.to_midi(beat, output_path)
    # elif export_format == "json":
    #     beat_generation.export.to_json(beat, output_path)

def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point for the beat generation CLI.
    
    Args:
        argv: Command line arguments (defaults to sys.argv[1:])
        
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    if argv is None:
        argv = sys.argv[1:]
    
    try:
        # Parse and validate arguments
        args = parse_arguments(argv)
        
        # Set logging level based on verbosity
        if args.verbose:
            logger.setLevel(logging.DEBUG)
            logger.debug("Verbose mode enabled")
        
        # Validate arguments
        validate_arguments(args)
        
        # Determine export format
        export_format = determine_export_format(args)
        
        # Prepare generation parameters
        params = prepare_generation_parameters(args)
        
        # Generate the beat
        beat = generate_beat(params)
        
        # Export the beat
        export_beat(beat, args.output, export_format, args)
        
        logger.info(f"Successfully generated and exported {args.style} beat to {args.output}")
        return 0
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return 1
    except ImportError as e:
        logger.error(f"Import error: {e}\nPlease ensure BeatProductionBeast is correctly installed.")
        return 2
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 3

if __name__ == "__main__":
    sys.exit(main())

