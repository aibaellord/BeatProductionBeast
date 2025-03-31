#!/usr/bin/env python
"""
Command Line Interface for BeatProductionBeast.

This module provides a command-line interface for accessing the various 
functionalities of the BeatProductionBeast package, including beat generation,
style analysis, and audio processing.
"""

import argparse
import sys
from typing import List, Optional

# Import package version
try:
    from importlib.metadata import version
    __version__ = version("BeatProductionBeast")
except ImportError:
    __version__ = "0.1.0"  # Fallback version

# Import functionality modules
try:
    from src import beat_generation, style_analysis, audio_engine, neural_processing
except ImportError:
    # When installed as a package
    from beatproductionbeast import beat_generation, style_analysis, audio_engine, neural_processing


def create_parser() -> argparse.ArgumentParser:
    """Create the command line argument parser."""
    parser = argparse.ArgumentParser(
        prog="beatbeast",
        description="BeatProductionBeast - AI-powered music production toolkit",
        epilog="For more information visit: https://github.com/yourusername/BeatProductionBeast"
    )
    
    parser.add_argument(
        "-v", "--version", 
        action="version", 
        version=f"BeatProductionBeast v{__version__}"
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        help="Command to execute"
    )
    
    # Beat Generation command
    generate_parser = subparsers.add_parser(
        "generate", 
        help="Generate beats with AI assistance"
    )
    generate_parser.add_argument(
        "-s", "--style",
        help="Music style (e.g., hip-hop, edm, lo-fi)",
        default="hip-hop"
    )
    generate_parser.add_argument(
        "-b", "--bpm",
        type=int,
        help="Beats per minute",
        default=90
    )
    generate_parser.add_argument(
        "-d", "--duration",
        type=float,
        help="Duration in seconds",
        default=30.0
    )
    generate_parser.add_argument(
        "-o", "--output",
        help="Output file path",
        default="output.wav"
    )
    
    # Style Analysis command
    analyze_parser = subparsers.add_parser(
        "analyze", 
        help="Analyze music style from audio files"
    )
    analyze_parser.add_argument(
        "input_file",
        help="Audio file to analyze"
    )
    analyze_parser.add_argument(
        "-d", "--detailed",
        action="store_true",
        help="Provide detailed analysis"
    )
    
    # Audio Processing command
    process_parser = subparsers.add_parser(
        "process", 
        help="Apply audio processing effects"
    )
    process_parser.add_argument(
        "input_file",
        help="Audio file to process"
    )
    process_parser.add_argument(
        "-e", "--effects",
        nargs="+",
        help="Effects to apply (e.g., reverb, delay, eq)",
        default=[]
    )
    process_parser.add_argument(
        "-o", "--output",
        help="Output file path",
        default="processed.wav"
    )
    
    # Neural Processing command
    neural_parser = subparsers.add_parser(
        "enhance", 
        help="Apply neural enhancement to audio"
    )
    neural_parser.add_argument(
        "input_file",
        help="Audio file to enhance"
    )
    neural_parser.add_argument(
        "-m", "--model",
        help="Neural model to use",
        default="default"
    )
    neural_parser.add_argument(
        "-o", "--output",
        help="Output file path",
        default="enhanced.wav"
    )
    
    return parser


def handle_generate(args: argparse.Namespace) -> int:
    """Handle beat generation command."""
    print(f"Generating {args.duration}s {args.style} beat at {args.bpm} BPM...")
    try:
        # This would be replaced with actual functionality
        beat_generation.generate_beat(
            style=args.style,
            bpm=args.bpm,
            duration=args.duration,
            output_path=args.output
        )
        print(f"Beat generated successfully and saved to {args.output}")
        return 0
    except Exception as e:
        print(f"Error generating beat: {e}", file=sys.stderr)
        return 1


def handle_analyze(args: argparse.Namespace) -> int:
    """Handle style analysis command."""
    print(f"Analyzing {args.input_file}...")
    try:
        # This would be replaced with actual functionality
        results = style_analysis.analyze_audio(
            audio_path=args.input_file,
            detailed=args.detailed
        )
        
        print("\nAnalysis Results:")
        print("-----------------")
        for key, value in results.items():
            print(f"{key}: {value}")
        return 0
    except Exception as e:
        print(f"Error analyzing audio: {e}", file=sys.stderr)
        return 1


def handle_process(args: argparse.Namespace) -> int:
    """Handle audio processing command."""
    print(f"Processing {args.input_file} with effects: {', '.join(args.effects)}")
    try:
        # This would be replaced with actual functionality
        audio_engine.process_audio(
            input_path=args.input_file,
            effects=args.effects,
            output_path=args.output
        )
        print(f"Audio processed successfully and saved to {args.output}")
        return 0
    except Exception as e:
        print(f"Error processing audio: {e}", file=sys.stderr)
        return 1


def handle_enhance(args: argparse.Namespace) -> int:
    """Handle neural enhancement command."""
    print(f"Enhancing {args.input_file} using {args.model} model...")
    try:
        # This would be replaced with actual functionality
        neural_processing.enhance_audio(
            input_path=args.input_file,
            model=args.model,
            output_path=args.output
        )
        print(f"Audio enhanced successfully and saved to {args.output}")
        return 0
    except Exception as e:
        print(f"Error enhancing audio: {e}", file=sys.stderr)
        return 1


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the BeatProductionBeast CLI.
    
    Args:
        args: Command line arguments (uses sys.argv if None)
        
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    
    # If no command is provided, show help
    if not parsed_args.command:
        parser.print_help()
        return 0
    
    # Dispatch to appropriate handler
    if parsed_args.command == "generate":
        return handle_generate(parsed_args)
    elif parsed_args.command == "analyze":
        return handle_analyze(parsed_args)
    elif parsed_args.command == "process":
        return handle_process(parsed_args)
    elif parsed_args.command == "enhance":
        return handle_enhance(parsed_args)
    
    # Should never reach here
    return 0


if __name__ == "__main__":
    sys.exit(main())

