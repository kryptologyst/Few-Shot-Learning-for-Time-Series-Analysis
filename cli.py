#!/usr/bin/env python3
"""
Command Line Interface for Few-Shot Time Series Learning

This script provides a command-line interface for running the analysis
with various options and configurations.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from timeseries_analyzer import TimeSeriesAnalyzer


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Few-Shot Time Series Learning Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py --config config/config.yaml
  python cli.py --quick --architecture lstm
  python cli.py --interactive --output results/
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file (default: config/config.yaml)"
    )
    
    parser.add_argument(
        "--architecture",
        type=str,
        choices=["conv", "lstm", "gru", "transformer"],
        default="conv",
        help="Neural architecture to use (default: conv)"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick analysis with minimal samples"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive Streamlit interface"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="notebooks/",
        help="Output directory for results (default: notebooks/)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    if args.interactive:
        print("Launching Streamlit interface...")
        os.system("streamlit run src/streamlit_app.py")
        return
    
    print("="*60)
    print("FEW-SHOT TIME SERIES LEARNING ANALYSIS")
    print("="*60)
    
    try:
        # Initialize analyzer
        analyzer = TimeSeriesAnalyzer(args.config)
        
        # Modify config for quick run if requested
        if args.quick:
            print("Running quick analysis...")
            analyzer.config["data"]["num_support_samples"] = 2
            analyzer.config["data"]["num_query_samples"] = 3
            analyzer.config["data"]["sequence_length"] = 20
        
        # Update architecture if specified
        if args.architecture != "conv":
            print(f"Using {args.architecture.upper()} architecture...")
            # Note: Architecture selection would need to be implemented in the analyzer
        
        # Run analysis
        print("Starting analysis...")
        results = analyzer.run_analysis()
        
        # Print results
        print("\nResults:")
        print(f"  Accuracy: {results['accuracy']:.2%}")
        print(f"  Precision: {results['precision']:.2%}")
        print(f"  Recall: {results['recall']:.2%}")
        print(f"  F1-Score: {results['f1_score']:.2%}")
        
        # Create visualizations
        print("\nCreating visualizations...")
        analyzer.create_visualizations(results)
        
        print(f"\nAnalysis completed successfully!")
        print(f"Results saved to: {args.output}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
