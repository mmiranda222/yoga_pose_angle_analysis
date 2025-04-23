#!/usr/bin/env python3
"""
Yoga Pose Angle Analysis System - Main Script

This script is the entry point for the Yoga Pose Angle Analysis System.
It provides options for analyzing yoga poses in images, visualizing results,
and debugging individual poses.

Usage:
    python main.py --mode analyze --input_folder /path/to/images --output_csv results.csv
    python main.py --mode visualize --input_csv results.csv
    python main.py --mode debug --input_folder /path/to/images
"""

import os
import argparse
import sys
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from yoga_pose_analyzer import YogaPoseAnalyzer
from debug_tool import YogaPoseDebugger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Yoga Pose Analysis System')
    
    parser.add_argument('--mode', type=str, required=True, choices=['analyze', 'visualize', 'debug'],
                        help='Operation mode: analyze images, visualize results, or debug poses')
    
    parser.add_argument('--input_folder', type=str,
                        help='Path to the folder containing yoga pose images (for analyze and debug modes)')
    
    parser.add_argument('--output_csv', type=str, default='yoga_pose_analysis_results.csv',
                        help='Path to save the output CSV file (for analyze mode)')
    
    parser.add_argument('--input_csv', type=str,
                        help='Path to the CSV file with analysis results (for visualize mode)')
    
    parser.add_argument('--output_folder', type=str, default='results',
                        help='Path to save visualization outputs (for visualize mode)')

    parser.add_argument('--debug_output', action='store_true',
                        help='Enable detailed debugging output (print per-image debug info in debug mode)')

    
    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.mode == 'analyze' and not args.input_folder:
        parser.error("--input_folder is required for 'analyze' mode")
    
    if args.mode == 'visualize' and not args.input_csv:
        parser.error("--input_csv is required for 'visualize' mode")
    
    if args.mode == 'debug' and not args.input_folder:
        parser.error("--input_folder is required for 'debug' mode")
    
    return args


def analyze_poses(input_folder: str, output_csv: str):
    """
    Analyze yoga poses in images and save results to CSV.
    
    Args:
        input_folder: Path to the folder containing images
        output_csv: Path to save the analysis results
    """
    print(f"Analyzing yoga poses from {input_folder}")
    print(f"Results will be saved to {output_csv}")
    
    try:
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        
        # Initialize and run the analyzer
        analyzer = YogaPoseAnalyzer(
            image_folder=input_folder,
            output_csv=output_csv
        )
        analyzer.process_images()
        
        print(f"Analysis completed successfully! Results saved to {output_csv}")
        
        # Show a summary of the results
        if os.path.exists(output_csv):
            df = pd.read_csv(output_csv)
            # Summary of filename and angle only
            print("\nAnalysis Summary:")
            print(f"Total images processed: {len(df)}")
            if 'angle' in df.columns:
                num_missing = df['angle'].isna().sum()
                print(f"Images with missing angle: {num_missing}")
                valid_angles = df['angle'].dropna()
                if len(valid_angles) > 0:
                    print(f"Angle stats: min {valid_angles.min():.2f}, max {valid_angles.max():.2f}, mean {valid_angles.mean():.2f}")
            else:
                print("No 'angle' column found to summarize.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return False
    
    return True


def visualize_results(input_csv: str, output_folder: str):
    """
    Generate visualizations of analysis results.
    
    Args:
        input_csv: Path to the CSV file with analysis results
        output_folder: Path to save visualization outputs
    """
    print(f"Generating visualizations from {input_csv}")
    print(f"Visualizations will be saved to {output_folder}")
    
    try:
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)
        
        # Read the results
        df = pd.read_csv(input_csv)
        
        if df.empty:
            print("The results file is empty. No visualizations to generate.")
            return False
        
        # 1. Distribution of poses
        plt.figure(figsize=(12, 6))
        ax = sns.countplot(x='pose_type', data=df)
        plt.title('Distribution of Yoga Poses')
        plt.xlabel('Pose Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 10), textcoords='offset points')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'pose_distribution.png'))
        plt.close()
        
        # 2. Angle distribution by pose type
        plt.figure(figsize=(12, 6))
        df_with_angle = df.dropna(subset=['angle'])
        if not df_with_angle.empty:
            ax = sns.boxplot(x='pose_type', y='angle', data=df_with_angle)
            plt.title('Angle Distribution by Pose Type')
            plt.xlabel('Pose Type')
            plt.ylabel('Angle (degrees)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, 'angle_distribution.png'))
        plt.close()
        
        # 3. Active vs. Passive distribution
        plt.figure(figsize=(10, 6))
        df_variant = df[df['pose_variant'].isin(['active', 'passive'])]
        if not df_variant.empty:
            ax = sns.countplot(x='pose_type', hue='pose_variant', data=df_variant)
            plt.title('Active vs. Passive Distribution by Pose Type')
            plt.xlabel('Pose Type')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, 'variant_distribution.png'))
        plt.close()
        
        # 4. Side distribution (left vs. right)
        plt.figure(figsize=(10, 6))
        df_side = df[df['pose_side'].isin(['left', 'right'])]
        if not df_side.empty:
            ax = sns.countplot(x='pose_type', hue='pose_side', data=df_side)
            plt.title('Left vs. Right Distribution by Pose Type')
            plt.xlabel('Pose Type')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, 'side_distribution.png'))
        plt.close()
        
        # 5. Angle histogram for each pose type
        pose_types = df['pose_type'].unique()
        for pose in pose_types:
            if pose == 'unknown':
                continue
                
            pose_df = df[df['pose_type'] == pose].dropna(subset=['angle'])
            if len(pose_df) > 0:
                plt.figure(figsize=(10, 6))
                sns.histplot(data=pose_df, x='angle', bins=15, kde=True)
                plt.title(f'Angle Distribution for {pose}')
                plt.xlabel('Angle (degrees)')
                plt.ylabel('Count')
                plt.tight_layout()
                plt.savefig(os.path.join(output_folder, f'angle_histogram_{pose}.png'))
                plt.close()
        
        print(f"Visualizations generated successfully in {output_folder}")
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        return False
    
    return True


def debug_poses(input_folder: str, debug_output: bool):
    """
    Launch the debugging tool for yoga poses.
    
    Args:
        input_folder: Path to the folder containing images
        debug_output: Enable detailed debugging output
    """
    # Initialize analyzer for debug
    analyzer = YogaPoseAnalyzer(
        image_folder=input_folder,
        output_csv='yoga_pose_analysis_results.csv'
    )

    print(f"Launching debugging tool for images in {input_folder}")

    if debug_output:
        print("Running with detailed debugging output...")
        for img_file in os.listdir(input_folder):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(input_folder, img_file)
                analyzer.analyze_image_debug(img_path, img_file)
    else:
        analyzer.process_images()
    try:
        debugger = YogaPoseDebugger(input_folder)
        debugger.show()
    except Exception as e:
        print(f"Error launching debug tool: {e}")
        return False
    
    return True


def main():
    """Main function to run the Yoga Pose Analysis System."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Execute the selected mode
    if args.mode == 'analyze':
        success = analyze_poses(args.input_folder, args.output_csv)
    elif args.mode == 'visualize':
        success = visualize_results(args.input_csv, args.output_folder)
    elif args.mode == 'debug':
        success = debug_poses(args.input_folder, args.debug_output)
    else:
        print(f"Unknown mode: {args.mode}")
        success = False
    
    # Return appropriate exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()