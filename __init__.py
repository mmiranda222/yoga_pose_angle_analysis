"""
Yoga Pose Angle Analysis System

A system for analyzing yoga poses in images to identify the pose type,
and calculate angles between specific landmarks for each pose.

This module provides:
1. YogaPoseAnalyzer - The main class for analyzing yoga poses in images
2. Debugging tools for pose detection
3. Visualization utilities for angle analysis
"""

from .yoga_pose_analyzer import YogaPoseAnalyzer
from .debug_tool import YogaPoseDebugger

__version__ = "0.1.0"
__author__ = "Manuela Miranda"