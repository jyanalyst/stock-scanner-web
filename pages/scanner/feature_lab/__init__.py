"""
Feature Lab - Style Learning System for Stock Scanner

This module provides tools to learn and optimize trading style by:
1. Recording historical winner selections
2. Analyzing patterns in winners vs non-winners
3. Optimizing scoring weights to match trading preferences
4. Validating improvements through forward testing
"""

from .feature_tracker import FeatureTracker

__all__ = ['FeatureTracker']
