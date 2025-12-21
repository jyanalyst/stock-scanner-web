#!/usr/bin/env python3
"""
Validate Current Scoring Features (Part A)

Runs enhanced statistical validation on the 7 features currently used in scoring:
- Overall significance testing
- Directional analysis (bullish vs bearish)
- Quintile analysis
- Win rate metrics
- Actionable recommendations

Usage:
    python scripts/validate_scoring_features.py [--output OUTPUT_PATH]

Arguments:
    --output: Optional output path for the report (default: auto-generated timestamp)
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import argparse
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pages.scanner.feature_lab.feature_tracker import FeatureTracker

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate current scoring features")
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for validation report (optional)",
        default=None
    )

    args = parser.parse_args()

    try:
        logger.info("Starting enhanced validation of scoring features...")

        # Initialize feature tracker
        tracker = FeatureTracker()

        # Run enhanced validation on all scoring features
        logger.info("Running comprehensive validation analysis...")
        validation_report = tracker.validate_all_scoring_features()

        if 'error' in validation_report:
            logger.error(f"Validation failed: {validation_report['error']}")
            sys.exit(1)

        # Export the report
        logger.info("Generating validation report...")
        report_path = tracker.export_validation_report(validation_report, args.output)

        if report_path:
            logger.info(f"✅ Validation complete! Report saved to: {report_path}")

            # Print summary to console
            print_validation_summary(validation_report)

        else:
            logger.error("Failed to export validation report")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Unexpected error during validation: {e}")
        sys.exit(1)


def print_validation_summary(report_data: dict):
    """Print a concise summary of validation results to console."""
    print("\n" + "="*80)
    print("SCORING FEATURE VALIDATION REPORT")
    print("="*80)

    summary = report_data.get('summary_statistics', {})
    insights = report_data.get('insights', [])

    # Summary stats
    print("\nSUMMARY STATISTICS:")
    print(f"  Features Analyzed: {summary.get('analyzed_features', 0)}/{summary.get('total_features', 0)}")
    print(f"  Keep: {summary.get('keep_features', 0)}")
    print(f"  Review: {summary.get('review_features', 0)}")
    print(f"  Remove: {summary.get('remove_features', 0)}")
    print(f"  Bullish Only: {summary.get('bullish_only_features', 0)}")
    print(f"  Bearish Only: {summary.get('bearish_only_features', 0)}")

    # Key insights
    if insights:
        print("\nKEY INSIGHTS:")
        for insight in insights:
            print(f"  • {insight}")

    # Individual feature results
    validation_results = report_data.get('validation_results', {})
    print("\nINDIVIDUAL FEATURE RESULTS:")
    print("-" * 80)
    print(f"{'Feature':<20} {'Overall':<8} {'Bullish':<8} {'Bearish':<8} {'Quintile':<8} {'Recommendation'}")
    print("-" * 80)

    for feature_name, results in validation_results.items():
        if 'error' in results:
            print(f"{feature_name:<20} {'ERROR':<8} {'-':<8} {'-':<8} {'-':<8} {'Check logs'}")
            continue

        overall_sig = "✓" if results.get('overall_mann_whitney_significant') else "✗"
        bullish_sig = "✓" if results.get('directional_analysis', {}).get('bullish', {}).get('mann_whitney_significant') else "✗"
        bearish_sig = "✓" if results.get('directional_analysis', {}).get('bearish', {}).get('mann_whitney_significant') else "✗"

        monotonic = "✓" if results.get('quintile_analysis', {}).get('monotonic') else "✗"
        recommendation = results.get('recommendation', 'UNKNOWN')

        print(f"{feature_name:<20} {overall_sig:<8} {bullish_sig:<8} {bearish_sig:<8} {monotonic:<8} {recommendation}")

    # Recommendations
    recommendations = report_data.get('recommendations', {})
    immediate_actions = recommendations.get('immediate_actions', [])

    if immediate_actions:
        print("\nIMMEDIATE ACTIONS REQUIRED:")
        for action in immediate_actions:
            print(f"  • {action}")

    print("\n" + "="*80)
    print(f"Report generated: {report_data.get('generated_at', 'Unknown')}")
    print("="*80)


if __name__ == "__main__":
    main()
