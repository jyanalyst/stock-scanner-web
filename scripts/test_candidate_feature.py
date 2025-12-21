#!/usr/bin/env python3
"""
Test New Candidate Features (Part B)

Simplified workflow for testing archived/new features:
1. Register feature in config
2. Calculate for all historical dates
3. Run basic statistical screening
4. Output PASS/FAIL recommendation

Usage:
    python scripts/test_candidate_feature.py --feature FEATURE_NAME [--config CONFIG_PATH]

Arguments:
    --feature: Name of the feature to test (required)
    --config: Path to feature config file (default: configs/feature_config.yaml)
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime
import argparse
import logging
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pages.scanner.feature_lab.feature_tracker import FeatureTracker
from utils.statistical_tests import analyze_feature_complete

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main candidate feature testing function."""
    parser = argparse.ArgumentParser(description="Test candidate feature")
    parser.add_argument(
        "--feature",
        type=str,
        required=True,
        help="Name of the feature to test"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/feature_config.yaml",
        help="Path to feature config file"
    )

    args = parser.parse_args()

    try:
        feature_name = args.feature
        config_path = Path(args.config)

        logger.info(f"Starting candidate feature test for: {feature_name}")

        # Validate feature exists in config
        if not validate_feature_config(feature_name, config_path):
            logger.error(f"Feature '{feature_name}' not found in config: {config_path}")
            sys.exit(1)

        # Initialize feature tracker
        tracker = FeatureTracker()

        # Start feature tracking
        logger.info("Initializing feature tracking...")
        if not tracker.start_feature_tracking(feature_name):
            logger.error("Failed to initialize feature tracking")
            sys.exit(1)

        # Calculate feature for all historical dates
        logger.info("Calculating feature for all historical dates...")
        if not tracker.calculate_feature_for_all_history(feature_name):
            logger.error("Failed to calculate feature for historical dates")
            sys.exit(1)

        # Run basic statistical analysis
        logger.info("Running statistical analysis...")
        analysis_result = tracker.analyze_feature_significance(feature_name)

        if 'error' in analysis_result:
            logger.error(f"Analysis failed: {analysis_result['error']}")
            sys.exit(1)

        # Generate screening report
        screening_result = generate_screening_report(feature_name, analysis_result)

        # Print results to console
        print_screening_results(feature_name, screening_result)

        # Update features_testing.json with results
        update_testing_status(tracker, feature_name, screening_result)

        logger.info("✅ Candidate feature testing complete!")

    except Exception as e:
        logger.error(f"Unexpected error during candidate testing: {e}")
        sys.exit(1)


def validate_feature_config(feature_name: str, config_path: Path) -> bool:
    """Validate that the feature exists in the config file."""
    try:
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return False

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        experimental_features = config.get('experimental_features', {})
        return feature_name in experimental_features

    except Exception as e:
        logger.error(f"Error reading config file: {e}")
        return False


def generate_screening_report(feature_name: str, analysis_result: dict) -> dict:
    """
    Generate simplified PASS/FAIL screening report.

    Uses relaxed criteria for initial screening:
    - p < 0.10 (vs 0.05 for production)
    - |Cohen's d| > 0.15 (vs 0.20 for production)
    - Max correlation < 0.85 (same as production)
    """

    # Extract key metrics
    p_value = analysis_result.get('mann_whitney_p')
    cohens_d = analysis_result.get('cohens_d')
    max_correlation = analysis_result.get('correlation_max')

    # Apply screening criteria (relaxed thresholds)
    passes_significance = p_value is not None and p_value < 0.10
    passes_effect_size = cohens_d is not None and abs(cohens_d) > 0.15
    passes_correlation = max_correlation is not None and abs(max_correlation) < 0.85

    # Overall recommendation
    if passes_significance and passes_effect_size and passes_correlation:
        recommendation = "PASS"
        reasoning = "Meets all screening criteria - proceed to enhanced validation"
    elif not passes_significance:
        recommendation = "FAIL"
        reasoning = "Not statistically significant (p >= 0.10)"
    elif not passes_effect_size:
        recommendation = "FAIL"
        reasoning = "Effect size too small (|Cohen's d| <= 0.15)"
    elif not passes_correlation:
        recommendation = "FAIL"
        reasoning = "Too highly correlated with existing features"
    else:
        recommendation = "REVIEW"
        reasoning = "Mixed results - manual review recommended"

    return {
        "feature_name": feature_name,
        "screening_timestamp": datetime.now().isoformat(),
        "criteria": {
            "significance_threshold": 0.10,
            "effect_size_threshold": 0.15,
            "correlation_threshold": 0.85
        },
        "results": {
            "p_value": p_value,
            "cohens_d": cohens_d,
            "max_correlation": max_correlation,
            "passes_significance": passes_significance,
            "passes_effect_size": passes_effect_size,
            "passes_correlation": passes_correlation
        },
        "recommendation": recommendation,
        "reasoning": reasoning,
        "next_steps": get_next_steps(recommendation)
    }


def get_next_steps(recommendation: str) -> str:
    """Get next steps based on screening recommendation."""
    if recommendation == "PASS":
        return "Proceed to enhanced validation (Part A) for detailed analysis"
    elif recommendation == "FAIL":
        return "Archive feature or investigate why it failed screening"
    elif recommendation == "REVIEW":
        return "Manual review required - check data quality and feature calculation"
    else:
        return "Unexpected recommendation - check analysis results"


def print_screening_results(feature_name: str, screening_result: dict):
    """Print screening results to console."""
    print("\n" + "="*70)
    print(f"CANDIDATE FEATURE SCREENING: {feature_name}")
    print("="*70)

    results = screening_result.get('results', {})
    criteria = screening_result.get('criteria', {})

    print("\nSCREENING CRITERIA:")
    sig_threshold = criteria.get('significance_threshold', 'N/A')
    effect_threshold = criteria.get('effect_size_threshold', 'N/A')
    corr_threshold = criteria.get('correlation_threshold', 'N/A')

    print(f"  Statistical Significance: p < {sig_threshold}")
    print(f"  Effect Size: |Cohen's d| > {effect_threshold}")
    print(f"  Max Correlation: < {corr_threshold}")

    print("
RESULTS:")
    print(f"  P-value: {results.get('p_value', 'N/A'):.4f}")
    print(f"  Cohen's d: {results.get('cohens_d', 'N/A'):.3f}")
    print(f"  Max Correlation: {results.get('max_correlation', 'N/A'):.3f}")

    print("\nCRITERIA CHECK:")
    significance_status = '✅ PASS' if results.get('passes_significance') else '❌ FAIL'
    effect_size_status = '✅ PASS' if results.get('passes_effect_size') else '❌ FAIL'
    correlation_status = '✅ PASS' if results.get('passes_correlation') else '❌ FAIL'

    print(f"  Significance: {significance_status}")
    print(f"  Effect Size: {effect_size_status}")
    print(f"  Correlation: {correlation_status}")

    recommendation = screening_result.get('recommendation', 'UNKNOWN')
    reasoning = screening_result.get('reasoning', 'N/A')
    next_steps = screening_result.get('next_steps', 'N/A')

    print(f"\nRECOMMENDATION: {recommendation}")
    print(f"Reasoning: {reasoning}")
    print(f"Next Steps: {next_steps}")

    print("\n" + "="*70)


def update_testing_status(tracker: FeatureTracker, feature_name: str, screening_result: dict):
    """Update the features_testing.json with screening results."""
    try:
        # Load current testing data
        testing_path = tracker.data_dir / "features_testing.json"
        if not testing_path.exists():
            logger.warning("features_testing.json not found - creating new file")
            testing_data = {
                "version": "1.0",
                "created_date": datetime.now().isoformat(),
                "features": {}
            }
        else:
            with open(testing_path, 'r') as f:
                testing_data = yaml.safe_load(f) if testing_path.suffix == '.yaml' else json.load(f)

        # Update feature status
        if feature_name not in testing_data.get('features', {}):
            testing_data['features'][feature_name] = {}

        testing_data['features'][feature_name].update({
            "status": "screened",
            "screening_result": screening_result,
            "screened_at": datetime.now().isoformat()
        })

        testing_data["last_modified"] = datetime.now().isoformat()

        # Save updated data
        with open(testing_path, 'w') as f:
            json.dump(testing_data, f, indent=2)

        logger.info(f"Updated testing status for {feature_name}")

    except Exception as e:
        logger.error(f"Failed to update testing status: {e}")


if __name__ == "__main__":
    main()
