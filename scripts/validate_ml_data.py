#!/usr/bin/env python3
"""
CLI Script for ML Data Quality Validation
Run this before starting ML training to ensure data quality
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml.data_validator import MLDataValidator
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Validate data quality for ML training')
    parser.add_argument('--start-date', default='2023-01-01', help='Start date for validation (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2024-12-31', help='End date for validation (YYYY-MM-DD)')
    parser.add_argument('--min-days', type=int, default=100, help='Minimum trading days required per stock')
    parser.add_argument('--output', default='data/ml_training/validation_report.html', help='Output report path')
    parser.add_argument('--quiet', action='store_true', help='Suppress detailed output')

    args = parser.parse_args()

    try:
        logger.info(f"Starting ML data validation: {args.start_date} to {args.end_date}")

        # Initialize validator
        validator = MLDataValidator(
            start_date=args.start_date,
            end_date=args.end_date,
            min_days=args.min_days
        )

        # Run validation
        results = validator.validate_all_stocks()

        # Generate report
        report_path = validator.generate_report(results, args.output)

        # Print summary
        summary = results['summary']
        print("\n" + "="*60)
        print("ML DATA QUALITY VALIDATION REPORT")
        print("="*60)
        print(f"Date Range: {summary['date_range']}")
        print(f"Expected Trading Days: {summary['expected_trading_days']}")
        print(f"Total Stocks: {summary['total_stocks']}")
        print()
        print("RESULTS:")
        print(f"  ✅ Ready:     {summary['ready_count']:3d} stocks ({summary['ready_count']/summary['total_stocks']:.1%})")
        print(f"  ⚠️  Partial:   {summary['partial_count']:3d} stocks ({summary['partial_count']/summary['total_stocks']:.1%})")
        print(f"  ❌ Failed:    {summary['failed_count']:3d} stocks ({summary['failed_count']/summary['total_stocks']:.1%})")
        print()
        print("ESTIMATED ML SAMPLES:")
        print(f"  Ready only:     {summary['estimated_samples']['ready_only']:6,d}")
        print(f"  With partial:   {summary['estimated_samples']['including_partial']:6,d}")
        print()

        # Show issues
        issues = results['issues']
        if any(issues.values()):
            print("ISSUES DETECTED:")
            if issues['missing_files']:
                print(f"  Missing files: {len(issues['missing_files'])} stocks")
                if not args.quiet:
                    for stock in issues['missing_files'][:5]:  # Show first 5
                        print(f"    - {stock}")
                    if len(issues['missing_files']) > 5:
                        print(f"    ... and {len(issues['missing_files'])-5} more")

            if issues['insufficient_data']:
                print(f"  Insufficient data: {len(issues['insufficient_data'])} stocks")
            if issues['data_gaps']:
                print(f"  Data gaps: {len(issues['data_gaps'])} stocks")
            if issues['null_values']:
                print(f"  NULL values: {len(issues['null_values'])} stocks")
            if issues['integrity_issues']:
                print(f"  Integrity issues: {len(issues['integrity_issues'])} stocks")
            print()

        print("RECOMMENDATIONS:")
        if summary['ready_count'] > 0:
            print(f"  ✓ Proceed with {summary['ready_count']} ready stocks")
            print(f"    Expected samples: {summary['estimated_samples']['ready_only']:,}")
        else:
            print("  ❌ No stocks ready for ML training")
            print("    Fix data issues before proceeding")

        if summary['partial_count'] > 0:
            print(f"  ⚠️  Consider {summary['partial_count']} partial stocks if more data needed")

        if summary['failed_count'] > 0:
            print(f"  ❌ Exclude {summary['failed_count']} failed stocks")

        print()
        print(f"Full report saved to: {report_path}")
        print("="*60)

        # Exit with appropriate code
        if summary['ready_count'] == 0:
            logger.error("No stocks ready for ML training")
            sys.exit(1)
        else:
            logger.info("Validation successful")
            sys.exit(0)

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
