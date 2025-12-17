#!/usr/bin/env python3
"""
Phase 0: Data Quality Validation
Run this BEFORE any analysis to ensure data meets quality standards
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import logging
from ml.statistical_validator import DataQualityValidator, StatisticalValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    print("=" * 80)
    print("PHASE 0: DATA QUALITY VALIDATION")
    print("=" * 80)

    # Load training data
    print("\n1️⃣ Loading training data...")
    df = pd.read_parquet('data/ml_training/raw/training_data_complete.parquet')
    print(f"✅ Loaded {len(df):,} samples")
    print(f"   Date range: {df['entry_date'].min()} to {df['entry_date'].max()}")
    print(f"   Unique stocks: {df['Ticker'].nunique()}")

    # Initialize validator
    validator = DataQualityValidator()
    stat_validator = StatisticalValidator()

    # Run comprehensive validation
    print("\n2️⃣ Running data quality checks...")
    validation_results = validator.validate_dataset(df)

    # Display results
    print("\n" + "=" * 80)
    print("DATA QUALITY REPORT")
    print("=" * 80)

    # Missing values
    print("\n📊 Missing Values:")
    missing = validation_results['missing_values']
    total_missing = sum(missing.values())
    if total_missing > 0:
        for col, count in missing.items():
            if count > 0:
                pct = count / len(df) * 100
                print(f"   {col}: {count:,} ({pct:.2f}%)")
    else:
        print("   ✅ No missing values found")

    # Infinite values
    print("\n📊 Infinite Values:")
    infinite = validation_results['infinite_values']
    total_infinite = sum(infinite.values())
    if total_infinite > 0:
        for col, count in infinite.items():
            if count > 0:
                print(f"   ⚠️ {col}: {count:,}")
    else:
        print("   ✅ No infinite values found")

    # Duplicates
    duplicates = validation_results['duplicate_rows']
    print(f"\n📊 Duplicate Rows: {duplicates:,}")
    if duplicates > 0:
        print("   ⚠️ Duplicate rows found - may indicate data collection issues")

    # Outliers in returns
    return_cols = [col for col in df.columns if col.startswith('return_')]
    if return_cols:
        print("\n📊 Return Outliers:")
        outlier_issues = 0
        for col in return_cols:
            if col in df.columns:
                outlier_key = f'outliers_{col}'
                if outlier_key in validation_results:
                    outlier_info = validation_results[outlier_key]
                    status = "✅" if outlier_info['is_acceptable'] else "⚠️"
                    print(f"   {status} {col}: {outlier_info['n_outliers']:,} ({outlier_info['outlier_pct']:.2f}%)")
                    if not outlier_info['is_acceptable']:
                        outlier_issues += 1

        if outlier_issues > 0:
            print("   ⚠️ Excessive outliers detected - may indicate data quality issues")
    else:
        print("\n📊 Return Outliers: No return columns found")

    # Data gaps
    if 'data_gaps' in validation_results:
        gap_info = validation_results['data_gaps']
        print(f"\n📊 Data Completeness:")
        print(f"   Missing dates: {gap_info['n_missing_dates']:,}")
        print(f"   Completeness: {gap_info['data_completeness_pct']:.1f}%")
        if gap_info['n_missing_dates'] > 0:
            print("   ⚠️ Missing trading dates - may indicate gaps in data collection")
    else:
        print("\n📊 Data Completeness: No date column found")

    # Survivorship bias
    if 'survivorship' in validation_results:
        surv_info = validation_results['survivorship']
        print(f"\n📊 Survivorship Bias Check:")
        print(f"   Total stocks: {surv_info['total_stocks']}")
        print(f"   With recent data: {surv_info['stocks_with_recent_data']}")
        if surv_info['potentially_delisted']:
            print(f"   ⚠️ Potentially delisted: {len(surv_info['potentially_delisted'])}")
            print(f"   {surv_info['potentially_delisted'][:5]}")  # Show first 5
            print("   ⚠️ Survivorship bias risk - delisted stocks may be missing")
        else:
            print("   ✅ No survivorship bias detected")

    # Overall assessment
    print("\n" + "=" * 80)
    overall = validation_results['overall_quality']
    print(f"📊 OVERALL QUALITY SCORE: {overall['quality_score']}/100")

    if overall['is_acceptable']:
        print("✅ Data quality is acceptable - proceed to Phase 1")
    else:
        print("⚠️ Data quality issues found:")
        for issue in overall['issues_found']:
            print(f"   - {issue}")
        print("\n❌ Fix data quality issues before proceeding to analysis")
        return False

    # Calculate minimum sample size requirements
    print("\n" + "=" * 80)
    print("STATISTICAL REQUIREMENTS")
    print("=" * 80)

    min_samples = stat_validator.calculate_minimum_sample_size(target_ic=0.08)
    print(f"\n📊 Minimum Sample Size:")
    print(f"   Required: {min_samples:,} samples (for IC=0.08, power=0.80)")
    print(f"   Available: {len(df):,} samples")

    if len(df) >= min_samples:
        print(f"   ✅ Sufficient data (excess: {len(df) - min_samples:,} samples)")
    else:
        print(f"   ⚠️ Insufficient data (shortfall: {min_samples - len(df):,} samples)")
        print("   ⚠️ Consider collecting more data or reducing target IC")

    # Save validation report
    print("\n3️⃣ Saving validation report...")
    report_path = 'data/ml_training/analysis/data_quality_report.txt'
    with open(report_path, 'w') as f:
        f.write("DATA QUALITY VALIDATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Date: {pd.Timestamp.now()}\n")
        f.write(f"Dataset: {len(df):,} samples\n")
        f.write(f"Quality Score: {overall['quality_score']}/100\n")
        f.write(f"Status: {'PASS' if overall['is_acceptable'] else 'FAIL'}\n")
        f.write("\n" + str(validation_results))

    print(f"✅ Report saved to: {report_path}")

    return overall['is_acceptable']


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
