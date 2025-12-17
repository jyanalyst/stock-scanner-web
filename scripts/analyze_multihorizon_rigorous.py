#!/usr/bin/env python3
"""
Phase 1: Statistically Rigorous Multi-Horizon IC Analysis
Determines optimal holding period with proper hypothesis testing
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import logging
from ml.factor_analyzer import MLFactorAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    print("=" * 80)
    print("PHASE 1: STATISTICALLY RIGOROUS MULTI-HORIZON IC ANALYSIS")
    print("=" * 80)

    # Load data
    print("\n1️⃣ Loading training data...")
    analyzer = MLFactorAnalyzer(target='return_2d')  # Default target, will be changed
    df = analyzer.load_data()
    print(f"✅ Loaded {len(df):,} samples")

    # Run rigorous multi-horizon analysis
    print("\n2️⃣ Running multi-horizon IC analysis with statistical tests...")
    print("   (This may take 15-20 minutes due to bootstrapping...)\n")

    summary, detailed_results, optimal_horizon = analyzer.analyze_multihorizon_ic_rigorous(
        horizons=[1, 2, 3, 4, 5]
    )

    # Save results
    print("\n3️⃣ Saving results...")
    analysis_dir = Path('data/ml_training/analysis')
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Save summary
    summary.to_csv(analysis_dir / 'multihorizon_ic_summary_rigorous.csv', index=False)
    print(f"✅ Saved summary to: {analysis_dir / 'multihorizon_ic_summary_rigorous.csv'}")

    # Save detailed results for each horizon
    for horizon, results in detailed_results.items():
        ic_df = results['ic_results']
        ic_df.to_csv(analysis_dir / f'ic_detailed_{horizon}.csv', index=False)
        print(f"✅ Saved {horizon} details to: {analysis_dir / f'ic_detailed_{horizon}.csv'}")

    # Final decision
    print("\n" + "=" * 80)
    print("PHASE 1 RESULTS")
    print("=" * 80)

    if optimal_horizon is None:
        print("\n❌ STOP: No statistically significant horizons found")
        print("\nPossible actions:")
        print("1. Collect more data (increase sample size)")
        print("2. Engineer better features")
        print("3. Review signal generation logic")
        print("\nDo NOT proceed to Phase 2 without statistical significance.")
        return None
    else:
        print(f"\n✅ Optimal Holding Period: {optimal_horizon}")
        print(f"\nTop significant features:")
        optimal_features = detailed_results[optimal_horizon]['significant_features']
        for i, feat in enumerate(optimal_features[:5], 1):
            print(f"   {i}. {feat}")

        print(f"\n✅ Proceed to Phase 2 using {optimal_horizon} as holding period")
        return optimal_horizon


if __name__ == "__main__":
    optimal = main()
    if optimal:
        print(f"\n✅ Ready for Phase 2: Use holding period = {optimal}")
    else:
        print("\n❌ Cannot proceed to Phase 2")
        sys.exit(1)
