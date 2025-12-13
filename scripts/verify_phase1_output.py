"""
Phase 1 Data Verification Script
Comprehensive validation of collected training data
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def verify_phase1_data():
    """Comprehensive verification of Phase 1 output"""
    
    print("=" * 80)
    print("PHASE 1 DATA VERIFICATION")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load data
    data_path = 'data/ml_training/raw/training_data_complete.parquet'
    
    try:
        print(f"📥 Loading data from {data_path}...")
        df = pd.read_parquet(data_path)
        print(f"✅ Loaded successfully!")
        print()
    except Exception as e:
        print(f"❌ ERROR: Could not load data: {e}")
        return False
    
    # ===== SECTION 1: BASIC STATS =====
    print("=" * 80)
    print("1. BASIC STATISTICS")
    print("=" * 80)
    
    print(f"📊 Total Samples: {len(df):,}")
    print(f"📈 Unique Stocks: {df['Ticker'].nunique()}")
    print(f"📅 Date Range: {df['entry_date'].min()} to {df['entry_date'].max()}")
    print(f"🔢 Total Features: {len(df.columns)}")
    print()
    
    # ===== SECTION 2: CROSS-SECTIONAL RANKS =====
    print("=" * 80)
    print("2. CROSS-SECTIONAL RANKS VALIDATION")
    print("=" * 80)
    
    cs_rank_cols = [col for col in df.columns if col.endswith('_CS_Rank')]
    print(f"📊 Found {len(cs_rank_cols)} CS Rank features:")
    print()
    
    expected_cs_ranks = [
        'MPI_Percentile_CS_Rank',
        'IBS_Percentile_CS_Rank',
        'VPI_Percentile_CS_Rank',
        'IBS_Accel_CS_Rank',
        'RVol_Accel_CS_Rank',
        'RRange_Accel_CS_Rank',
        'VPI_Accel_CS_Rank',
        'Flow_Price_Gap_CS_Rank'
    ]
    
    cs_status = []
    for col in expected_cs_ranks:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            mean_val = df[col].mean()
            nan_count = df[col].isna().sum()
            
            # Validate range
            if min_val >= 0 and max_val <= 100:
                status = "✅"
            else:
                status = "❌"
            
            print(f"   {status} {col}")
            print(f"      Range: {min_val:.1f} to {max_val:.1f}, Mean: {mean_val:.1f}, NaN: {nan_count}")
            cs_status.append(True)
        else:
            print(f"   ⚠️  {col} - MISSING (may be expected)")
            cs_status.append(False)
    
    print()
    print(f"📈 CS Rank Summary: {sum(cs_status)}/{len(expected_cs_ranks)} features present")
    print()
    
    # ===== SECTION 3: TIME-DECAY FEATURES =====
    print("=" * 80)
    print("3. TIME-DECAY FEATURES VALIDATION")
    print("=" * 80)
    
    time_decay_cols = [
        'Days_Since_High',
        'Days_Since_Low',
        'Days_Since_Triple_Aligned',
        'Days_Since_Flow_Regime_Change'
    ]
    
    td_status = []
    for col in time_decay_cols:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            mean_val = df[col].mean()
            nan_count = df[col].isna().sum()
            
            # Validate non-negative
            if min_val >= 0:
                status = "✅"
            else:
                status = "❌"
            
            print(f"   {status} {col}")
            print(f"      Range: {min_val:.0f} to {max_val:.0f} days, Mean: {mean_val:.1f}, NaN: {nan_count}")
            td_status.append(True)
        else:
            print(f"   ❌ {col} - MISSING")
            td_status.append(False)
    
    print()
    print(f"⏱️  Time-Decay Summary: {sum(td_status)}/{len(time_decay_cols)} features present")
    print()
    
    # ===== SECTION 4: FORWARD RETURN LABELS =====
    print("=" * 80)
    print("4. FORWARD RETURN LABELS VALIDATION")
    print("=" * 80)
    
    return_cols = [col for col in df.columns if col.startswith('return_') and col.endswith('d')]
    win_cols = [col for col in df.columns if col.startswith('win_') and col.endswith('d')]
    
    print(f"📊 Return columns: {', '.join(return_cols)}")
    print(f"🎯 Win columns: {', '.join(win_cols)}")
    print()
    
    for col in return_cols:
        valid_count = df[col].notna().sum()
        valid_pct = valid_count / len(df) * 100
        mean_return = df[col].mean()
        
        print(f"   {col}:")
        print(f"      Valid: {valid_count:,} ({valid_pct:.1f}%)")
        print(f"      Mean Return: {mean_return*100:.2f}%")
    
    print()
    
    # ===== SECTION 5: CORE ML FEATURES =====
    print("=" * 80)
    print("5. CORE ML FEATURES VALIDATION")
    print("=" * 80)
    
    core_features = [
        'MPI_Percentile', 'IBS_Percentile', 'VPI_Percentile',
        'IBS_Accel', 'RVol_Accel', 'RRange_Accel', 'VPI_Accel',
        'Flow_Percentile', 'Volume_Conviction', 'Relative_Volume'
    ]
    
    missing_core = []
    for col in core_features:
        if col in df.columns:
            print(f"   ✅ {col}")
        else:
            print(f"   ❌ {col} - MISSING")
            missing_core.append(col)
    
    print()
    if missing_core:
        print(f"⚠️  Missing {len(missing_core)} core features: {missing_core}")
    else:
        print(f"✅ All core features present!")
    print()
    
    # ===== SECTION 6: DATA QUALITY =====
    print("=" * 80)
    print("6. DATA QUALITY CHECKS")
    print("=" * 80)
    
    # Check for excessive missing values
    missing_pct = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    high_missing = missing_pct[missing_pct > 10]
    
    if len(high_missing) > 0:
        print(f"⚠️  Features with >10% missing values:")
        for col, pct in high_missing.head(10).items():
            print(f"      {col}: {pct:.1f}%")
    else:
        print(f"✅ No features with excessive missing values")
    
    print()
    
    # Check sample distribution
    samples_per_stock = df.groupby('Ticker').size()
    print(f"📊 Samples per stock:")
    print(f"      Min: {samples_per_stock.min()}")
    print(f"      Max: {samples_per_stock.max()}")
    print(f"      Mean: {samples_per_stock.mean():.1f}")
    print(f"      Median: {samples_per_stock.median():.1f}")
    
    print()
    
    # ===== SECTION 7: FEATURE BREAKDOWN =====
    print("=" * 80)
    print("7. FEATURE CATEGORY BREAKDOWN")
    print("=" * 80)
    
    all_cols = set(df.columns)
    
    # Categorize features
    cs_ranks = [col for col in all_cols if col.endswith('_CS_Rank')]
    time_decay = [col for col in all_cols if col.startswith('Days_Since_')]
    returns = [col for col in all_cols if col.startswith('return_') or col.startswith('win_')]
    meta = ['Ticker', 'entry_date', 'entry_price', 'max_drawdown', 'Date', 'Name', 'Close']
    
    ml_features = all_cols - set(cs_ranks) - set(time_decay) - set(returns) - set(meta)
    
    print(f"📊 Feature Categories:")
    print(f"   - Cross-Sectional Ranks: {len(cs_ranks)}")
    print(f"   - Time-Decay Features: {len(time_decay)}")
    print(f"   - Forward Return Labels: {len(returns)}")
    print(f"   - ML Features (base): {len(ml_features)}")
    print(f"   - Metadata: {len(meta)}")
    print(f"   - TOTAL: {len(df.columns)}")
    
    print()
    
    # Calculate ML feature count (for ml_config.yaml)
    ml_feature_count = len(cs_ranks) + len(time_decay) + len(ml_features)
    print(f"🎯 Total ML Features (for training): {ml_feature_count}")
    print(f"   (CS Ranks: {len(cs_ranks)} + Time-Decay: {len(time_decay)} + Base: {len(ml_features)})")
    
    print()
    
    # ===== FINAL VERDICT =====
    print("=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    print()
    
    issues = []
    
    # Check 1: Sample count
    if len(df) < 10000:
        issues.append(f"Low sample count: {len(df):,} (expected >10,000)")
    
    # Check 2: CS ranks
    if len(cs_ranks) < 5:
        issues.append(f"Too few CS ranks: {len(cs_ranks)} (expected 7-8)")
    
    # Check 3: Time-decay
    if len(time_decay) < 2:
        issues.append(f"Too few time-decay features: {len(time_decay)} (expected 4)")
    
    # Check 4: Return labels
    if 'return_2d' not in df.columns:
        issues.append("Missing return_2d label")
    
    # Check 5: Core features
    if len(missing_core) > 0:
        issues.append(f"Missing core features: {missing_core}")
    
    if len(issues) == 0:
        print("🎉 ALL CHECKS PASSED!")
        print()
        print("✅ Data quality: EXCELLENT")
        print(f"✅ Sample count: {len(df):,} (EXCELLENT)")
        print(f"✅ CS Ranks: {len(cs_ranks)} features")
        print(f"✅ Time-Decay: {len(time_decay)} features")
        print(f"✅ ML Features: {ml_feature_count} total")
        print()
        print("🚀 READY FOR PHASE 2: Factor Analysis")
        print()
        print("Next steps:")
        print("   1. python scripts/add_categorical_encoding.py")
        print("   2. python scripts/test_factor_analysis.py")
        print()
        return True
    else:
        print("⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")
        print()
        print("Please investigate before proceeding to Phase 2")
        return False


def main():
    """Run verification"""
    success = verify_phase1_data()
    
    print("=" * 80)
    
    if success:
        print("✅ VERIFICATION COMPLETE - READY FOR PHASE 2!")
    else:
        print("❌ VERIFICATION FAILED - PLEASE FIX ISSUES")
    
    print("=" * 80)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
