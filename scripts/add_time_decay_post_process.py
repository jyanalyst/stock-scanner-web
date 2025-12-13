"""
Post-Processing Script: Add Time-Decay Features
Adds 4 time-decay features to existing Phase 1 data without re-running collection
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.local_file_loader import get_local_loader


def calculate_days_since_high(hist_df):
    """Calculate days since 252-day high"""
    if len(hist_df) < 20:
        return 0
    
    rolling_high = hist_df['High'].rolling(window=min(252, len(hist_df)), min_periods=20).max()
    
    # Find most recent high
    for i in range(len(hist_df) - 1, -1, -1):
        if hist_df['High'].iloc[i] >= rolling_high.iloc[i]:
            return len(hist_df) - 1 - i
    
    return len(hist_df) - 1


def calculate_days_since_low(hist_df):
    """Calculate days since 252-day low (CRITICAL for mean reversion)"""
    if len(hist_df) < 20:
        return 0
    
    rolling_low = hist_df['Low'].rolling(window=min(252, len(hist_df)), min_periods=20).min()
    
    # Find most recent low
    for i in range(len(hist_df) - 1, -1, -1):
        if hist_df['Low'].iloc[i] <= rolling_low.iloc[i]:
            return len(hist_df) - 1 - i
    
    return len(hist_df) - 1


def calculate_days_since_triple_aligned(hist_df):
    """Calculate duration of three-indicator alignment"""
    if 'Is_Triple_Aligned' not in hist_df.columns:
        return 0
    
    # Count consecutive days of alignment from the end
    counter = 0
    for i in range(len(hist_df) - 1, -1, -1):
        if hist_df['Is_Triple_Aligned'].iloc[i]:
            counter += 1
        else:
            break
    
    return counter


def calculate_days_since_flow_regime_change(hist_df):
    """Calculate days since last flow regime change"""
    if 'Flow_10D' not in hist_df.columns or len(hist_df) < 2:
        return 0
    
    # Detect flow regime changes (sign flips)
    flow_sign = np.sign(hist_df['Flow_10D'])
    
    # Find most recent regime change
    for i in range(len(hist_df) - 1, 0, -1):
        if flow_sign.iloc[i] != flow_sign.iloc[i-1]:
            return len(hist_df) - 1 - i
    
    return len(hist_df) - 1


def add_time_decay_features_to_sample(sample, hist_data_cache, loader):
    """
    Add time-decay features to a single sample
    
    Args:
        sample: Dictionary with sample data
        hist_data_cache: Cache of historical data by ticker
        loader: Data loader instance
        
    Returns:
        Enhanced sample with time-decay features
    """
    ticker = sample['Ticker']
    entry_date = pd.to_datetime(sample['entry_date'])
    
    # Get cached historical data
    if ticker not in hist_data_cache:
        try:
            hist_data_cache[ticker] = loader.load_historical_data(ticker)
        except Exception as e:
            print(f"⚠️  Failed to load {ticker}: {e}")
            hist_data_cache[ticker] = None
    
    hist_df = hist_data_cache[ticker]
    
    if hist_df is None or hist_df.empty:
        # Use default values if no data
        sample['Days_Since_High'] = 0
        sample['Days_Since_Low'] = 0
        sample['Days_Since_Triple_Aligned'] = 0
        sample['Days_Since_Flow_Regime_Change'] = 0
        return sample
    
    # Make a copy and ensure Date is a column
    hist_df = hist_df.copy()
    if 'Date' not in hist_df.columns:
        hist_df = hist_df.reset_index()
    hist_df['Date'] = pd.to_datetime(hist_df['Date'])
    
    # Filter to entry_date (NO LOOKAHEAD!)
    hist_df = hist_df[hist_df['Date'] <= entry_date].copy()
    
    if len(hist_df) == 0:
        # No historical data available
        sample['Days_Since_High'] = 0
        sample['Days_Since_Low'] = 0
        sample['Days_Since_Triple_Aligned'] = 0
        sample['Days_Since_Flow_Regime_Change'] = 0
        return sample
    
    # Calculate time-decay features
    try:
        sample['Days_Since_High'] = calculate_days_since_high(hist_df)
    except:
        sample['Days_Since_High'] = 0
    
    try:
        sample['Days_Since_Low'] = calculate_days_since_low(hist_df)
    except:
        sample['Days_Since_Low'] = 0
    
    try:
        sample['Days_Since_Triple_Aligned'] = calculate_days_since_triple_aligned(hist_df)
    except:
        sample['Days_Since_Triple_Aligned'] = 0
    
    try:
        sample['Days_Since_Flow_Regime_Change'] = calculate_days_since_flow_regime_change(hist_df)
    except:
        sample['Days_Since_Flow_Regime_Change'] = 0
    
    return sample


def main():
    """Add time-decay features to existing Phase 1 data"""
    
    print("=" * 80)
    print("TIME-DECAY POST-PROCESSING")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load existing data
    input_path = 'data/ml_training/raw/training_data_complete.parquet'
    output_path = 'data/ml_training/raw/training_data_enhanced.parquet'
    
    print(f"📥 Loading data from {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"✅ Loaded {len(df):,} samples")
    print()
    
    # Initialize data loader
    loader = get_local_loader()
    hist_data_cache = {}
    
    print("🔧 Adding time-decay features...")
    print(f"   Processing {len(df):,} samples across {df['Ticker'].nunique()} stocks")
    print()
    
    # Convert to list of dicts for processing
    samples = df.to_dict('records')
    
    # Process each sample with progress bar
    enhanced_samples = []
    for sample in tqdm(samples, desc="Processing samples", unit="samples"):
        enhanced_sample = add_time_decay_features_to_sample(sample, hist_data_cache, loader)
        enhanced_samples.append(enhanced_sample)
    
    # Convert back to DataFrame
    df_enhanced = pd.DataFrame(enhanced_samples)
    
    print()
    print("=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    
    # Verify new features
    time_decay_cols = [
        'Days_Since_High',
        'Days_Since_Low',
        'Days_Since_Triple_Aligned',
        'Days_Since_Flow_Regime_Change'
    ]
    
    print("📊 Time-Decay Features Added:")
    for col in time_decay_cols:
        if col in df_enhanced.columns:
            min_val = df_enhanced[col].min()
            max_val = df_enhanced[col].max()
            mean_val = df_enhanced[col].mean()
            nan_count = df_enhanced[col].isna().sum()
            
            print(f"   ✅ {col}")
            print(f"      Range: {min_val:.0f} to {max_val:.0f}, Mean: {mean_val:.1f}, NaN: {nan_count}")
        else:
            print(f"   ❌ {col} - FAILED TO ADD")
    
    print()
    print(f"📈 Feature Count:")
    print(f"   Before: {len(df.columns)}")
    print(f"   After: {len(df_enhanced.columns)}")
    print(f"   Added: {len(df_enhanced.columns) - len(df.columns)}")
    
    print()
    
    # Save enhanced data
    print(f"💾 Saving enhanced data to {output_path}...")
    df_enhanced.to_parquet(output_path)
    print(f"✅ Saved successfully!")
    
    print()
    print("=" * 80)
    print("✅ POST-PROCESSING COMPLETE!")
    print("=" * 80)
    print()
    print(f"📊 Enhanced Data:")
    print(f"   - Samples: {len(df_enhanced):,}")
    print(f"   - Features: {len(df_enhanced.columns)}")
    print(f"   - CS Ranks: 7")
    print(f"   - Time-Decay: 4")
    print(f"   - File: {output_path}")
    print()
    print("🎯 Next Steps:")
    print("   1. python scripts/add_categorical_encoding.py")
    print("   2. python scripts/test_factor_analysis.py")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
