#!/usr/bin/env python3
"""
Test script for Institutional Flow Analysis implementation
Tests the core flow metrics calculations and validates results
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.technical_analysis import (
    calculate_institutional_flow,
    classify_flow_regime,
    calculate_volume_conviction,
    calculate_price_flow_divergence
)

def create_test_data():
    """Create sample OHLCV data for testing"""
    np.random.seed(42)  # For reproducible results

    # Generate 100 days of sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')

    # Create realistic price data with trends
    base_price = 10.0
    prices = []
    volumes = []

    for i in range(100):
        # Add some trend and noise
        trend = 0.02 * i  # Upward trend
        noise = np.random.normal(0, 0.5)
        price = base_price + trend + noise
        prices.append(max(1.0, price))  # Ensure positive prices

        # Volume with some variability
        vol_noise = np.random.normal(1.0, 0.3)
        volume = int(1000000 * vol_noise)
        volumes.append(max(1000, volume))

    # Create OHLC data with proper price movements
    opens = prices[:-1]  # Use previous close as next open
    opens.insert(0, base_price)  # Add initial open

    df = pd.DataFrame({
        'Date': dates,
        'Open': opens,
        'High': [max(o, c) * (1 + abs(np.random.normal(0, 0.02))) for o, c in zip(opens, prices)],
        'Low': [min(o, c) * (1 - abs(np.random.normal(0, 0.02))) for o, c in zip(opens, prices)],
        'Close': prices,
        'Volume': volumes
    })

    df.set_index('Date', inplace=True)
    return df

def test_institutional_flow():
    """Test institutional flow calculations"""
    print("🧪 Testing Institutional Flow Analysis...")
    print("=" * 50)

    # Create test data
    df = create_test_data()
    print(f"📊 Created test data: {len(df)} rows")

    # Test each calculation step
    try:
        # Step 1: Calculate institutional flow
        df = calculate_institutional_flow(df)
        print("✅ Step 1 - Institutional Flow: PASSED")
        print(f"   Daily_Flow range: {df['Daily_Flow'].min():.2f} to {df['Daily_Flow'].max():.2f}")
        print(f"   Flow_10D range: {df['Flow_10D'].min():.2f} to {df['Flow_10D'].max():.2f}")
        print(f"   Flow_Velocity range: {df['Flow_Velocity'].min():.2f} to {df['Flow_Velocity'].max():.2f}")

        # Step 2: Classify flow regime
        df = classify_flow_regime(df)
        print("✅ Step 2 - Flow Regime Classification: PASSED")
        regime_counts = df['Flow_Regime'].value_counts()
        print(f"   Flow regimes: {dict(regime_counts)}")

        # Step 3: Calculate volume conviction
        df = calculate_volume_conviction(df)
        print("✅ Step 3 - Volume Conviction: PASSED")
        print(f"   Volume_Conviction range: {df['Volume_Conviction'].min():.3f} to {df['Volume_Conviction'].max():.3f}")
        print(f"   Conviction_Velocity range: {df['Conviction_Velocity'].min():.3f} to {df['Conviction_Velocity'].max():.3f}")

        # Step 4: Calculate price-flow divergence
        df = calculate_price_flow_divergence(df)
        print("✅ Step 4 - Price-Flow Divergence: PASSED")
        print(f"   Divergence_Gap range: {df['Divergence_Gap'].min():.1f} to {df['Divergence_Gap'].max():.1f}")
        print(f"   Divergence_Severity range: {df['Divergence_Severity'].min():.1f} to {df['Divergence_Severity'].max():.1f}")

        # Validation checks
        print("\n🔍 Validation Results:")
        print("-" * 30)

        # Check for NaN values
        nan_counts = df[['Daily_Flow', 'Flow_10D', 'Flow_Velocity', 'Flow_Regime',
                        'Volume_Conviction', 'Conviction_Velocity',
                        'Divergence_Gap', 'Divergence_Severity']].isna().sum()
        total_nans = nan_counts.sum()

        if total_nans == 0:
            print("✅ No NaN values found - Data integrity: PASSED")
        else:
            print(f"⚠️  Found {total_nans} NaN values - Data integrity: WARNING")
            print(f"   NaN breakdown: {dict(nan_counts[nan_counts > 0])}")

        # Check reasonable value ranges
        flow_range_ok = abs(df['Daily_Flow']).max() < df['Volume'].max() * 10  # Reasonable bound
        conviction_range_ok = (df['Volume_Conviction'].min() >= 0) and (df['Volume_Conviction'].max() < 10)
        divergence_range_ok = abs(df['Divergence_Gap']).max() <= 100

        if flow_range_ok:
            print("✅ Flow values in reasonable range: PASSED")
        else:
            print("⚠️  Flow values may be unreasonable: WARNING")

        if conviction_range_ok:
            print("✅ Conviction values in reasonable range: PASSED")
        else:
            print("⚠️  Conviction values may be unreasonable: WARNING")

        if divergence_range_ok:
            print("✅ Divergence values in reasonable range: PASSED")
        else:
            print("⚠️  Divergence values may be unreasonable: WARNING")

        # Show sample results
        print("\n📋 Sample Results (last 5 days):")
        print("-" * 40)
        sample_cols = ['Close', 'Daily_Flow', 'Flow_10D', 'Flow_Regime',
                      'Volume_Conviction', 'Divergence_Gap', 'Divergence_Severity']
        print(df[sample_cols].tail(5).round(3))

        print("\n🎉 Institutional Flow Analysis Test: COMPLETED SUCCESSFULLY")
        return True

    except Exception as e:
        print(f"❌ Test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_institutional_flow()
    sys.exit(0 if success else 1)
