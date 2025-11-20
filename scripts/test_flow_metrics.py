#!/usr/bin/env python3
"""
Test script for the new institutional flow metrics
Verifies that the three new functions work correctly
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.technical_analysis import (
    calculate_institutional_flow_system,
    calculate_volume_conviction_metrics,
    calculate_flow_divergence_metrics
)

def create_test_data():
    """Create synthetic OHLCV data for testing"""
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=50, freq='D')

    # Create realistic price series
    base_price = 10.0
    price_changes = np.random.normal(0, 0.02, 50)
    prices = base_price * np.exp(np.cumsum(price_changes))

    # Create OHLCV data
    df = pd.DataFrame({
        'Open': prices * np.random.uniform(0.995, 1.005, 50),
        'High': prices * np.random.uniform(1.005, 1.03, 50),
        'Low': prices * np.random.uniform(0.97, 0.995, 50),
        'Close': prices,
        'Volume': np.random.uniform(500000, 2000000, 50)  # 500K to 2M shares
    }, index=dates)

    return df

def test_flow_system():
    """Test institutional flow system"""
    print("Testing Institutional Flow System...")

    df = create_test_data()

    # Test flow calculation
    df_flow = calculate_institutional_flow_system(df)

    # Verify columns exist
    required_columns = ['Daily_Flow', 'Daily_Flow_Normalized', 'Flow_10D', 'Flow_Velocity', 'Flow_Regime']
    for col in required_columns:
        assert col in df_flow.columns, f"Missing column: {col}"

    # Verify calculations
    assert not df_flow['Daily_Flow'].isna().all(), "Daily_Flow should not be all NaN"
    assert not df_flow['Flow_10D'].isna().all(), "Flow_10D should not be all NaN"
    assert df_flow['Flow_Regime'].notna().any(), "Should have some flow regime classifications"

    print("✅ Institutional Flow System test passed")
    return df_flow

def test_conviction_metrics():
    """Test volume conviction metrics"""
    print("Testing Volume Conviction Metrics...")

    df = create_test_data()

    # Test conviction calculation
    df_conviction = calculate_volume_conviction_metrics(df)

    # Verify columns exist
    required_columns = ['Volume_Up_Days', 'Volume_Down_Days', 'Volume_Conviction', 'Conviction_Velocity']
    for col in required_columns:
        assert col in df_conviction.columns, f"Missing column: {col}"

    # Verify calculations
    assert not df_conviction['Volume_Conviction'].isna().all(), "Volume_Conviction should not be all NaN"
    assert (df_conviction['Volume_Conviction'] > 0).any(), "Should have positive conviction ratios"

    print("✅ Volume Conviction Metrics test passed")
    return df_conviction

def test_divergence_metrics():
    """Test flow divergence metrics"""
    print("Testing Flow Divergence Metrics...")

    df = create_test_data()

    # First calculate flow (required for divergence)
    df = calculate_institutional_flow_system(df)

    # Test divergence calculation
    df_divergence = calculate_flow_divergence_metrics(df)

    # Verify columns exist
    required_columns = ['Price_Slope_10D', 'Flow_Slope_10D', 'Price_Percentile',
                       'Flow_Percentile', 'Divergence_Gap', 'Divergence_Severity']
    for col in required_columns:
        assert col in df_divergence.columns, f"Missing column: {col}"

    # Verify calculations
    assert not df_divergence['Divergence_Gap'].isna().all(), "Divergence_Gap should not be all NaN"
    assert (df_divergence['Divergence_Severity'] >= 0).all(), "Divergence_Severity should be non-negative"

    print("✅ Flow Divergence Metrics test passed")
    return df_divergence

def test_integration():
    """Test all three functions together"""
    print("Testing Integration of All Three Functions...")

    df = create_test_data()

    # Apply all three functions in sequence
    df = calculate_institutional_flow_system(df)
    df = calculate_volume_conviction_metrics(df)
    df = calculate_flow_divergence_metrics(df)

    # Verify all columns exist
    all_required_columns = [
        # Flow columns
        'Daily_Flow', 'Daily_Flow_Normalized', 'Flow_10D', 'Flow_Velocity', 'Flow_Regime',
        # Conviction columns
        'Volume_Up_Days', 'Volume_Down_Days', 'Volume_Conviction', 'Conviction_Velocity',
        # Divergence columns
        'Price_Slope_10D', 'Flow_Slope_10D', 'Price_Percentile', 'Flow_Percentile',
        'Divergence_Gap', 'Divergence_Severity'
    ]

    for col in all_required_columns:
        assert col in df.columns, f"Missing column: {col}"

    # Verify no NaN values in key columns (should be filled)
    key_columns = ['Flow_10D', 'Volume_Conviction', 'Divergence_Gap']
    for col in key_columns:
        assert not df[col].isna().any(), f"Column {col} should not contain NaN values"

    print("✅ Integration test passed")
    return df

def main():
    """Run all tests"""
    print("🧪 Testing New Institutional Flow Metrics")
    print("=" * 50)

    try:
        # Run individual tests
        df_flow = test_flow_system()
        df_conviction = test_conviction_metrics()
        df_divergence = test_divergence_metrics()

        # Run integration test
        df_complete = test_integration()

        print("\n" + "=" * 50)
        print("🎉 All tests passed!")
        print("📊 Sample results from integration test:")
        print(f"   - Average Flow_10D: {df_complete['Flow_10D'].mean():.3f}")
        print(f"   - Average Volume_Conviction: {df_complete['Volume_Conviction'].mean():.3f}")
        print(f"   - Average Divergence_Severity: {df_complete['Divergence_Severity'].mean():.1f}")
        print(f"   - Flow Regimes: {df_complete['Flow_Regime'].value_counts().to_dict()}")
        print("=" * 50)

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
