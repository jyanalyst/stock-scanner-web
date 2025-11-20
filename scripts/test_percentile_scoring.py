#!/usr/bin/env python3
"""
Test script for the new percentile-based scoring system
Verifies that Phase 2 integration works correctly
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.technical_analysis import (
    calculate_percentile_ranks,
    calculate_acceleration_score_v3
)

def create_test_data_with_metrics():
    """Create synthetic data with all required metrics for testing"""
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=100, freq='D')

    # Create realistic OHLCV data
    base_price = 10.0
    price_changes = np.random.normal(0, 0.02, 100)
    prices = base_price * np.exp(np.cumsum(price_changes))

    df = pd.DataFrame({
        'Open': prices * np.random.uniform(0.995, 1.005, 100),
        'High': prices * np.random.uniform(1.005, 1.03, 100),
        'Low': prices * np.random.uniform(0.97, 0.995, 100),
        'Close': prices,
        'Volume': np.random.uniform(500000, 2000000, 100)
    }, index=dates)

    # Add the required acceleration metrics (simulated)
    df['IBS_Accel'] = np.random.normal(0, 0.05, 100)
    df['RVol_Accel'] = np.random.normal(0, 0.02, 100)
    df['RRange_Accel'] = np.random.normal(0, 0.1, 100)
    df['Flow_Velocity'] = np.random.normal(0, 0.5, 100)
    df['Volume_Conviction'] = np.random.normal(1.0, 0.2, 100)

    return df

def test_percentile_calculation():
    """Test percentile rank calculation"""
    print("Testing Percentile Rank Calculation...")

    df = create_test_data_with_metrics()

    # Test percentile calculation
    df_pct = calculate_percentile_ranks(df)

    # Verify columns exist
    required_columns = [
        'IBS_Accel_Percentile', 'IBS_Bullish_Pct', 'IBS_Bearish_Pct',
        'RVol_Accel_Percentile', 'RRange_Accel_Percentile',
        'Flow_Velocity_Percentile', 'Flow_Bullish_Pct', 'Flow_Bearish_Pct',
        'Volume_Conviction_Percentile'
    ]

    for col in required_columns:
        assert col in df_pct.columns, f"Missing column: {col}"

    # Verify percentiles are in valid range [0, 1]
    percentile_cols = [col for col in df_pct.columns if 'Percentile' in col or 'Pct' in col]
    for col in percentile_cols:
        values = df_pct[col].dropna()
        assert (values >= 0).all() and (values <= 1).all(), f"Column {col} has invalid percentile values"

    # Verify direction-specific logic
    bullish_pct = df_pct['IBS_Bullish_Pct']
    bearish_pct = df_pct['IBS_Bearish_Pct']

    # When IBS_Accel > 0, bullish_pct should be > 0 and bearish_pct should be 0
    positive_ibs = df_pct['IBS_Accel'] > 0
    assert (df_pct.loc[positive_ibs, 'IBS_Bullish_Pct'] > 0).all(), "Bullish percentiles should be > 0 for positive IBS"
    assert (df_pct.loc[positive_ibs, 'IBS_Bearish_Pct'] == 0).all(), "Bearish percentiles should be 0 for positive IBS"

    # When IBS_Accel < 0, bearish_pct should be > 0 and bullish_pct should be 0
    negative_ibs = df_pct['IBS_Accel'] < 0
    assert (df_pct.loc[negative_ibs, 'IBS_Bearish_Pct'] > 0).all(), "Bearish percentiles should be > 0 for negative IBS"
    assert (df_pct.loc[negative_ibs, 'IBS_Bullish_Pct'] == 0).all(), "Bullish percentiles should be 0 for negative IBS"

    print("✅ Percentile calculation test passed")
    return df_pct

def test_scoring_system():
    """Test the new percentile-based scoring system"""
    print("Testing Percentile-Based Scoring System...")

    df = create_test_data_with_metrics()
    df_pct = calculate_percentile_ranks(df)

    # Test scoring for a sample row
    sample_row = df_pct.iloc[50]  # Middle row with enough history

    # Test bullish scoring
    total_score, component_scores, quality_label = calculate_acceleration_score_v3(
        direction='bullish',
        ibs_pct=sample_row['IBS_Bullish_Pct'],
        rvol_pct=sample_row['RVol_Accel_Percentile'],
        rrange_pct=sample_row['RRange_Accel_Percentile'],
        flow_pct=sample_row['Flow_Bullish_Pct'],
        conviction_pct=sample_row['Volume_Conviction_Percentile']
    )

    # Verify score is in valid range
    assert 0 <= total_score <= 100, f"Total score {total_score} is out of range [0, 100]"

    # Verify component scores exist and sum correctly
    expected_components = ['ibs', 'rvol', 'rrange', 'flow', 'conviction']
    for component in expected_components:
        assert component in component_scores, f"Missing component score: {component}"
        assert 0 <= component_scores[component] <= 100, f"Component {component} score {component_scores[component]} is out of range"

    # Verify components sum to total
    calculated_total = sum(component_scores.values())
    assert abs(calculated_total - total_score) < 1, f"Component scores don't sum correctly: {calculated_total} vs {total_score}"

    # Verify quality label is valid
    valid_labels = ['🔥 TOP 20%', '🟢 TOP 40%', '🟡 AVERAGE', '🟠 BELOW AVERAGE']
    assert quality_label in valid_labels, f"Invalid quality label: {quality_label}"

    # Test bearish scoring
    bearish_score, bearish_components, bearish_label = calculate_acceleration_score_v3(
        direction='bearish',
        ibs_pct=sample_row['IBS_Bearish_Pct'],
        rvol_pct=sample_row['RVol_Accel_Percentile'],
        rrange_pct=sample_row['RRange_Accel_Percentile'],
        flow_pct=sample_row['Flow_Bearish_Pct'],
        conviction_pct=sample_row['Volume_Conviction_Percentile']
    )

    assert 0 <= bearish_score <= 100, f"Bearish total score {bearish_score} is out of range [0, 100]"
    assert bearish_label in valid_labels, f"Invalid bearish quality label: {bearish_label}"

    print("✅ Scoring system test passed")
    return total_score, component_scores, quality_label

def test_weight_integration():
    """Test integration with data-driven weights from Phase 0"""
    print("Testing Weight Integration...")

    df = create_test_data_with_metrics()
    df_pct = calculate_percentile_ranks(df)
    sample_row = df_pct.iloc[50]

    # Test with custom weights (simulating Phase 0 results)
    custom_weights = {
        'IBS_Accel': 1.5,      # Low weight
        'RVol_Accel': 86.5,    # High weight (dominant)
        'RRange_Accel': 11.3,  # Medium weight
        'Flow_Velocity': 0.0,  # Zero weight (as found in Phase 0)
        'Volume_Conviction': 0.7  # Low weight
    }

    score_with_weights, components, label = calculate_acceleration_score_v3(
        direction='bullish',
        ibs_pct=sample_row['IBS_Bullish_Pct'],
        rvol_pct=sample_row['RVol_Accel_Percentile'],
        rrange_pct=sample_row['RRange_Accel_Percentile'],
        flow_pct=sample_row['Flow_Bullish_Pct'],
        conviction_pct=sample_row['Volume_Conviction_Percentile'],
        weights=custom_weights
    )

    # Verify RVol gets the most weight in scoring
    assert components['rvol'] >= components['rrange'], "RVol should have higher score than RRange"
    assert components['flow'] == 0, "Flow should have zero score with 0.0 weight"
    assert components['ibs'] <= components['rvol'], "IBS should have lower score than RVol"

    print("✅ Weight integration test passed")
    return score_with_weights, components

def test_edge_cases():
    """Test edge cases and boundary conditions"""
    print("Testing Edge Cases...")

    # Test with all percentiles at boundaries
    edge_cases = [
        (0.0, 0.0, 0.0, 0.0, 0.0),  # All minimum
        (1.0, 1.0, 1.0, 1.0, 1.0),  # All maximum
        (0.5, 0.5, 0.5, 0.5, 0.5),  # All neutral
    ]

    for ibs, rvol, rrange, flow, conviction in edge_cases:
        score, components, label = calculate_acceleration_score_v3(
            direction='bullish',
            ibs_pct=ibs, rvol_pct=rvol, rrange_pct=rrange,
            flow_pct=flow, conviction_pct=conviction
        )

        assert 0 <= score <= 100, f"Edge case score {score} out of range"
        assert label in ['🔥 TOP 20%', '🟢 TOP 40%', '🟡 AVERAGE', '🟠 BELOW AVERAGE'], f"Invalid label: {label}"

    print("✅ Edge cases test passed")

def main():
    """Run all tests"""
    print("🧪 Testing Phase 2: Percentile-Based Scoring Integration")
    print("=" * 60)

    try:
        # Run individual tests
        df_pct = test_percentile_calculation()
        score, components, label = test_scoring_system()
        weighted_score, weighted_components = test_weight_integration()
        test_edge_cases()

        print("\n" + "=" * 60)
        print("🎉 All Phase 2 tests passed!")
        print("📊 Sample results:")
        print(f"   - Sample score: {score} points")
        print(f"   - Quality label: {label}")
        print(f"   - Component breakdown: {components}")
        print(f"   - Weighted score: {weighted_score} points")
        print(f"   - RVol dominance: {weighted_components['rvol']} points (86.5% of max)")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
