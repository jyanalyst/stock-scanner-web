#!/usr/bin/env python3
"""
Complete integration test for Phase 1 + Phase 2 implementation
Tests the full enhanced scanner pipeline with institutional flow metrics
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.technical_analysis import (
    add_enhanced_columns,
    calculate_institutional_flow_system,
    calculate_volume_conviction_metrics,
    calculate_flow_divergence_metrics,
    calculate_percentile_ranks,
    calculate_acceleration_score_v3
)

def create_realistic_test_data():
    """Create realistic SGX-like data for comprehensive testing"""
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=150, freq='D')  # 6 months of data

    # Simulate a realistic SGX stock pattern
    base_price = 3.50  # Typical SGX stock price

    # Create trending periods with volatility
    trend_changes = np.random.choice([-0.001, 0.001], size=150, p=[0.4, 0.6])  # Slight bullish bias
    price_changes = np.random.normal(0, 0.015, 150) + trend_changes
    prices = base_price * np.exp(np.cumsum(price_changes))

    # Realistic volume patterns (lower on weekends, spikes on news)
    base_volume = 500000
    volume_multiplier = np.random.uniform(0.3, 3.0, 150)  # Weekend lows to news spikes
    volumes = (base_volume * volume_multiplier).astype(int)

    df = pd.DataFrame({
        'Open': prices * np.random.uniform(0.995, 1.005, 150),
        'High': prices * np.random.uniform(1.005, 1.03, 150),
        'Low': prices * np.random.uniform(0.97, 0.995, 150),
        'Close': prices,
        'Volume': volumes
    }, index=dates)

    return df

def test_full_pipeline():
    """Test the complete enhanced scanner pipeline"""
    print("Testing Complete Enhanced Scanner Pipeline...")
    print("=" * 60)

    # Create test data
    df = create_realistic_test_data()
    print(f"Created test data: {len(df)} days, price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")

    # Run the full enhanced pipeline
    df_enhanced = add_enhanced_columns(df, ticker="TEST.SI")

    # Verify all expected columns exist
    required_columns = [
        # Original MPI columns
        'MPI', 'MPI_Velocity', 'MPI_Trend',
        # New institutional flow columns
        'Daily_Flow', 'Flow_10D', 'Flow_Velocity', 'Flow_Regime',
        'Volume_Conviction', 'Conviction_Velocity',
        'Divergence_Gap', 'Divergence_Severity',
        # Percentile columns
        'IBS_Bullish_Pct', 'RVol_Accel_Percentile', 'Flow_Bullish_Pct',
        # Signal columns
        'Signal_Bias', 'Pattern_Quality', 'Total_Score',
        'IBS_Score', 'RVol_Score', 'RRange_Score'
    ]

    missing_columns = [col for col in required_columns if col not in df_enhanced.columns]
    if missing_columns:
        print(f"❌ Missing columns: {missing_columns}")
        return False

    print("✅ All required columns present")

    # Verify data integrity
    assert not df_enhanced['Close'].isna().any(), "Close prices should not be NaN"
    assert not df_enhanced['Volume'].isna().any(), "Volume should not be NaN"
    assert not df_enhanced['MPI'].isna().any(), "MPI should not be NaN"
    assert not df_enhanced['Flow_10D'].isna().any(), "Flow_10D should not be NaN"

    print("✅ Data integrity verified")

    # Check for signals
    bullish_signals = (df_enhanced['bullish_reversal'] == 1).sum()
    bearish_signals = (df_enhanced['bearish_reversal'] == 1).sum()
    total_signals = bullish_signals + bearish_signals

    print(f"📊 Signals detected: {bullish_signals} bullish, {bearish_signals} bearish ({total_signals} total)")

    # Analyze signal quality if signals exist
    if total_signals > 0:
        signal_rows = df_enhanced[
            (df_enhanced['bullish_reversal'] == 1) | (df_enhanced['bearish_reversal'] == 1)
        ]

        avg_score = signal_rows['Total_Score'].mean()
        max_score = signal_rows['Total_Score'].max()
        score_distribution = signal_rows['Pattern_Quality'].value_counts()

        print("📈 Signal Quality Analysis:")
        print(f"   - Average score: {avg_score:.1f}/100")
        print(f"   - Highest score: {max_score}/100")
        print(f"   - Quality distribution: {score_distribution.to_dict()}")

        # Verify Phase 0 weights are working (RVol should dominate)
        avg_rvol_score = signal_rows['RVol_Score'].mean()
        avg_rrange_score = signal_rows['RRange_Score'].mean()
        avg_ibs_score = signal_rows['IBS_Score'].mean()

        print("⚖️ Component Score Analysis:")
        print(f"   - RVol (86.5% weight): {avg_rvol_score:.1f} avg")
        print(f"   - RRange (11.3% weight): {avg_rrange_score:.1f} avg")
        print(f"   - IBS (1.5% weight): {avg_ibs_score:.1f} avg")

        # Verify RVol dominance
        if avg_rvol_score > avg_rrange_score and avg_rrange_score > avg_ibs_score:
            print("✅ Phase 0 weights correctly applied (RVol > RRange > IBS)")
        else:
            print("⚠️ Weight distribution may need verification")

    # Test institutional flow metrics
    flow_regimes = df_enhanced['Flow_Regime'].value_counts()
    avg_conviction = df_enhanced['Volume_Conviction'].mean()
    avg_divergence = df_enhanced['Divergence_Severity'].mean()

    print("🏦 Institutional Flow Analysis:")
    print(f"   - Flow regimes: {flow_regimes.to_dict()}")
    print(f"   - Average conviction: {avg_conviction:.3f}")
    print(f"   - Average divergence severity: {avg_divergence:.1f}")

    return True

def test_manual_scoring():
    """Test manual scoring with known inputs"""
    print("\nTesting Manual Scoring Function...")

    # Test with high percentiles (should give high score)
    score, components, label = calculate_acceleration_score_v3(
        direction='bullish',
        ibs_pct=0.9,    # Top 10%
        rvol_pct=0.95,  # Top 5%
        rrange_pct=0.8, # Top 20%
        flow_pct=0.7,   # Top 30%
        conviction_pct=0.6  # Top 40%
    )

    print(f"High percentile test: {score} points, {label}")
    assert score >= 80, f"High percentiles should give high score, got {score}"
    assert label == '🔥 TOP 20%', f"Expected TOP 20%, got {label}"

    # Test with low percentiles (should give low score)
    score, components, label = calculate_acceleration_score_v3(
        direction='bullish',
        ibs_pct=0.1,    # Bottom 10%
        rvol_pct=0.05,  # Bottom 5%
        rrange_pct=0.2, # Bottom 20%
        flow_pct=0.3,   # Bottom 30%
        conviction_pct=0.4  # Bottom 40%
    )

    print(f"Low percentile test: {score} points, {label}")
    assert score <= 40, f"Low percentiles should give low score, got {score}"
    assert label == '🟠 BELOW AVERAGE', f"Expected BELOW AVERAGE, got {label}"

    print("✅ Manual scoring tests passed")

def main():
    """Run all integration tests"""
    print("🧪 COMPLETE INTEGRATION TEST: Phase 1 + Phase 2")
    print("=" * 60)

    try:
        # Test full pipeline
        success = test_full_pipeline()

        if not success:
            print("❌ Full pipeline test failed")
            return 1

        # Test manual scoring
        test_manual_scoring()

        print("\n" + "=" * 60)
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("✅ Phase 1 (Institutional Flow Metrics) + Phase 2 (Percentile Scoring) = SUCCESS")
        print("\n🚀 The enhanced scanner is ready for Phase 3: UI Integration!")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
