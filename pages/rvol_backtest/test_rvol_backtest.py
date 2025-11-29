# Test Script for RVOL BackTest
"""
Comprehensive testing and validation for RVOL BackTest components
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pages.rvol_backtest.vwap_engine import calculate_monthly_vwap, validate_vwap_calculation
from pages.rvol_backtest.backtest_engine import backtest_strategy_with_retest, validate_backtest_results
from pages.rvol_backtest.optimizer import run_parameter_optimization, validate_optimization_results

def create_test_data():
    """Create synthetic test data for validation"""
    # Create 2 years of daily data (500 trading days)
    dates = pd.date_range('2022-01-01', '2023-12-31', freq='B')  # Business days only

    np.random.seed(42)  # For reproducible results

    # Generate realistic price data
    n_days = len(dates)

    # Start with base price of $2.00
    prices = [2.00]

    for i in range(1, n_days):
        # Random walk with slight upward trend
        change = np.random.normal(0.001, 0.02)  # Mean 0.1%, std 2%
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 0.1))  # Floor at $0.10

    # Create OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Generate OHLC around close price
        volatility = 0.02  # 2% daily volatility
        high = close * (1 + abs(np.random.normal(0, volatility)))
        low = close * (1 - abs(np.random.normal(0, volatility)))
        open_price = close * (1 + np.random.normal(0, volatility/2))

        # Ensure OHLC relationships
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        # Volume (realistic range)
        volume = int(np.random.normal(100000, 30000))

        data.append({
            'Date': date,
            'Open': round(open_price, 4),
            'High': round(high, 4),
            'Low': round(low, 4),
            'Close': round(close, 4),
            'Volume': max(volume, 1000)  # Minimum volume
        })

    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')

    return df

def test_vwap_engine():
    """Test VWAP engine functionality"""
    print("🧪 Testing VWAP Engine...")

    # Create test data
    df = create_test_data()

    # Test VWAP calculation
    try:
        df_with_vwap = calculate_monthly_vwap(df.copy())

        # Basic validation
        assert 'Monthly_VWAP' in df_with_vwap.columns, "Monthly_VWAP column missing"
        assert not df_with_vwap['Monthly_VWAP'].isna().all(), "All VWAP values are NaN"

        # Check monthly resets
        validation = validate_vwap_calculation(df_with_vwap)
        assert validation['vwap_completeness'] > 0.9, f"VWAP completeness too low: {validation['vwap_completeness']}"

        print("✅ VWAP Engine: PASSED")
        print(f"   - Data points: {len(df_with_vwap)}")
        print(f"   - Months covered: {validation['months_covered']}")
        print(".2f")

        return df_with_vwap

    except Exception as e:
        print(f"❌ VWAP Engine: FAILED - {e}")
        return None

def test_backtest_engine(df_with_vwap):
    """Test backtest engine functionality"""
    print("\n🧪 Testing Backtest Engine...")

    if df_with_vwap is None:
        print("❌ Backtest Engine: SKIPPED - VWAP data unavailable")
        return None

    try:
        # Test with a single threshold
        threshold = 0.05  # 5%
        trades, metrics, fill_stats = backtest_strategy_with_retest(
            df=df_with_vwap,
            deviation_threshold=threshold,
            stop_loss=0.012,
            position_size=50000
        )

        # Validate results
        validation = validate_backtest_results(trades, fill_stats)
        assert validation['is_valid'], f"Backtest validation failed: {validation['issues']}"

        # Check reasonable ranges
        assert 0 <= fill_stats['fill_rate'] <= 100, f"Unrealistic fill rate: {fill_stats['fill_rate']}%"
        assert metrics['total_trades'] >= 0, f"Negative trade count: {metrics['total_trades']}"

        print("✅ Backtest Engine: PASSED")
        print(f"   - Signals generated: {fill_stats['signals_generated']}")
        print(f"   - Orders filled: {fill_stats['orders_filled']}")
        print(".1f")
        print(f"   - Total trades: {metrics['total_trades']}")
        print(".2f")

        return trades, metrics, fill_stats

    except Exception as e:
        print(f"❌ Backtest Engine: FAILED - {e}")
        return None

def test_optimizer(df_with_vwap):
    """Test optimizer functionality"""
    print("\n🧪 Testing Optimizer...")

    if df_with_vwap is None:
        print("❌ Optimizer: SKIPPED - VWAP data unavailable")
        return None

    try:
        # Run optimization with limited range for speed
        results_df = run_parameter_optimization(
            df=df_with_vwap,
            threshold_min=0.03,  # 3%
            threshold_max=0.08,  # 8%
            threshold_step=0.01, # 1% steps
            stop_loss=0.012,
            position_size=50000
        )

        # Validate results
        validation = validate_optimization_results(results_df)
        assert validation['is_valid'], f"Optimization validation failed: {validation['issues']}"

        # Check we have results
        assert not results_df.empty, "No optimization results"
        assert len(results_df) >= 3, f"Too few threshold tests: {len(results_df)}"

        # Check sorting (should be by Sharpe ratio descending)
        sharpe_values = results_df['sharpe_ratio'].values
        assert np.all(sharpe_values[:-1] >= sharpe_values[1:]), "Results not sorted by Sharpe ratio"

        print("✅ Optimizer: PASSED")
        print(f"   - Thresholds tested: {len(results_df)}")
        print(".2f")
        print(".1f")

        return results_df

    except Exception as e:
        print(f"❌ Optimizer: FAILED - {e}")
        return None

def run_all_tests():
    """Run complete test suite"""
    print("🚀 RVOL BackTest - Test Suite")
    print("=" * 50)

    # Test VWAP Engine
    df_with_vwap = test_vwap_engine()

    # Test Backtest Engine
    backtest_result = test_backtest_engine(df_with_vwap)

    # Test Optimizer
    optimizer_result = test_optimizer(df_with_vwap)

    # Summary
    print("\n" + "=" * 50)
    if df_with_vwap is not None and backtest_result is not None and optimizer_result is not None:
        print("🎉 ALL TESTS PASSED!")
        print("\n📊 Test Summary:")
        print(f"   - VWAP calculations: ✅ ({len(df_with_vwap)} data points)")
        print(f"   - Backtest execution: ✅ ({backtest_result[1]['total_trades']} trades)")
        print(f"   - Optimization sweep: ✅ ({len(optimizer_result)} thresholds)")
        print("\n🚀 RVOL BackTest is ready for use!")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please check the error messages above and fix issues before using.")

if __name__ == "__main__":
    run_all_tests()
