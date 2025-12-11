"""
Test Script for Cross-Sectional Ranks and Time-Decay Features
Validates the implementation before running full Phase 1 data collection
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.data_collection import add_cross_sectional_percentiles
from core.technical_analysis import add_time_decay_features, add_enhanced_columns
from core.local_file_loader import get_local_loader

def test_cross_sectional_ranks():
    """Test cross-sectional rank calculation"""
    print("\n" + "="*80)
    print("TEST 1: Cross-Sectional Ranks")
    print("="*80)
    
    # Create sample data with 3 stocks on same date
    sample_data = pd.DataFrame({
        'entry_date': [datetime(2024, 1, 1)] * 3,
        'Ticker': ['A17U.SG', 'CICT.SG', 'D05.SG'],
        'MPI_Percentile': [85.0, 60.0, 40.0],
        'IBS_Percentile': [90.0, 70.0, 50.0],
        'VPI_Percentile': [75.0, 55.0, 35.0],
        'IBS_Accel': [0.05, 0.02, -0.01],
        'RVol_Accel': [0.3, 0.1, -0.1],
        'RRange_Accel': [0.2, 0.0, -0.2],
        'VPI_Accel': [0.1, 0.05, -0.05],
        'Flow_Price_Gap': [15.0, 5.0, -10.0],
    })
    
    print(f"\n📊 Input data ({len(sample_data)} stocks):")
    print(sample_data[['Ticker', 'MPI_Percentile', 'IBS_Percentile', 'VPI_Percentile']])
    
    # Apply CS ranks
    result = add_cross_sectional_percentiles(sample_data)
    
    # Check results
    cs_cols = [col for col in result.columns if col.endswith('_CS_Rank')]
    print(f"\n✅ Added {len(cs_cols)} CS rank columns:")
    for col in cs_cols:
        print(f"   - {col}")
    
    # Verify rankings
    print(f"\n📈 Sample rankings (MPI_Percentile):")
    print(result[['Ticker', 'MPI_Percentile', 'MPI_Percentile_CS_Rank']].sort_values('MPI_Percentile_CS_Rank', ascending=False))
    
    # Validate
    assert len(cs_cols) == 8, f"Expected 8 CS rank columns, got {len(cs_cols)}"
    assert 'MPI_Percentile_CS_Rank' in result.columns, "MPI_Percentile_CS_Rank missing"
    assert 'IBS_Percentile_CS_Rank' in result.columns, "IBS_Percentile_CS_Rank missing"
    assert 'VPI_Percentile_CS_Rank' in result.columns, "VPI_Percentile_CS_Rank missing"
    
    # Check scale (0-100)
    for col in cs_cols:
        assert result[col].min() >= 0, f"{col} has values < 0"
        assert result[col].max() <= 100, f"{col} has values > 100"
    
    print("\n✅ TEST 1 PASSED: Cross-sectional ranks working correctly!")
    return True


def test_time_decay_features():
    """Test time-decay feature calculation"""
    print("\n" + "="*80)
    print("TEST 2: Time-Decay Features")
    print("="*80)
    
    # Load real stock data
    loader = get_local_loader()
    
    try:
        print("\n📥 Loading A17U.SG historical data...")
        df = loader.load_historical_data("A17U.SG")
        
        # Ensure Date is index
        if 'Date' in df.columns:
            df = df.set_index('Date')
        
        print(f"✅ Loaded {len(df)} days of data")
        print(f"   Date range: {df.index.min()} to {df.index.max()}")
        
        # Run full technical analysis (includes time-decay)
        print("\n🔧 Running technical analysis with time-decay features...")
        df_enhanced = add_enhanced_columns(df, ticker="A17U.SG")
        
        # Check for time-decay columns
        time_decay_cols = [
            'Days_Since_High',
            'Days_Since_Low',
            'Days_Since_Triple_Aligned',
            'Days_Since_Flow_Regime_Change'
        ]
        
        found_cols = [col for col in time_decay_cols if col in df_enhanced.columns]
        print(f"\n✅ Found {len(found_cols)}/{len(time_decay_cols)} time-decay features:")
        
        for col in found_cols:
            max_val = df_enhanced[col].max()
            min_val = df_enhanced[col].min()
            mean_val = df_enhanced[col].mean()
            print(f"   - {col}: min={min_val:.0f}, max={max_val:.0f}, mean={mean_val:.1f}")
        
        # Validate
        assert len(found_cols) >= 2, f"Expected at least 2 time-decay features, found {len(found_cols)}"
        assert 'Days_Since_High' in found_cols, "Days_Since_High missing"
        assert 'Days_Since_Low' in found_cols, "Days_Since_Low missing (CRITICAL)"
        
        # Check non-negative
        for col in found_cols:
            assert df_enhanced[col].min() >= 0, f"{col} has negative values"
        
        # Show recent values
        print(f"\n📊 Recent values (last 5 days):")
        display_cols = ['Close'] + found_cols
        print(df_enhanced[display_cols].tail())
        
        print("\n✅ TEST 2 PASSED: Time-decay features working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test full integration with data collection"""
    print("\n" + "="*80)
    print("TEST 3: Integration Test")
    print("="*80)
    
    print("\n🔧 Testing integration with MLDataCollector...")
    
    # Create sample scan results
    scan_results = pd.DataFrame({
        'Ticker': ['A17U.SG', 'CICT.SG'],
        'Close': [3.50, 2.80],
        'MPI_Percentile': [75.0, 55.0],
        'IBS_Percentile': [80.0, 60.0],
        'VPI_Percentile': [70.0, 50.0],
        'IBS_Accel': [0.03, 0.01],
        'RVol_Accel': [0.2, 0.1],
        'RRange_Accel': [0.1, 0.0],
        'VPI_Accel': [0.05, 0.02],
        'Flow_Price_Gap': [10.0, 5.0],
        'Date': [datetime(2024, 1, 1)] * 2,
    })
    
    print(f"\n📊 Sample scan results ({len(scan_results)} stocks):")
    print(scan_results[['Ticker', 'Close', 'MPI_Percentile']])
    
    # Simulate labeled samples
    labeled_samples = []
    for _, stock in scan_results.iterrows():
        sample = {
            **stock.to_dict(),
            'entry_date': datetime(2024, 1, 1),
            'entry_price': stock['Close'],
            'return_2d': 0.02,
            'win_2d': True,
        }
        labeled_samples.append(sample)
    
    # Convert to DataFrame and add CS ranks
    labeled_samples_df = pd.DataFrame(labeled_samples)
    
    print(f"\n🔧 Adding cross-sectional ranks...")
    labeled_samples_df = add_cross_sectional_percentiles(labeled_samples_df)
    
    # Check results
    cs_cols = [col for col in labeled_samples_df.columns if col.endswith('_CS_Rank')]
    print(f"✅ Added {len(cs_cols)} CS rank columns")
    
    # Show final result
    display_cols = ['Ticker', 'MPI_Percentile', 'MPI_Percentile_CS_Rank', 'IBS_Percentile', 'IBS_Percentile_CS_Rank']
    print(f"\n📈 Final result:")
    print(labeled_samples_df[display_cols])
    
    print("\n✅ TEST 3 PASSED: Integration working correctly!")
    return True


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("CROSS-SECTIONAL RANKS & TIME-DECAY FEATURES VALIDATION")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    # Test 1: CS Ranks
    try:
        results.append(("Cross-Sectional Ranks", test_cross_sectional_ranks()))
    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Cross-Sectional Ranks", False))
    
    # Test 2: Time-Decay
    try:
        results.append(("Time-Decay Features", test_time_decay_features()))
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Time-Decay Features", False))
    
    # Test 3: Integration
    try:
        results.append(("Integration Test", test_integration()))
    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Integration Test", False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n✅ Ready to run Phase 1 data collection:")
        print("   python scripts/run_ml_collection_clean.py")
    else:
        print("\n⚠️  SOME TESTS FAILED - Please fix issues before running Phase 1")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
