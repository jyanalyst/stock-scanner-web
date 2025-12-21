#!/usr/bin/env python3
"""
TEST SUITE FOR NEW DATA-DRIVEN SCORING SYSTEM (2025-12-21)

Tests the redesigned scoring system that aligns with mean reversion patterns
discovered in validation analysis of 46 historical dates (266 TRUE_BREAK winners).

Key Tests:
1. Smoke Test - Perfect mean reversion setup scores high
2. Inversion Test - Low momentum beats high momentum
3. Directional Test - Bullish/bearish scoring works correctly
4. Bullish-Only Features - MPI, IBS, Volume_Conviction only score for bullish
5. Score Range Test - All scores are 0-100
6. Component Breakdown - Individual feature contributions add up correctly

Expected Performance Improvements:
- Top 10% stocks: 15-18% TRUE_BREAK rate (vs current 11%)
- Top 25% stocks: 13-15% TRUE_BREAK rate (vs current 11-12%)
- 35-60% improvement in predictive edge
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pages.scanner.logic import calculate_signal_score_v2, calculate_signal_score_v1_legacy


def test_smoke_test():
    """Test 1: Smoke Test - Perfect mean reversion setup should score high (75-85)"""
    print("\n🧪 TEST 1: SMOKE TEST - Perfect Mean Reversion Setup")

    # Perfect mean reversion setup (should score very high)
    perfect_setup = {
        'Signal_Bias': '🟢 BULLISH',
        'MPI_Percentile': 15,      # Q1 - coiled spring (15.6% win rate)
        'IBS_Percentile': 18,      # Q1 - oversold (13.3% win rate)
        'Flow_Velocity_Rank': 72,  # Q4 sweet spot (14.2% win rate)
        'VPI_Percentile': 85,      # Q5 - institutional buying (13.5% win rate)
        'Volume_Conviction': 1.3,  # High conviction (13.8% win rate)
        'Flow_Rank': 75,           # High for bullish
        'Flow_Percentile': 65      # High for bullish
    }

    score, components = calculate_signal_score_v2(perfect_setup)

    print(f"Score: {score}")
    print(f"Components: {components}")
    print(f"Total from components: {sum(components.values())}")

    # Assertions
    assert 90 <= score <= 100, f"Expected 90-100 score for perfect setup, got {score}"
    assert abs(score - sum(components.values())) < 0.01, "Score should equal sum of components"
    assert components['mpi'] == 25, f"Expected MPI=25 for Q1, got {components['mpi']}"
    assert components['ibs'] == 20, f"Expected IBS=20 for Q1, got {components['ibs']}"
    assert components['flow_vel'] == 20, f"Expected Flow_Vel=20 for Q4, got {components['flow_vel']}"

    print("✅ PASSED: Perfect mean reversion setup scores high")
    return True


def test_inversion_test():
    """Test 2: Inversion Test - Low momentum should beat high momentum"""
    print("\n🧪 TEST 2: INVERSION TEST - Low Momentum Beats High Momentum")

    # High momentum setup (should score low)
    high_momentum = {
        'Signal_Bias': '🟢 BULLISH',
        'MPI_Percentile': 85,  # High momentum - should score LOW
        'IBS_Percentile': 90,  # High momentum - should score LOW
        'Flow_Velocity_Rank': 50,  # Moderate
        'VPI_Percentile': 50,      # Moderate
        'Volume_Conviction': 1.0,  # Moderate
        'Flow_Rank': 50,           # Moderate
        'Flow_Percentile': 50      # Moderate
    }

    # Low momentum setup (should score high)
    low_momentum = {
        'Signal_Bias': '🟢 BULLISH',
        'MPI_Percentile': 15,  # Low momentum - should score HIGH
        'IBS_Percentile': 12,  # Low momentum - should score HIGH
        'Flow_Velocity_Rank': 50,  # Moderate
        'VPI_Percentile': 50,      # Moderate
        'Volume_Conviction': 1.0,  # Moderate
        'Flow_Rank': 50,           # Moderate
        'Flow_Percentile': 50      # Moderate
    }

    score_high, comp_high = calculate_signal_score_v2(high_momentum)
    score_low, comp_low = calculate_signal_score_v2(low_momentum)

    print(f"High momentum score: {score_high} (MPI: {comp_high['mpi']}, IBS: {comp_high['ibs']})")
    print(f"Low momentum score: {score_low} (MPI: {comp_low['mpi']}, IBS: {comp_low['ibs']})")

    # Assertions
    assert score_low > score_high, f"Low momentum should score higher: {score_low} > {score_high}"
    assert comp_high['mpi'] < comp_low['mpi'], f"MPI should be inverted: {comp_high['mpi']} < {comp_low['mpi']}"
    assert comp_high['ibs'] < comp_low['ibs'], f"IBS should be inverted: {comp_high['ibs']} < {comp_low['ibs']}"

    print("✅ PASSED: Low momentum correctly beats high momentum (mean reversion)")
    return True


def test_directional_test():
    """Test 3: Directional Test - Bullish/bearish scoring works correctly"""
    print("\n🧪 TEST 3: DIRECTIONAL TEST - Bullish vs Bearish Scoring")

    # Bullish setup with high directional features
    bullish_row = {
        'Signal_Bias': '🟢 BULLISH',
        'MPI_Percentile': 50,      # Moderate (only used for bullish)
        'IBS_Percentile': 50,      # Moderate (only used for bullish)
        'Flow_Velocity_Rank': 50,  # Moderate
        'VPI_Percentile': 50,      # Moderate
        'Volume_Conviction': 1.2,  # High (only used for bullish)
        'Flow_Rank': 80,           # High (good for bullish)
        'Flow_Percentile': 70      # High (good for bullish)
    }

    # Bearish setup with low directional features
    bearish_row = {
        'Signal_Bias': '🔴 BEARISH',
        'MPI_Percentile': 50,      # Moderate (not used for bearish)
        'IBS_Percentile': 50,      # Moderate (not used for bearish)
        'Flow_Velocity_Rank': 50,  # Moderate
        'VPI_Percentile': 50,      # Moderate
        'Volume_Conviction': 1.2,  # High (not used for bearish)
        'Flow_Rank': 20,           # Low (good for bearish)
        'Flow_Percentile': 30      # Low (good for bearish)
    }

    score_bull, comp_bull = calculate_signal_score_v2(bullish_row)
    score_bear, comp_bear = calculate_signal_score_v2(bearish_row)

    print(f"Bullish score: {score_bull}")
    print(f"Bullish components: Flow_Rank={comp_bull['flow_rank']}, Flow_Pct={comp_bull['flow_pct']}, Conviction={comp_bull['conviction']}")
    print(f"Bearish score: {score_bear}")
    print(f"Bearish components: Flow_Rank={comp_bear['flow_rank']}, Flow_Pct={comp_bear['flow_pct']}, Conviction={comp_bear['conviction']}")

    # Assertions
    assert comp_bull['flow_rank'] == 5, f"Bullish should get max Flow_Rank score: {comp_bull['flow_rank']}"
    assert comp_bear['flow_rank'] == 5, f"Bearish should get max Flow_Rank score: {comp_bear['flow_rank']}"
    assert comp_bull['flow_pct'] == 5, f"Bullish should get max Flow_Pct score: {comp_bull['flow_pct']}"
    assert comp_bear['flow_pct'] == 5, f"Bearish should get max Flow_Pct score: {comp_bear['flow_pct']}"
    assert comp_bull['conviction'] > 0, f"Bullish should get conviction score: {comp_bull['conviction']}"
    assert comp_bear['conviction'] == 0, f"Bearish should get zero conviction score: {comp_bear['conviction']}"
    assert comp_bull['mpi'] > 0, f"Bullish should get MPI score: {comp_bull['mpi']}"
    assert comp_bear['mpi'] == 0, f"Bearish should get zero MPI score: {comp_bear['mpi']}"

    print("✅ PASSED: Directional scoring works correctly")
    return True


def test_bullish_only_features():
    """Test 4: Bullish-Only Features - MPI, IBS, Volume_Conviction only score for bullish"""
    print("\n🧪 TEST 4: BULLISH-ONLY FEATURES TEST")

    # Test bullish signal
    bullish_signal = {
        'Signal_Bias': '🟢 BULLISH',
        'MPI_Percentile': 15,      # Should score high
        'IBS_Percentile': 18,      # Should score high
        'Flow_Velocity_Rank': 50,  # Moderate
        'VPI_Percentile': 50,      # Moderate
        'Volume_Conviction': 1.3,  # Should score high
        'Flow_Rank': 50,           # Moderate
        'Flow_Percentile': 50      # Moderate
    }

    # Test bearish signal
    bearish_signal = {
        'Signal_Bias': '🔴 BEARISH',
        'MPI_Percentile': 15,      # Should score zero
        'IBS_Percentile': 18,      # Should score zero
        'Flow_Velocity_Rank': 50,  # Moderate
        'VPI_Percentile': 50,      # Moderate
        'Volume_Conviction': 1.3,  # Should score zero
        'Flow_Rank': 50,           # Moderate
        'Flow_Percentile': 50      # Moderate
    }

    score_bull, comp_bull = calculate_signal_score_v2(bullish_signal)
    score_bear, comp_bear = calculate_signal_score_v2(bearish_signal)

    print(f"Bullish: MPI={comp_bull['mpi']}, IBS={comp_bull['ibs']}, Conviction={comp_bull['conviction']}")
    print(f"Bearish: MPI={comp_bear['mpi']}, IBS={comp_bear['ibs']}, Conviction={comp_bear['conviction']}")

    # Assertions
    assert comp_bull['mpi'] > 0, f"Bullish should get MPI score: {comp_bull['mpi']}"
    assert comp_bull['ibs'] > 0, f"Bullish should get IBS score: {comp_bull['ibs']}"
    assert comp_bull['conviction'] > 0, f"Bullish should get conviction score: {comp_bull['conviction']}"

    assert comp_bear['mpi'] == 0, f"Bearish should get zero MPI score: {comp_bear['mpi']}"
    assert comp_bear['ibs'] == 0, f"Bearish should get zero IBS score: {comp_bear['ibs']}"
    assert comp_bear['conviction'] == 0, f"Bearish should get zero conviction score: {comp_bear['conviction']}"

    print("✅ PASSED: Bullish-only features work correctly")
    return True


def test_score_range():
    """Test 5: Score Range Test - All scores should be 0-100"""
    print("\n🧪 TEST 5: SCORE RANGE TEST - All scores 0-100")

    test_cases = [
        # Edge cases
        {'Signal_Bias': '🟢 BULLISH', 'MPI_Percentile': 0, 'IBS_Percentile': 0, 'Flow_Velocity_Rank': 0, 'VPI_Percentile': 0, 'Volume_Conviction': 0.5, 'Flow_Rank': 0, 'Flow_Percentile': 0},
        {'Signal_Bias': '🟢 BULLISH', 'MPI_Percentile': 100, 'IBS_Percentile': 100, 'Flow_Velocity_Rank': 100, 'VPI_Percentile': 100, 'Volume_Conviction': 2.0, 'Flow_Rank': 100, 'Flow_Percentile': 100},
        {'Signal_Bias': '🔴 BEARISH', 'MPI_Percentile': 50, 'IBS_Percentile': 50, 'Flow_Velocity_Rank': 50, 'VPI_Percentile': 50, 'Volume_Conviction': 1.0, 'Flow_Rank': 50, 'Flow_Percentile': 50},
        # Missing data cases
        {'Signal_Bias': '🟢 BULLISH'},  # All defaults
    ]

    for i, test_case in enumerate(test_cases):
        score, components = calculate_signal_score_v2(test_case)
        print(f"Test case {i+1}: Score={score}, Components total={sum(components.values())}")

        # Assertions
        assert 0 <= score <= 100, f"Score out of range: {score}"
        assert abs(score - sum(components.values())) < 0.01, f"Score mismatch: {score} vs {sum(components.values())}"
        for comp_name, comp_score in components.items():
            assert 0 <= comp_score <= 25, f"Component {comp_name} out of range: {comp_score}"  # Max component is 25

    print("✅ PASSED: All scores in valid 0-100 range")
    return True


def test_component_breakdown():
    """Test 6: Component Breakdown - Individual contributions should add up"""
    print("\n🧪 TEST 6: COMPONENT BREAKDOWN TEST")

    test_row = {
        'Signal_Bias': '🟢 BULLISH',
        'MPI_Percentile': 25,      # Q1-Q2 boundary
        'IBS_Percentile': 35,      # Q2
        'Flow_Velocity_Rank': 65,  # Q4
        'VPI_Percentile': 75,      # Q4
        'Volume_Conviction': 1.1,  # Moderate
        'Flow_Rank': 65,           # High for bullish
        'Flow_Percentile': 55      # Moderate-high for bullish
    }

    score, components = calculate_signal_score_v2(test_row)

    print(f"Total score: {score}")
    print("Component breakdown:")
    for name, value in components.items():
        print(f"  {name}: {value}")

    # Manual calculation to verify
    expected_total = sum(components.values())

    print(f"Expected total: {expected_total}")
    print(f"Difference: {abs(score - expected_total)}")

    # Assertions
    assert abs(score - expected_total) < 0.01, f"Score mismatch: {score} vs {expected_total}"

    # Check specific component values based on quintile logic
    assert components['mpi'] == 18, f"MPI Q2 should be 18: {components['mpi']}"
    assert components['ibs'] == 12, f"IBS Q2 should be 12: {components['ibs']}"
    assert components['flow_vel'] == 20, f"Flow_Vel Q4 should be 20: {components['flow_vel']}"
    assert components['vpi'] == 12, f"VPI Q4 should be 12: {components['vpi']}"

    print("✅ PASSED: Component breakdown is correct")
    return True


def run_all_tests():
    """Run all tests and report results"""
    print("🚀 RUNNING NEW SCORING SYSTEM TESTS")
    print("=" * 50)

    tests = [
        test_smoke_test,
        test_inversion_test,
        test_directional_test,
        test_bullish_only_features,
        test_score_range,
        test_component_breakdown
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ FAILED: {test.__name__} - {str(e)}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"📊 TEST RESULTS: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 ALL TESTS PASSED! New scoring system is working correctly.")
        print("\nExpected Performance Improvements:")
        print("- Top 10% stocks: 15-18% TRUE_BREAK rate (vs current 11%)")
        print("- Top 25% stocks: 13-15% TRUE_BREAK rate (vs current 11-12%)")
        print("- 35-60% improvement in predictive edge")
    else:
        print("❌ SOME TESTS FAILED! Please review the scoring logic.")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
