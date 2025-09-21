# File: analyze_results.py
"""
Simple Analysis of Your Breakout Results
"""

import pandas as pd
import numpy as np

def analyze_your_strategy():
    print("🔬 Analyzing Your Breakout Strategy Results")
    print("=" * 60)
    
    # Load your CSV file
    try:
        df = pd.read_csv('breakout_analysis_6913_breakouts.csv')
        print(f"✅ Loaded {len(df)} breakout records")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        print("Make sure 'breakout_analysis_6913_breakouts.csv' is in the same folder as this script")
        return
    
    # Calculate overall baseline
    total_breakouts = len(df)
    overall_success = df['success_binary'].sum()
    overall_success_rate = (overall_success / total_breakouts) * 100
    overall_avg_return = df['return_percentage'].mean() * 100
    
    print(f"\n📊 OVERALL BASELINE:")
    print(f"Total Breakouts: {total_breakouts:,}")
    print(f"Overall Success Rate: {overall_success_rate:.1f}%")
    print(f"Overall Average Return: {overall_avg_return:.2f}%")
    
    # Filter for IBS >= 0.3 (as your strategy uses)
    ibs_filtered = df[df['setup_ibs'] >= 0.3]
    print(f"\n📊 WITH IBS >= 0.3 FILTER:")
    print(f"Breakouts: {len(ibs_filtered):,}")
    print(f"Success Rate: {(ibs_filtered['success_binary'].sum() / len(ibs_filtered)) * 100:.1f}%")
    
    # Analyze your specific combination
    your_combo = df[
        (df['setup_mpi_trend'].isin(['Expanding', 'Flat'])) &
        (df['setup_ibs'] >= 0.3) &
        (df['setup_higher_hl'] == 1) &
        (df['setup_valid_crt'] == 1)
    ]
    
    your_success = your_combo['success_binary'].sum()
    your_total = len(your_combo)
    your_success_rate = (your_success / your_total) * 100 if your_total > 0 else 0
    your_avg_return = your_combo['return_percentage'].mean() * 100 if your_total > 0 else 0
    
    print(f"\n🎯 YOUR STRATEGY: Expanding+Flat + IBS≥0.3 + Higher_HL + Valid_CRT")
    print(f"Sample Size: {your_total:,} breakouts")
    print(f"Successes: {your_success}")
    print(f"Success Rate: {your_success_rate:.1f}%")
    print(f"Average Return: {your_avg_return:.2f}%")
    
    # Compare to random chance
    print(f"\n📈 PERFORMANCE COMPARISON:")
    print(f"Your Strategy:    {your_success_rate:.1f}%")
    print(f"Overall Baseline: {overall_success_rate:.1f}%")
    print(f"Random Chance:    50.0%")
    
    difference_vs_baseline = your_success_rate - overall_success_rate
    difference_vs_random = your_success_rate - 50.0
    
    print(f"\n📊 PERFORMANCE ANALYSIS:")
    print(f"Vs Baseline: {difference_vs_baseline:+.1f} percentage points")
    print(f"Vs Random:   {difference_vs_random:+.1f} percentage points")
    
    # Simple recommendation
    print(f"\n💡 SIMPLE RECOMMENDATION:")
    
    if your_success_rate < 50:
        print(f"❌ NOT RECOMMENDED: {your_success_rate:.1f}% is worse than random chance (50%)")
        print(f"❌ This strategy loses money on average")
    elif your_success_rate < overall_success_rate:
        print(f"⚠️  BELOW AVERAGE: {your_success_rate:.1f}% is worse than the overall baseline ({overall_success_rate:.1f}%)")
        print(f"⚠️  You might want to look for better strategies")
    elif your_success_rate > 52:
        print(f"✅ POTENTIALLY GOOD: {your_success_rate:.1f}% is above random chance")
        print(f"✅ This might be worth exploring further")
    else:
        print(f"😐 MARGINAL: {your_success_rate:.1f}% is only slightly better than random")
        print(f"😐 The edge is very small")
    
    # Look for better alternatives
    print(f"\n🔍 LOOKING FOR BETTER ALTERNATIVES...")
    
    # Test a few simple alternatives
    alternatives = [
        ("Just Expanding MPI", df[df['setup_mpi_trend'] == 'Expanding']),
        ("Just Valid CRT", df[df['setup_valid_crt'] == 1]),
        ("Just Higher HL", df[df['setup_higher_hl'] == 1]),
        ("Expanding + IBS≥0.5", df[(df['setup_mpi_trend'] == 'Expanding') & (df['setup_ibs'] >= 0.5)]),
        ("High IBS only (≥0.7)", df[df['setup_ibs'] >= 0.7])
    ]
    
    print(f"\n🏆 ALTERNATIVE STRATEGIES:")
    for name, alt_df in alternatives:
        if len(alt_df) >= 100:  # Only show if enough samples
            alt_success_rate = (alt_df['success_binary'].sum() / len(alt_df)) * 100
            alt_return = alt_df['return_percentage'].mean() * 100
            print(f"{name:25} | {alt_success_rate:5.1f}% | {len(alt_df):4d} samples | {alt_return:+5.2f}% return")
    
    print(f"\n" + "=" * 60)
    print(f"🎯 BOTTOM LINE:")
    if your_success_rate >= 52:
        print(f"Your strategy shows promise with {your_success_rate:.1f}% success rate")
    elif your_success_rate >= 50:
        print(f"Your strategy is barely better than random at {your_success_rate:.1f}%")
    else:
        print(f"Your strategy is losing strategy at {your_success_rate:.1f}% - avoid it")

if __name__ == "__main__":
    analyze_your_strategy()