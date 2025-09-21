# File: analyze_profit_targets.py
"""
Breakout Analysis with Fixed Profit Target Exit Strategies
Test 1%, 2%, 3%, 4%, 5% profit targets vs exit at close
"""

import pandas as pd
import numpy as np

def analyze_with_profit_targets():
    print("🎯 Breakout Analysis with Fixed Profit Target Exits")
    print("=" * 70)
    
    # Load your CSV file
    try:
        df = pd.read_csv('breakout_analysis_6913_breakouts.csv')
        print(f"✅ Loaded {len(df)} breakout records")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return
    
    # Calculate baseline (current exit at close strategy)
    print(f"\n📊 CURRENT STRATEGY (Exit at Close):")
    your_combo = df[
        (df['setup_mpi_trend'].isin(['Expanding', 'Flat'])) &
        (df['setup_ibs'] >= 0.3) &
        (df['setup_higher_hl'] == 1) &
        (df['setup_valid_crt'] == 1)
    ]
    
    print(f"Sample Size: {len(your_combo):,} breakouts")
    print(f"Success Rate: {(your_combo['success_binary'].sum() / len(your_combo)) * 100:.1f}%")
    print(f"Average Return: {your_combo['return_percentage'].mean() * 100:.2f}%")
    
    # Test different profit targets
    profit_targets = [0.01, 0.02, 0.03, 0.04, 0.05]  # 1%, 2%, 3%, 4%, 5%
    target_labels = ["1%", "2%", "3%", "4%", "5%"]
    
    print(f"\n🎯 TESTING PROFIT TARGET EXIT STRATEGIES:")
    print(f"{'Target':<8} {'Success Rate':<12} {'Avg Return':<12} {'Hit Rate':<10} {'Samples':<8}")
    print("-" * 60)
    
    results_summary = []
    
    for i, target in enumerate(profit_targets):
        # For each breakout, check if the profit target was reachable
        your_combo_copy = your_combo.copy()
        
        # Calculate if profit target was hit during the day
        # If breakout_high >= entry_price * (1 + target), then target was reachable
        target_price = your_combo_copy['entry_price'] * (1 + target)
        target_hit = your_combo_copy['breakout_high'] >= target_price
        
        # New success logic: 
        # - If target was hit during the day = SUCCESS (exit at target price)
        # - If target not hit = exit at close (original logic)
        new_success = target_hit.astype(int)
        
        # Calculate new returns:
        # - If target hit: return = target percentage
        # - If target not hit: return = original return (exit at close)
        new_returns = np.where(
            target_hit,
            target,  # Fixed return at target level
            your_combo_copy['return_percentage']  # Original return if target not hit
        )
        
        # Calculate metrics
        success_count = new_success.sum()
        total_count = len(your_combo_copy)
        success_rate = (success_count / total_count) * 100
        avg_return = new_returns.mean() * 100
        hit_rate = (target_hit.sum() / total_count) * 100
        
        results_summary.append({
            'target': target_labels[i],
            'success_rate': success_rate,
            'avg_return': avg_return,
            'hit_rate': hit_rate,
            'samples': total_count,
            'target_numeric': target
        })
        
        print(f"{target_labels[i]:<8} {success_rate:<12.1f} {avg_return:<12.2f} {hit_rate:<10.1f} {total_count:<8}")
    
    # Find the best profit target
    best_target = max(results_summary, key=lambda x: x['success_rate'])
    
    print(f"\n🏆 BEST PROFIT TARGET:")
    print(f"Target: {best_target['target']}")
    print(f"Success Rate: {best_target['success_rate']:.1f}%")
    print(f"Average Return: {best_target['avg_return']:.2f}%")
    print(f"Hit Rate: {best_target['hit_rate']:.1f}% (how often target was reachable)")
    
    # Compare to current strategy
    current_success = (your_combo['success_binary'].sum() / len(your_combo)) * 100
    improvement = best_target['success_rate'] - current_success
    
    print(f"\n📈 IMPROVEMENT ANALYSIS:")
    print(f"Current Strategy (Exit at Close): {current_success:.1f}%")
    print(f"Best Target Strategy ({best_target['target']}): {best_target['success_rate']:.1f}%")
    print(f"Improvement: {improvement:+.1f} percentage points")
    
    # Check if any target gets above 50%
    above_50 = [r for r in results_summary if r['success_rate'] > 50]
    
    print(f"\n🎯 PROFITABILITY CHECK:")
    if above_50:
        print(f"✅ SUCCESS! {len(above_50)} profit target(s) achieve >50% success rate:")
        for target in above_50:
            print(f"   • {target['target']}: {target['success_rate']:.1f}% success rate")
    else:
        print(f"❌ None of the profit targets achieve >50% success rate")
        print(f"❌ Even with optimized exits, this strategy struggles to be profitable")
    
    # Test the same analysis on simpler strategies
    print(f"\n🔍 TESTING PROFIT TARGETS ON SIMPLER STRATEGIES:")
    
    simple_strategies = [
        ("Just Valid CRT", df[df['setup_valid_crt'] == 1]),
        ("Just Higher HL", df[df['setup_higher_hl'] == 1]),
        ("Valid CRT + Higher HL", df[(df['setup_valid_crt'] == 1) & (df['setup_higher_hl'] == 1)])
    ]
    
    for strategy_name, strategy_df in simple_strategies:
        if len(strategy_df) >= 500:  # Only test strategies with decent sample size
            print(f"\n📊 {strategy_name} ({len(strategy_df)} samples):")
            print(f"{'Target':<8} {'Success Rate':<12} {'Hit Rate':<10}")
            print("-" * 35)
            
            best_simple = None
            best_simple_rate = 0
            
            for i, target in enumerate(profit_targets):
                target_price = strategy_df['entry_price'] * (1 + target)
                target_hit = strategy_df['breakout_high'] >= target_price
                success_rate = (target_hit.sum() / len(strategy_df)) * 100
                hit_rate = (target_hit.sum() / len(strategy_df)) * 100
                
                print(f"{target_labels[i]:<8} {success_rate:<12.1f} {hit_rate:<10.1f}")
                
                if success_rate > best_simple_rate:
                    best_simple_rate = success_rate
                    best_simple = target_labels[i]
            
            if best_simple_rate > 50:
                print(f"   ✅ PROFITABLE: {best_simple} target achieves {best_simple_rate:.1f}%")
            else:
                print(f"   ❌ Best: {best_simple} target only achieves {best_simple_rate:.1f}%")
    
    print(f"\n" + "=" * 70)
    print(f"🎯 FINAL RECOMMENDATIONS:")
    
    if best_target['success_rate'] > 50:
        print(f"✅ Use {best_target['target']} profit target for {best_target['success_rate']:.1f}% success rate")
        print(f"✅ This turns your strategy into a winning approach!")
    elif best_target['success_rate'] > 49:
        print(f"⚠️ {best_target['target']} target gives {best_target['success_rate']:.1f}% - close but not quite profitable")
        print(f"⚠️ Consider combining with other filters or look for different setups")
    else:
        print(f"❌ Even with best profit target ({best_target['target']}), only {best_target['success_rate']:.1f}% success")
        print(f"❌ This strategy needs fundamental changes, not just better exits")
    
    # Show practical trading advice
    print(f"\n💡 PRACTICAL TRADING ADVICE:")
    print(f"If you proceed with this strategy:")
    print(f"1. Set profit target at {best_target['target']} ({best_target['target_numeric']*100:.0f}%)")
    print(f"2. If target hit during the day: SELL immediately")
    print(f"3. If target not hit by close: EXIT at market close")
    print(f"4. Expected success rate: {best_target['success_rate']:.1f}%")
    print(f"5. Target will be hit {best_target['hit_rate']:.1f}% of the time")

if __name__ == "__main__":
    analyze_with_profit_targets()