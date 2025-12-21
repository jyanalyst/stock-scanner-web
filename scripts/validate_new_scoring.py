#!/usr/bin/env python3
"""
NEW SCORING SYSTEM VALIDATION SCRIPT (2025-12-21)

This script validates the redesigned scoring system by:
1. Loading historical selection data (46 dates, 266 TRUE_BREAK winners)
2. Calculating BOTH old (v1) and new (v2) scores for all historical stocks
3. Analyzing TRUE_BREAK rates by score quintiles/deciles
4. Comparing old vs new system performance
5. Generating comprehensive validation report

Expected Results:
- Old system: Top 10% = 11% TRUE_BREAK rate (baseline)
- New system: Top 10% = 15-18% TRUE_BREAK rate (+35-60% improvement)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pages.scanner.logic import calculate_signal_score_v1_legacy, calculate_signal_score_v2


def load_historical_data():
    """
    Load selection history with all scanned stocks and their outcomes
    
    Returns:
        DataFrame with columns: date, ticker, features, is_winner (TRUE_BREAK)
    """
    history_path = Path("data/feature_lab/selection_history.json")
    
    if not history_path.exists():
        raise FileNotFoundError(f"Selection history not found: {history_path}")
    
    print(f"📂 Loading historical data from {history_path}")
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Convert to flat DataFrame
    rows = []
    
    for date_str, date_data in history.get('dates', {}).items():
        scan_results = date_data.get('scan_results', {})
        
        for ticker, features in scan_results.items():
            row = {
                'date': date_str,
                'ticker': ticker,
                'is_winner': features.get('is_winner', False),
                **features  # Include all feature values
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    print(f"✅ Loaded {len(df)} stock-date combinations")
    print(f"   - Dates: {df['date'].nunique()}")
    print(f"   - Unique stocks: {df['ticker'].nunique()}")
    print(f"   - TRUE_BREAK winners: {df['is_winner'].sum()}")
    print(f"   - Non-winners: {(~df['is_winner']).sum()}")
    print(f"   - Baseline TRUE_BREAK rate: {df['is_winner'].mean():.1%}")
    
    return df


def calculate_scores(df):
    """
    Calculate both v1 (old) and v2 (new) scores for all stocks
    
    Args:
        df: DataFrame with feature columns
        
    Returns:
        DataFrame with added score columns
    """
    print("\n🔢 Calculating scores for all stocks...")
    
    # Calculate v1 (legacy) scores
    print("   - Calculating v1 (legacy) scores...")
    v1_scores = df.apply(lambda row: calculate_signal_score_v1_legacy(row), axis=1)
    df['score_v1'] = v1_scores
    
    # Calculate v2 (new) scores
    print("   - Calculating v2 (new) scores...")
    v2_results = df.apply(lambda row: calculate_signal_score_v2(row), axis=1)
    df['score_v2'] = v2_results.apply(lambda x: x[0])  # Extract score
    df['components_v2'] = v2_results.apply(lambda x: x[1])  # Extract components
    
    print(f"✅ Scores calculated")
    print(f"   - V1 score range: {df['score_v1'].min():.1f} - {df['score_v1'].max():.1f}")
    print(f"   - V2 score range: {df['score_v2'].min():.1f} - {df['score_v2'].max():.1f}")
    
    return df


def analyze_quintile_performance(df, score_column, label):
    """
    Analyze TRUE_BREAK rate by score quintiles
    
    Args:
        df: DataFrame with scores and is_winner
        score_column: Name of score column to analyze
        label: Label for this analysis (e.g., "V1 Legacy", "V2 New")
        
    Returns:
        dict with quintile analysis results
    """
    print(f"\n📊 Analyzing {label} quintile performance...")
    
    # Create quintiles based on score
    df['quintile'] = pd.qcut(df[score_column], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
    
    # Calculate TRUE_BREAK rate by quintile
    quintile_stats = []
    
    for quintile in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        quintile_df = df[df['quintile'] == quintile]
        
        if len(quintile_df) > 0:
            win_rate = quintile_df['is_winner'].mean()
            count = len(quintile_df)
            winners = quintile_df['is_winner'].sum()
            
            quintile_stats.append({
                'quintile': quintile,
                'win_rate': win_rate,
                'count': count,
                'winners': winners
            })
    
    quintile_df_stats = pd.DataFrame(quintile_stats)
    
    # Print results
    print(f"\n{label} Quintile Performance:")
    print("=" * 60)
    for _, row in quintile_df_stats.iterrows():
        print(f"{row['quintile']}: {row['win_rate']:.1%} win rate ({row['winners']}/{row['count']} stocks)")
    
    # Calculate improvement vs baseline
    baseline = df['is_winner'].mean()
    best_quintile = quintile_df_stats.loc[quintile_df_stats['win_rate'].idxmax()]
    worst_quintile = quintile_df_stats.loc[quintile_df_stats['win_rate'].idxmin()]
    
    print(f"\nBaseline: {baseline:.1%}")
    print(f"Best quintile ({best_quintile['quintile']}): {best_quintile['win_rate']:.1%} ({(best_quintile['win_rate']/baseline - 1)*100:+.1f}% vs baseline)")
    print(f"Worst quintile ({worst_quintile['quintile']}): {worst_quintile['win_rate']:.1%} ({(worst_quintile['win_rate']/baseline - 1)*100:+.1f}% vs baseline)")
    print(f"Spread: {(best_quintile['win_rate'] - worst_quintile['win_rate']):.1%}")
    
    return {
        'quintile_stats': quintile_df_stats,
        'baseline': baseline,
        'best_quintile': best_quintile,
        'worst_quintile': worst_quintile,
        'spread': best_quintile['win_rate'] - worst_quintile['win_rate']
    }


def analyze_top_precision(df, score_column, label):
    """
    Analyze TRUE_BREAK rate for top-ranked stocks
    
    Args:
        df: DataFrame with scores and is_winner
        score_column: Name of score column to analyze
        label: Label for this analysis
        
    Returns:
        dict with precision analysis results
    """
    print(f"\n🎯 Analyzing {label} top-ranked precision...")
    
    # Sort by score (descending)
    df_sorted = df.sort_values(score_column, ascending=False).copy()
    
    # Calculate precision at different thresholds
    thresholds = [0.10, 0.25, 0.50]
    precision_stats = []
    
    for threshold in thresholds:
        n_stocks = int(len(df_sorted) * threshold)
        top_stocks = df_sorted.head(n_stocks)
        
        win_rate = top_stocks['is_winner'].mean()
        count = len(top_stocks)
        winners = top_stocks['is_winner'].sum()
        
        precision_stats.append({
            'threshold': f'Top {int(threshold*100)}%',
            'win_rate': win_rate,
            'count': count,
            'winners': winners
        })
    
    precision_df = pd.DataFrame(precision_stats)
    
    # Print results
    print(f"\n{label} Top-Ranked Precision:")
    print("=" * 60)
    baseline = df['is_winner'].mean()
    
    for _, row in precision_df.iterrows():
        improvement = (row['win_rate'] / baseline - 1) * 100
        print(f"{row['threshold']}: {row['win_rate']:.1%} ({row['winners']}/{row['count']}) - {improvement:+.1f}% vs baseline")
    
    return {
        'precision_stats': precision_df,
        'baseline': baseline
    }


def compare_systems(df):
    """
    Side-by-side comparison of v1 vs v2 scoring systems
    
    Args:
        df: DataFrame with both v1 and v2 scores
        
    Returns:
        dict with comparison results
    """
    print("\n" + "=" * 80)
    print("📊 SYSTEM COMPARISON: V1 (Legacy) vs V2 (New Data-Driven)")
    print("=" * 80)
    
    # Analyze both systems
    v1_quintile = analyze_quintile_performance(df, 'score_v1', 'V1 Legacy')
    v2_quintile = analyze_quintile_performance(df, 'score_v2', 'V2 New')
    
    v1_precision = analyze_top_precision(df, 'score_v1', 'V1 Legacy')
    v2_precision = analyze_top_precision(df, 'score_v2', 'V2 New')
    
    # Calculate improvements
    print("\n" + "=" * 80)
    print("🚀 IMPROVEMENTS (V2 vs V1)")
    print("=" * 80)
    
    print("\nTop-Ranked Precision Improvements:")
    for i, threshold in enumerate(['Top 10%', 'Top 25%', 'Top 50%']):
        v1_rate = v1_precision['precision_stats'].iloc[i]['win_rate']
        v2_rate = v2_precision['precision_stats'].iloc[i]['win_rate']
        improvement = (v2_rate / v1_rate - 1) * 100 if v1_rate > 0 else 0
        
        print(f"{threshold}:")
        print(f"  V1: {v1_rate:.1%}")
        print(f"  V2: {v2_rate:.1%}")
        print(f"  Improvement: {improvement:+.1f}%")
    
    print(f"\nQuintile Spread:")
    print(f"  V1: {v1_quintile['spread']:.1%}")
    print(f"  V2: {v2_quintile['spread']:.1%}")
    print(f"  Improvement: {(v2_quintile['spread'] - v1_quintile['spread']):.1%}")
    
    return {
        'v1_quintile': v1_quintile,
        'v2_quintile': v2_quintile,
        'v1_precision': v1_precision,
        'v2_precision': v2_precision
    }


def generate_report(comparison_results, output_path):
    """
    Generate markdown validation report
    
    Args:
        comparison_results: Results from compare_systems()
        output_path: Path to save report
    """
    print(f"\n📝 Generating validation report...")
    
    report_lines = []
    
    # Header
    report_lines.append("# New Scoring System Validation Report")
    report_lines.append(f"\n**Generated:** {datetime.now().isoformat()}")
    report_lines.append("\n## Executive Summary")
    
    # Top 10% comparison
    v1_top10 = comparison_results['v1_precision']['precision_stats'].iloc[0]['win_rate']
    v2_top10 = comparison_results['v2_precision']['precision_stats'].iloc[0]['win_rate']
    improvement_top10 = (v2_top10 / v1_top10 - 1) * 100 if v1_top10 > 0 else 0
    
    report_lines.append(f"\n**Key Finding:** The new V2 scoring system achieves a **{v2_top10:.1%}** TRUE_BREAK rate for top 10% ranked stocks, compared to **{v1_top10:.1%}** for the legacy V1 system.")
    report_lines.append(f"\n**Improvement:** {improvement_top10:+.1f}% increase in predictive accuracy.")
    
    # Baseline
    baseline = comparison_results['v1_precision']['baseline']
    report_lines.append(f"\n**Baseline:** {baseline:.1%} overall TRUE_BREAK rate")
    
    # Quintile comparison table
    report_lines.append("\n## Quintile Performance Comparison")
    report_lines.append("\n| Quintile | V1 Legacy | V2 New | Improvement |")
    report_lines.append("|----------|-----------|--------|-------------|")
    
    v1_quintiles = comparison_results['v1_quintile']['quintile_stats']
    v2_quintiles = comparison_results['v2_quintile']['quintile_stats']
    
    for i in range(len(v1_quintiles)):
        quintile = v1_quintiles.iloc[i]['quintile']
        v1_rate = v1_quintiles.iloc[i]['win_rate']
        v2_rate = v2_quintiles.iloc[i]['win_rate']
        improvement = (v2_rate / v1_rate - 1) * 100 if v1_rate > 0 else 0
        
        report_lines.append(f"| {quintile} | {v1_rate:.1%} | {v2_rate:.1%} | {improvement:+.1f}% |")
    
    # Top-ranked precision table
    report_lines.append("\n## Top-Ranked Precision Comparison")
    report_lines.append("\n| Threshold | V1 Legacy | V2 New | Improvement |")
    report_lines.append("|-----------|-----------|--------|-------------|")
    
    v1_precision = comparison_results['v1_precision']['precision_stats']
    v2_precision = comparison_results['v2_precision']['precision_stats']
    
    for i in range(len(v1_precision)):
        threshold = v1_precision.iloc[i]['threshold']
        v1_rate = v1_precision.iloc[i]['win_rate']
        v2_rate = v2_precision.iloc[i]['win_rate']
        improvement = (v2_rate / v1_rate - 1) * 100 if v1_rate > 0 else 0
        
        report_lines.append(f"| {threshold} | {v1_rate:.1%} | {v2_rate:.1%} | {improvement:+.1f}% |")
    
    # Key metrics
    report_lines.append("\n## Key Metrics")
    
    v1_spread = comparison_results['v1_quintile']['spread']
    v2_spread = comparison_results['v2_quintile']['spread']
    
    report_lines.append(f"\n**Quintile Spread (Best - Worst):**")
    report_lines.append(f"- V1 Legacy: {v1_spread:.1%}")
    report_lines.append(f"- V2 New: {v2_spread:.1%}")
    report_lines.append(f"- Improvement: {(v2_spread - v1_spread):.1%}")
    
    # Conclusion
    report_lines.append("\n## Conclusion")
    
    if improvement_top10 > 20:
        report_lines.append(f"\n✅ **SUCCESS:** The new V2 scoring system shows a **{improvement_top10:+.1f}% improvement** in top 10% precision.")
        report_lines.append("\nThe redesigned system successfully aligns with mean reversion patterns and provides a meaningful edge over the legacy system.")
    elif improvement_top10 > 0:
        report_lines.append(f"\n⚠️ **MODERATE IMPROVEMENT:** The new V2 scoring system shows a **{improvement_top10:+.1f}% improvement** in top 10% precision.")
        report_lines.append("\nThe system shows positive results but may benefit from further refinement.")
    else:
        report_lines.append(f"\n❌ **NO IMPROVEMENT:** The new V2 scoring system shows a **{improvement_top10:+.1f}% change** in top 10% precision.")
        report_lines.append("\nThe redesigned system did not improve over the legacy system. Further analysis needed.")
    
    # Write report
    report_text = '\n'.join(report_lines)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"✅ Report saved to: {output_path}")
    
    return report_text


def main():
    """Main validation workflow"""
    print("🚀 NEW SCORING SYSTEM VALIDATION")
    print("=" * 80)
    
    try:
        # 1. Load historical data
        df = load_historical_data()
        
        # 2. Calculate scores
        df = calculate_scores(df)
        
        # 3. Compare systems
        comparison_results = compare_systems(df)
        
        # 4. Generate report
        output_path = Path("data/feature_lab/new_scoring_validation_report.md")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = generate_report(comparison_results, output_path)
        
        # 5. Summary
        print("\n" + "=" * 80)
        print("✅ VALIDATION COMPLETE")
        print("=" * 80)
        print(f"\nReport location: {output_path}")
        
        # Print key finding
        v1_top10 = comparison_results['v1_precision']['precision_stats'].iloc[0]['win_rate']
        v2_top10 = comparison_results['v2_precision']['precision_stats'].iloc[0]['win_rate']
        improvement = (v2_top10 / v1_top10 - 1) * 100 if v1_top10 > 0 else 0
        
        print(f"\n🎯 KEY FINDING:")
        print(f"   Top 10% TRUE_BREAK Rate:")
        print(f"   - V1 Legacy: {v1_top10:.1%}")
        print(f"   - V2 New:    {v2_top10:.1%}")
        print(f"   - Improvement: {improvement:+.1f}%")
        
        if improvement > 20:
            print("\n🎉 SUCCESS! The new scoring system shows significant improvement!")
        elif improvement > 0:
            print("\n✓ The new scoring system shows positive improvement.")
        else:
            print("\n⚠️ WARNING: The new scoring system did not improve performance.")
            print("   Further analysis and refinement may be needed.")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
