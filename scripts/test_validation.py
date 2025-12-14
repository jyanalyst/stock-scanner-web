"""
Test Script for Phase 4: Model Validation
Run walk-forward validation and threshold optimization
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.validator import MLValidator
import json
from datetime import datetime


def print_section(title):
    """Print section header"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80 + "\n")


def print_metrics(metrics, indent=0):
    """Print metrics dictionary"""
    prefix = "  " * indent
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_metrics(value, indent + 1)
        elif isinstance(value, float):
            print(f"{prefix}{key}: {value:.4f}")
        else:
            print(f"{prefix}{key}: {value}")


def main():
    print_section("PHASE 4: MODEL VALIDATION")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Initialize validator
    print("1️⃣  Initializing validator...")
    validator = MLValidator(target='win_3d')
    print("✅ Validator initialized\n")
    
    # ===== WALK-FORWARD VALIDATION =====
    print_section("STEP 1: WALK-FORWARD VALIDATION")
    print("Testing model on 2024 quarters (Q1, Q2, Q3, Q4)...")
    print("This will take 2-3 minutes...\n")
    
    try:
        walk_forward_results = validator.walk_forward_validation(
            test_periods=[
                ('2024-01-01', '2024-03-31'),  # Q1
                ('2024-04-01', '2024-06-30'),  # Q2
                ('2024-07-01', '2024-09-30'),  # Q3
                ('2024-10-01', '2024-12-31'),  # Q4
            ],
            embargo_days=20  # Prevent data leakage from rolling windows
        )
        
        print("✅ Walk-forward validation complete!\n")
        
        # Print results
        print("📊 RESULTS BY QUARTER:")
        print("-" * 80)
        for period_result in walk_forward_results['period_metrics']:
            print(f"\n{period_result['period']} ({period_result['test_samples']} test samples, {period_result['train_samples']} train samples):")
            metrics = period_result['metrics']
            print(f"  Accuracy:  {metrics.get('accuracy', 0):.2%}")
            print(f"  Precision: {metrics.get('precision', 0):.2%}")
            print(f"  Recall:    {metrics.get('recall', 0):.2%}")
            print(f"  F1-Score:  {metrics.get('f1', 0):.2%}")
            if 'win_rate' in metrics:
                print(f"  Win Rate:  {metrics.get('win_rate', 0):.2%}")
            if 'total_return' in metrics:
                print(f"  Total Return: {metrics.get('total_return', 0):.2%}")
        
        print("\n" + "-" * 80)
        print("\n📈 OVERALL PERFORMANCE (All 2024):")
        overall = walk_forward_results['overall_metrics']
        print(f"  Total Samples: {walk_forward_results.get('n_total_samples', 0):,}")
        print(f"  Accuracy:  {overall.get('accuracy', 0):.2%}")
        print(f"  Precision: {overall.get('precision', 0):.2%}")
        print(f"  Recall:    {overall.get('recall', 0):.2%}")
        print(f"  F1-Score:  {overall.get('f1', 0):.2%}")
        print(f"  ROC-AUC:   {overall.get('roc_auc', 0):.4f}")
        
        # Calculate stability
        accuracies = [p['metrics'].get('accuracy', 0) for p in walk_forward_results['period_metrics']]
        import numpy as np
        acc_std = np.std(accuracies)
        print(f"\n  Accuracy Std Dev: {acc_std:.4f} (lower is better)")
        if acc_std < 0.05:
            print("  ✅ Performance is STABLE across quarters")
        else:
            print("  ⚠️  Performance varies across quarters")
        
    except Exception as e:
        print(f"❌ Walk-forward validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ===== THRESHOLD OPTIMIZATION =====
    print_section("STEP 2: THRESHOLD OPTIMIZATION")
    print("Testing confidence thresholds: 0.55, 0.60, 0.65, 0.70...")
    print("This will take 1-2 minutes...\n")
    
    try:
        threshold_results = validator.optimize_threshold(
            confidence_thresholds=[0.55, 0.60, 0.65, 0.70]
        )
        
        print("✅ Threshold optimization complete!\n")
        
        # Print results
        print("📊 RESULTS BY THRESHOLD:")
        print("-" * 80)
        for result in threshold_results['threshold_results']:
            print(f"\nThreshold: {result['threshold']:.2f}")
            print(f"  Trades:        {result['n_trades']:,} ({result['trade_rate']:.1%} of samples)")
            print(f"  Win Rate:      {result.get('win_rate', 0):.2%}")
            if 'total_return' in result:
                print(f"  Total Return:  {result.get('total_return', 0):.2%}")
                print(f"  Mean Return:   {result.get('mean_return', 0):.2%}")
                print(f"  Profit Factor: {result.get('profit_factor', 0):.2f}")
        
        print("\n" + "-" * 80)
        print("\n🎯 OPTIMAL THRESHOLD:")
        optimal = threshold_results['optimal_threshold']
        print(f"  Threshold:     {optimal['threshold']:.2f}")
        print(f"  Trades:        {optimal['n_trades']:,}")
        print(f"  Win Rate:      {optimal.get('win_rate', 0):.2%}")
        if 'profit_factor' in optimal:
            print(f"  Profit Factor: {optimal.get('profit_factor', 0):.2f}")
        
    except Exception as e:
        print(f"❌ Threshold optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ===== VALIDATION SUMMARY =====
    print_section("STEP 3: VALIDATION SUMMARY")
    
    try:
        summary = validator.generate_validation_summary(
            walk_forward_results,
            threshold_results
        )
        
        print("📋 VALIDATION SUMMARY:")
        print("-" * 80)
        print(f"Validation Date: {summary['validation_date']}")
        print(f"Target Variable: {summary['target']}")
        print()
        
        print("Walk-Forward Results:")
        wf = summary['walk_forward']
        print(f"  Train Period:      {wf['train_period']}")
        print(f"  Test Periods:      {wf['n_test_periods']} quarters")
        print(f"  Overall Accuracy:  {wf['overall_accuracy']:.2%}")
        print(f"  Overall F1-Score:  {wf['overall_f1']:.2%}")
        print(f"  Accuracy Std Dev:  {wf['accuracy_std']:.4f}")
        print()
        
        print("Optimal Threshold:")
        opt = summary['threshold_optimization']
        print(f"  Threshold:      {opt['optimal_threshold']:.2f}")
        print(f"  Win Rate:       {opt['optimal_win_rate']:.2%}")
        print(f"  Profit Factor:  {opt['optimal_profit_factor']:.2f}")
        print(f"  Trades:         {opt['optimal_n_trades']:,}")
        print()
        
        print("=" * 80)
        print(f"🎯 RECOMMENDATION: {summary['recommendation']}")
        print("=" * 80)
        
        # Save summary to file
        output_file = 'validation_summary.json'
        with open(output_file, 'w') as f:
            json.dump({
                'walk_forward_results': walk_forward_results,
                'threshold_results': threshold_results,
                'summary': summary
            }, f, indent=2, default=str)
        
        print(f"\n💾 Full results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Summary generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ===== COMPLETION =====
    print_section("VALIDATION COMPLETE")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n✅ Phase 4 validation completed successfully!")
    print("\nNext Steps:")
    print("1. Review the validation results above")
    print("2. Check validation_summary.json for full details")
    print("3. If recommendation is DEPLOY, proceed to Phase 5")
    print("4. If recommendation is DO NOT DEPLOY, retrain model or collect more data")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
