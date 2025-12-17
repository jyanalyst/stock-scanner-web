"""
Test Script: Regularization Impact Analysis
Compares baseline model vs regularized model to demonstrate overfitting fix
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from ml.model_evaluator import ModelEvaluator
from ml.feature_preprocessor import MLFeaturePreprocessor
from ml.model_trainer import MLModelTrainer
import joblib
import time
import json
from datetime import datetime


def print_section(title):
    """Print section header"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80 + "\n")


def main():
    print_section("REGULARIZATION IMPACT TEST")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load data
    print("📂 Loading data...")
    preprocessor = MLFeaturePreprocessor()
    data = preprocessor.get_preprocessed_data(
        target='win_3d',
        include_technical=True,
        include_fundamental=False,
        include_signal=True,
        normalization='standard',
        split_method='timeseries',
        split_date='2024-01-01'
    )
    print(f"✅ Loaded {len(data['X_train'])} train samples, {len(data['X_test'])} test samples\n")
    
    # ===== STEP 1: Test Baseline Model =====
    print_section("STEP 1: BASELINE MODEL (No Regularization)")
    
    try:
        baseline_model = joblib.load('models/production/best_baseline_classifier_20251214_160847.pkl')
        print("✅ Loaded baseline model\n")
        
        evaluator = ModelEvaluator()
        baseline_results = evaluator.diagnose_overfitting(
            baseline_model,
            data['X_train'], data['y_train'],
            data['X_test'], data['y_test']
        )
        
    except Exception as e:
        print(f"⚠️  Could not load baseline model: {e}")
        print("Skipping baseline comparison...\n")
        baseline_results = None
    
    # ===== STEP 2: Train Regularized Model =====
    print_section("STEP 2: TRAIN REGULARIZED MODEL")
    
    print("🔄 Training with moderate regularization...")
    print("   Parameters:")
    print("   - max_depth: [10, 15, 20] (bounded)")
    print("   - min_samples_split: [10, 20]")
    print("   - min_samples_leaf: [5, 10]")
    print("   - max_samples: [0.8, 0.9]")
    print("   - class_weight: ['balanced', 'balanced_subsample']")
    print()
    
    trainer = MLModelTrainer(random_state=42)
    start_time = time.time()
    
    regularized_model = trainer.train_random_forest_classifier(
        data['X_train'], data['y_train'],
        tune_hyperparameters=True,
        cv_folds=3,  # Faster for testing
        regularization_level='heavy'
    )
    
    training_time = time.time() - start_time
    print(f"\n✅ Training complete in {training_time:.1f}s")
    print(f"   Best params: {trainer.best_params.get('rf_classifier', {})}")
    print(f"   CV F1 Score: {trainer.cv_scores.get('rf_classifier', 0):.4f}\n")
    
    # ===== STEP 3: Test Regularized Model =====
    print_section("STEP 3: REGULARIZED MODEL DIAGNOSIS")
    
    regularized_results = evaluator.diagnose_overfitting(
        regularized_model,
        data['X_train'], data['y_train'],
        data['X_test'], data['y_test']
    )
    
    # ===== STEP 4: Comparison =====
    print_section("STEP 4: BEFORE/AFTER COMPARISON")
    
    if baseline_results:
        print("┌─────────────────────┬──────────┬──────────────┬─────────┐")
        print("│ Metric              │ Baseline │ Regularized  │ Change  │")
        print("├─────────────────────┼──────────┼──────────────┼─────────┤")
        print(f"│ Train Accuracy      │ {baseline_results['train_acc']:>7.2%} │   {regularized_results['train_acc']:>7.2%}    │ {regularized_results['train_acc']-baseline_results['train_acc']:>+6.2%} │")
        print(f"│ Test Accuracy       │ {baseline_results['test_acc']:>7.2%} │   {regularized_results['test_acc']:>7.2%}    │ {regularized_results['test_acc']-baseline_results['test_acc']:>+6.2%} │")
        print(f"│ Accuracy Gap        │ {baseline_results['acc_gap']:>7.2%} │   {regularized_results['acc_gap']:>7.2%}    │ {regularized_results['acc_gap']-baseline_results['acc_gap']:>+6.2%} │")
        print(f"│ Train F1            │ {baseline_results['train_f1']:>7.2%} │   {regularized_results['train_f1']:>7.2%}    │ {regularized_results['train_f1']-baseline_results['train_f1']:>+6.2%} │")
        print(f"│ Test F1             │ {baseline_results['test_f1']:>7.2%} │   {regularized_results['test_f1']:>7.2%}    │ {regularized_results['test_f1']-baseline_results['test_f1']:>+6.2%} │")
        print(f"│ F1 Gap              │ {baseline_results['f1_gap']:>7.2%} │   {regularized_results['f1_gap']:>7.2%}    │ {regularized_results['f1_gap']-baseline_results['f1_gap']:>+6.2%} │")
        print("├─────────────────────┼──────────┼──────────────┼─────────┤")
        
        baseline_status = "🔴 OVERFIT" if baseline_results['is_overfitting'] else "✅ OK"
        regularized_status = "🔴 OVERFIT" if regularized_results['is_overfitting'] else "✅ OK"
        print(f"│ Overfitting Status  │ {baseline_status:>8} │   {regularized_status:>8}   │  {'FIXED!' if baseline_results['is_overfitting'] and not regularized_results['is_overfitting'] else 'N/A':>5}  │")
        print("└─────────────────────┴──────────┴──────────────┴─────────┘")
        
        # Recommendation
        print("\n🎯 RECOMMENDATION:")
        if baseline_results['is_overfitting'] and not regularized_results['is_overfitting']:
            print("   ✅ Deploy regularized model - overfitting eliminated!")
            print(f"   - Accuracy gap reduced from {baseline_results['acc_gap']:.2%} to {regularized_results['acc_gap']:.2%}")
            print(f"   - Test performance {'improved' if regularized_results['test_acc'] > baseline_results['test_acc'] else 'maintained'}")
            recommendation = "DEPLOY_REGULARIZED"
        elif not regularized_results['is_overfitting']:
            print("   ✅ Regularized model generalizes well")
            recommendation = "DEPLOY_REGULARIZED"
        else:
            print("   ⚠️  Model still overfitting - try 'heavy' regularization")
            recommendation = "INCREASE_REGULARIZATION"
    else:
        print("Baseline comparison not available")
        print(f"\nRegularized Model Status: {'✅ OK' if not regularized_results['is_overfitting'] else '🔴 OVERFIT'}")
        recommendation = "DEPLOY_REGULARIZED" if not regularized_results['is_overfitting'] else "INCREASE_REGULARIZATION"
    
    # ===== STEP 5: Save Results =====
    print_section("STEP 5: SAVE RESULTS")
    
    # Save regularized model
    model_path = trainer.save_model(
        regularized_model,
        'regularized_rf_classifier',
        output_dir='models/production',
        metadata={
            'regularization_level': 'moderate',
            'best_params': trainer.best_params.get('rf_classifier', {}),
            'cv_score': trainer.cv_scores.get('rf_classifier', 0),
            'training_time': training_time,
            'test_results': {
                'train_acc': regularized_results['train_acc'],
                'test_acc': regularized_results['test_acc'],
                'acc_gap': regularized_results['acc_gap'],
                'is_overfitting': regularized_results['is_overfitting']
            }
        }
    )
    print(f"✅ Saved regularized model to: {model_path}")
    
    # Save comparison report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"regularization_test_results_{timestamp}.json"
    
    report = {
        'timestamp': timestamp,
        'baseline_results': baseline_results if baseline_results else None,
        'regularized_results': regularized_results,
        'recommendation': recommendation,
        'training_time': training_time,
        'best_params': trainer.best_params.get('rf_classifier', {}),
        'cv_score': trainer.cv_scores.get('rf_classifier', 0)
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"✅ Saved comparison report to: {report_path}")
    
    # ===== COMPLETION =====
    print_section("TEST COMPLETE")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n✅ Regularization test completed successfully!")
    print(f"\nNext Steps:")
    print(f"1. Review the comparison results above")
    print(f"2. Check {report_path} for full details")
    print(f"3. If recommendation is DEPLOY_REGULARIZED, use the new model")
    print(f"4. If recommendation is INCREASE_REGULARIZATION, try 'heavy' level")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
