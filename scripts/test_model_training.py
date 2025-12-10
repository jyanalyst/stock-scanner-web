"""
Test Model Training Pipeline - Phase 3.2
End-to-end training and evaluation of all models
"""

import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.feature_preprocessor import MLFeaturePreprocessor
from ml.model_trainer import MLModelTrainer
from ml.model_evaluator import ModelEvaluator
import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Test complete model training pipeline"""
    
    print("=" * 80)
    print("TESTING PHASE 3.2: MODEL TRAINING")
    print("=" * 80)
    
    start_time = time.time()
    
    # Step 1: Load preprocessed data
    print("\n1️⃣ Loading preprocessed data...")
    preprocessor = MLFeaturePreprocessor()
    
    # Get preprocessed data (classification)
    data_class = preprocessor.get_preprocessed_data(
        target='win_3d',
        include_technical=True,
        include_fundamental=False,
        include_signal=True,
        normalization='standard',
        split_method='timeseries',
        split_date='2024-01-01'
    )
    
    # Get preprocessed data (regression)
    data_reg = preprocessor.get_preprocessed_data(
        target='return_3d',
        include_technical=True,
        include_fundamental=False,
        include_signal=True,
        normalization='standard',
        split_method='timeseries',
        split_date='2024-01-01'
    )
    
    print(f"✅ Data loaded:")
    print(f"   Features: {data_class['n_features']}")
    print(f"   Train samples: {data_class['n_train']:,}")
    print(f"   Test samples: {data_class['n_test']:,}")
    
    # Step 2: Initialize trainer and evaluator
    print("\n2️⃣ Initializing trainer and evaluator...")
    trainer = MLModelTrainer(random_state=42)
    evaluator = ModelEvaluator()
    print("✅ Initialized")
    
    # Storage for models and metrics
    classification_models = {}
    classification_metrics = {}
    regression_models = {}
    regression_metrics = {}
    feature_importance = {}
    
    # Step 3: Train Classification Models
    print("\n" + "=" * 80)
    print("3️⃣ TRAINING CLASSIFICATION MODELS")
    print("=" * 80)
    
    # 3a. Baseline Classifier
    print("\n📊 Training Baseline Classifier...")
    baseline_clf = trainer.create_baseline_classifier(weights=data_class['weights'])
    baseline_clf.fit(data_class['X_train'], data_class['y_train'])
    
    y_pred_baseline = baseline_clf.predict(data_class['X_test'])
    y_pred_proba_baseline = baseline_clf.predict_proba(data_class['X_test'])
    
    metrics_baseline = evaluator.evaluate_classification(
        data_class['y_test'], y_pred_baseline, y_pred_proba_baseline,
        model_name='Baseline'
    )
    
    classification_models['Baseline'] = baseline_clf
    classification_metrics['Baseline'] = metrics_baseline
    
    print(f"✅ Baseline - Accuracy: {metrics_baseline['accuracy']:.4f}, F1: {metrics_baseline['f1']:.4f}")
    
    # 3b. Logistic Regression
    print("\n📊 Training Logistic Regression...")
    log_reg = trainer.train_logistic_regression(
        data_class['X_train'], data_class['y_train']
    )
    
    y_pred_logreg = log_reg.predict(data_class['X_test'])
    y_pred_proba_logreg = log_reg.predict_proba(data_class['X_test'])
    
    metrics_logreg = evaluator.evaluate_classification(
        data_class['y_test'], y_pred_logreg, y_pred_proba_logreg,
        model_name='LogisticRegression'
    )
    
    classification_models['LogisticRegression'] = log_reg
    classification_metrics['LogisticRegression'] = metrics_logreg
    feature_importance['LogisticRegression'] = trainer.get_feature_importance(log_reg, data_class['features'])
    
    print(f"✅ LogReg - Accuracy: {metrics_logreg['accuracy']:.4f}, F1: {metrics_logreg['f1']:.4f}")
    
    # 3c. Random Forest Classifier
    print("\n📊 Training Random Forest Classifier (with GridSearch)...")
    print("   This may take 2-3 minutes...")
    rf_clf = trainer.train_random_forest_classifier(
        data_class['X_train'], data_class['y_train'],
        tune_hyperparameters=True, cv_folds=5
    )
    
    y_pred_rf = rf_clf.predict(data_class['X_test'])
    y_pred_proba_rf = rf_clf.predict_proba(data_class['X_test'])
    
    metrics_rf = evaluator.evaluate_classification(
        data_class['y_test'], y_pred_rf, y_pred_proba_rf,
        model_name='RandomForest'
    )
    
    classification_models['RandomForest'] = rf_clf
    classification_metrics['RandomForest'] = metrics_rf
    feature_importance['RandomForest'] = trainer.get_feature_importance(rf_clf, data_class['features'])
    
    print(f"✅ RF - Accuracy: {metrics_rf['accuracy']:.4f}, F1: {metrics_rf['f1']:.4f}")
    print(f"   Best params: {trainer.best_params.get('rf_classifier', {})}")
    
    # 3d. XGBoost Classifier
    print("\n📊 Training XGBoost Classifier (with GridSearch)...")
    print("   This may take 3-4 minutes...")
    xgb_clf = trainer.train_xgboost_classifier(
        data_class['X_train'], data_class['y_train'],
        tune_hyperparameters=True, cv_folds=5
    )
    
    if xgb_clf is not None:
        y_pred_xgb = xgb_clf.predict(data_class['X_test'])
        y_pred_proba_xgb = xgb_clf.predict_proba(data_class['X_test'])
        
        metrics_xgb = evaluator.evaluate_classification(
            data_class['y_test'], y_pred_xgb, y_pred_proba_xgb,
            model_name='XGBoost'
        )
        
        classification_models['XGBoost'] = xgb_clf
        classification_metrics['XGBoost'] = metrics_xgb
        feature_importance['XGBoost'] = trainer.get_feature_importance(xgb_clf, data_class['features'])
        
        print(f"✅ XGB - Accuracy: {metrics_xgb['accuracy']:.4f}, F1: {metrics_xgb['f1']:.4f}")
        print(f"   Best params: {trainer.best_params.get('xgb_classifier', {})}")
    else:
        print("⚠️ XGBoost not available")
    
    # Step 4: Train Regression Models
    print("\n" + "=" * 80)
    print("4️⃣ TRAINING REGRESSION MODELS")
    print("=" * 80)
    
    # 4a. Baseline Regressor
    print("\n📊 Training Baseline Regressor...")
    baseline_reg = trainer.create_baseline_regressor(weights=data_reg['weights'])
    baseline_reg.fit(data_reg['X_train'], data_reg['y_train'])
    
    y_pred_baseline_reg = baseline_reg.predict(data_reg['X_test'])
    
    metrics_baseline_reg = evaluator.evaluate_regression(
        data_reg['y_test'], y_pred_baseline_reg,
        model_name='Baseline'
    )
    
    regression_models['Baseline'] = baseline_reg
    regression_metrics['Baseline'] = metrics_baseline_reg
    
    print(f"✅ Baseline - MAE: {metrics_baseline_reg['mae']:.6f}, IC: {metrics_baseline_reg['ic']:.4f}")
    
    # 4b. Random Forest Regressor
    print("\n📊 Training Random Forest Regressor (with GridSearch)...")
    print("   This may take 2-3 minutes...")
    rf_reg = trainer.train_random_forest_regressor(
        data_reg['X_train'], data_reg['y_train'],
        tune_hyperparameters=True, cv_folds=5
    )
    
    y_pred_rf_reg = rf_reg.predict(data_reg['X_test'])
    
    metrics_rf_reg = evaluator.evaluate_regression(
        data_reg['y_test'], y_pred_rf_reg,
        model_name='RandomForest'
    )
    
    regression_models['RandomForest'] = rf_reg
    regression_metrics['RandomForest'] = metrics_rf_reg
    
    print(f"✅ RF - MAE: {metrics_rf_reg['mae']:.6f}, IC: {metrics_rf_reg['ic']:.4f}")
    
    # 4c. XGBoost Regressor
    print("\n📊 Training XGBoost Regressor (with GridSearch)...")
    print("   This may take 3-4 minutes...")
    xgb_reg = trainer.train_xgboost_regressor(
        data_reg['X_train'], data_reg['y_train'],
        tune_hyperparameters=True, cv_folds=5
    )
    
    if xgb_reg is not None:
        y_pred_xgb_reg = xgb_reg.predict(data_reg['X_test'])
        
        metrics_xgb_reg = evaluator.evaluate_regression(
            data_reg['y_test'], y_pred_xgb_reg,
            model_name='XGBoost'
        )
        
        regression_models['XGBoost'] = xgb_reg
        regression_metrics['XGBoost'] = metrics_xgb_reg
        
        print(f"✅ XGB - MAE: {metrics_xgb_reg['mae']:.6f}, IC: {metrics_xgb_reg['ic']:.4f}")
    
    # Step 5: Calculate Profit Metrics
    print("\n" + "=" * 80)
    print("5️⃣ CALCULATING PROFIT METRICS")
    print("=" * 80)
    
    # Load actual returns for profit calculation
    df_full = preprocessor.df
    df_test = df_full[df_full['entry_date'] >= '2024-01-01']
    y_test_returns = df_test['return_3d'].values
    
    profit_metrics = {}
    
    for model_name, model in classification_models.items():
        y_pred = model.predict(data_class['X_test'])
        profit = evaluator.calculate_profit_metrics(
            y_test_returns, y_pred, model_name=model_name
        )
        profit_metrics[model_name] = profit
    
    print("✅ Profit metrics calculated for all models")
    
    # Step 6: Compare Models
    print("\n" + "=" * 80)
    print("6️⃣ MODEL COMPARISON")
    print("=" * 80)
    
    # Classification comparison
    print("\n📊 Classification Models Comparison:")
    comparison_clf = evaluator.compare_models(classification_metrics, metric_type='classification')
    print(comparison_clf.to_string())
    
    # Regression comparison
    print("\n📈 Regression Models Comparison:")
    comparison_reg = evaluator.compare_models(regression_metrics, metric_type='regression')
    print(comparison_reg.to_string())
    
    # Profit comparison
    print("\n💰 Profit Metrics Comparison:")
    profit_df = pd.DataFrame(profit_metrics).T
    print(profit_df[['n_trades', 'win_rate', 'total_return', 'profit_factor', 'sharpe_ratio']].to_string())
    
    # Step 7: Feature Importance
    print("\n" + "=" * 80)
    print("7️⃣ FEATURE IMPORTANCE")
    print("=" * 80)
    
    if feature_importance:
        importance_df = evaluator.get_feature_importance_comparison(feature_importance)
        print("\n🔍 Top 10 Features (averaged across models):")
        print(importance_df.head(10).to_string())
    
    # Step 8: Find Best Models
    print("\n" + "=" * 80)
    print("8️⃣ BEST MODELS")
    print("=" * 80)
    
    best_clf_name, best_clf_metrics = evaluator.get_best_model(
        classification_metrics, metric='f1', higher_is_better=True
    )
    print(f"\n🏆 Best Classification Model: {best_clf_name}")
    print(f"   F1-Score: {best_clf_metrics['f1']:.4f}")
    print(f"   Accuracy: {best_clf_metrics['accuracy']:.4f}")
    print(f"   ROC-AUC: {best_clf_metrics.get('roc_auc', 0):.4f}")
    
    best_reg_name, best_reg_metrics = evaluator.get_best_model(
        regression_metrics, metric='ic', higher_is_better=True
    )
    print(f"\n🏆 Best Regression Model: {best_reg_name}")
    print(f"   IC: {best_reg_metrics['ic']:.4f}")
    print(f"   MAE: {best_reg_metrics['mae']:.6f}")
    print(f"   R²: {best_reg_metrics['r2']:.4f}")
    
    # Step 9: Save Best Model
    print("\n" + "=" * 80)
    print("9️⃣ SAVING BEST MODEL")
    print("=" * 80)
    
    best_model = classification_models[best_clf_name]
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        """Convert numpy types to Python types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    metadata = {
        'model_type': best_clf_name,
        'task': 'classification',
        'target': 'win_3d',
        'features': data_class['features'],
        'n_features': int(data_class['n_features']),
        'train_samples': int(data_class['n_train']),
        'test_samples': int(data_class['n_test']),
        'metrics': convert_to_serializable(best_clf_metrics),
        'best_params': trainer.best_params.get(f"{best_clf_name.lower().replace(' ', '_')}_classifier", {}),
        'normalization': 'StandardScaler',
        'split_method': 'timeseries',
        'split_date': '2024-01-01'
    }
    
    model_path = trainer.save_model(
        best_model, 
        f"best_{best_clf_name.lower()}_classifier",
        output_dir="models/production",
        metadata=metadata
    )
    
    # Save scaler
    import joblib
    scaler_path = "models/production/scaler.pkl"
    joblib.dump(data_class['scaler'], scaler_path)
    print(f"✅ Saved scaler to {scaler_path}")
    
    print(f"✅ Best model saved to {model_path}")
    
    # Step 10: Generate Summary Report
    print("\n" + "=" * 80)
    print("🔟 SUMMARY REPORT")
    print("=" * 80)
    
    summary = evaluator.generate_summary_report()
    print(summary)
    
    # Save report to file
    report_path = "models/production/training_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"\n✅ Report saved to {report_path}")
    
    # Final summary
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("✅ MODEL TRAINING COMPLETE!")
    print("=" * 80)
    
    print(f"\n⏱️ Total Time: {elapsed_time/60:.1f} minutes")
    print(f"\n📊 Models Trained:")
    print(f"   Classification: {len(classification_models)}")
    print(f"   Regression: {len(regression_models)}")
    
    print(f"\n🏆 Best Classification Model: {best_clf_name} (F1={best_clf_metrics['f1']:.4f})")
    print(f"🏆 Best Regression Model: {best_reg_name} (IC={best_reg_metrics['ic']:.4f})")
    
    print(f"\n💾 Saved Files:")
    print(f"   Model: {model_path}")
    print(f"   Scaler: {scaler_path}")
    print(f"   Report: {report_path}")
    
    print("\n🚀 Ready for Phase 3.3: UI Integration!")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
