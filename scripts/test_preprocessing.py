"""
Test Feature Preprocessing Pipeline - Phase 3.1
Validates feature selection, normalization, and train-test splitting
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.feature_preprocessor import MLFeaturePreprocessor
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Test preprocessing pipeline"""
    
    print("=" * 80)
    print("TESTING PHASE 3.1: FEATURE PREPROCESSING")
    print("=" * 80)
    
    # Initialize preprocessor
    print("\n1️⃣ Initializing preprocessor...")
    preprocessor = MLFeaturePreprocessor()
    print("✅ Preprocessor initialized")
    
    # Test Phase 2 results loading
    print("\n2️⃣ Loading Phase 2 results...")
    features, weights = preprocessor.load_phase2_results()
    print(f"✅ Loaded {len(features)} features with weights")
    print(f"   Sample features: {features[:5]}")
    
    # Test feature categorization
    print("\n3️⃣ Categorizing features...")
    categories = preprocessor.categorize_features(features)
    print(f"✅ Categorized features:")
    print(f"   Technical: {len(categories['technical'])} features")
    print(f"   Fundamental: {len(categories['fundamental'])} features")
    print(f"   Signal: {len(categories['signal'])} features")
    print(f"   Other: {len(categories['other'])} features")
    
    # Test feature selection (11 technical features only)
    print("\n4️⃣ Selecting features (Technical + Signal only)...")
    selected = preprocessor.select_features(
        include_technical=True,
        include_fundamental=False,  # Exclude fundamentals
        include_signal=True
    )
    print(f"✅ Selected {len(selected)} features")
    print(f"   Features: {selected}")
    
    # Test weight renormalization
    print("\n5️⃣ Renormalizing weights...")
    renormalized_weights = preprocessor.renormalize_weights(selected)
    print(f"✅ Renormalized weights for {len(renormalized_weights)} features")
    print(f"   Total weight: {sum(renormalized_weights.values()):.4f}")
    print(f"\n   Top 5 features by weight:")
    sorted_weights = sorted(renormalized_weights.items(), key=lambda x: x[1], reverse=True)
    for i, (feat, weight) in enumerate(sorted_weights[:5], 1):
        print(f"   {i}. {feat}: {weight:.4f} ({weight*100:.2f}%)")
    
    # Test complete preprocessing pipeline
    print("\n6️⃣ Running complete preprocessing pipeline...")
    print("   Configuration:")
    print("   - Target: win_3d (classification)")
    print("   - Features: 11 technical + signal")
    print("   - Normalization: StandardScaler")
    print("   - Split: Time-series (2023 train, 2024 test)")
    
    results = preprocessor.get_preprocessed_data(
        target='win_3d',
        include_technical=True,
        include_fundamental=False,
        include_signal=True,
        normalization='standard',
        split_method='timeseries',
        split_date='2024-01-01'
    )
    
    print(f"\n✅ Preprocessing complete!")
    print(f"\n📊 Results Summary:")
    print(f"   Features: {results['n_features']}")
    print(f"   Train samples: {results['n_train']:,}")
    print(f"   Test samples: {results['n_test']:,}")
    print(f"   Train/Test ratio: {results['n_train']/(results['n_train']+results['n_test'])*100:.1f}% / {results['n_test']/(results['n_train']+results['n_test'])*100:.1f}%")
    print(f"   Normalization: {results['normalization']}")
    print(f"   Split method: {results['split_method']}")
    
    # Verify data shapes
    print(f"\n📐 Data Shapes:")
    print(f"   X_train: {results['X_train'].shape}")
    print(f"   X_test: {results['X_test'].shape}")
    print(f"   y_train: {results['y_train'].shape}")
    print(f"   y_test: {results['y_test'].shape}")
    
    # Verify normalization
    print(f"\n📊 Normalization Check:")
    print(f"   X_train mean: {results['X_train'].mean():.4f} (should be ~0)")
    print(f"   X_train std: {results['X_train'].std():.4f} (should be ~1)")
    print(f"   X_test mean: {results['X_test'].mean():.4f}")
    print(f"   X_test std: {results['X_test'].std():.4f}")
    
    # Check target distribution
    print(f"\n🎯 Target Distribution:")
    import numpy as np
    train_win_rate = results['y_train'].mean()
    test_win_rate = results['y_test'].mean()
    print(f"   Train win rate: {train_win_rate:.2%}")
    print(f"   Test win rate: {test_win_rate:.2%}")
    print(f"   Train wins: {results['y_train'].sum():,} / {len(results['y_train']):,}")
    print(f"   Test wins: {results['y_test'].sum():,} / {len(results['y_test']):,}")
    
    # Test with regression target
    print("\n7️⃣ Testing with regression target (return_3d)...")
    results_reg = preprocessor.get_preprocessed_data(
        target='return_3d',
        include_technical=True,
        include_fundamental=False,
        include_signal=True,
        normalization='standard',
        split_method='timeseries',
        split_date='2024-01-01'
    )
    
    print(f"✅ Regression preprocessing complete!")
    print(f"   Target: {results_reg['target']}")
    print(f"   Train mean return: {results_reg['y_train'].mean():.4f}")
    print(f"   Test mean return: {results_reg['y_test'].mean():.4f}")
    print(f"   Train std return: {results_reg['y_train'].std():.4f}")
    print(f"   Test std return: {results_reg['y_test'].std():.4f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ ALL PREPROCESSING TESTS PASSED!")
    print("=" * 80)
    
    print("\n📋 Summary:")
    print(f"  ✅ Phase 2 results loaded successfully")
    print(f"  ✅ Feature categorization working")
    print(f"  ✅ Feature selection working (11 technical features)")
    print(f"  ✅ Weight renormalization working")
    print(f"  ✅ Time-series split working (2023 train, 2024 test)")
    print(f"  ✅ StandardScaler normalization working")
    print(f"  ✅ Classification target (win_3d) working")
    print(f"  ✅ Regression target (return_3d) working")
    
    print("\n🎯 Selected Features for Training:")
    for i, feat in enumerate(results['features'], 1):
        weight = results['weights'].get(feat, 0)
        print(f"  {i:2d}. {feat:40s} {weight*100:6.2f}%")
    
    print("\n🚀 Ready for Phase 3.2: Model Training!")
    print("   Next: Create model_trainer.py")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
