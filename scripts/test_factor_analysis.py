"""
Test Script for Phase 2: Factor Analysis
Tests the complete factor analysis pipeline
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.factor_analyzer import MLFactorAnalyzer
from ml.visualizations import MLVisualizer
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_factor_analysis():
    """Test complete factor analysis pipeline"""
    
    print("=" * 80)
    print("TESTING PHASE 2: FACTOR ANALYSIS")
    print("=" * 80)
    
    try:
        # Step 1: Initialize analyzer
        print("\n1️⃣ Initializing Factor Analyzer...")
        analyzer = MLFactorAnalyzer(
            data_path="data/ml_training/raw/training_data_complete.parquet",
            target="return_3d"
        )
        print("✅ Analyzer initialized")
        
        # Step 2: Load data
        print("\n2️⃣ Loading training data...")
        df = analyzer.load_data()
        print(f"✅ Loaded {len(df):,} samples")
        print(f"✅ Identified {len(analyzer.features)} features")
        print(f"\nSample features: {analyzer.features[:10]}")
        
        # Check for the features user mentioned
        key_features = ['VW_Range_Velocity', 'VW_Range_Percentile', 'Flow_Velocity_Percentile']
        for feature in key_features:
            if feature in analyzer.features:
                print(f"✅ Found {feature}")
            else:
                print(f"❌ Missing {feature}")
        
        # Step 3: Calculate IC
        print("\n3️⃣ Calculating Information Coefficients...")
        ic_results = analyzer.calculate_information_coefficient(rolling_window=60)
        print(f"✅ Calculated IC for {len(ic_results)} features")
        
        # Show top 10
        print("\n📊 Top 10 Features by |IC|:")
        top_10 = ic_results.head(10)
        for i, row in enumerate(top_10.itertuples(), 1):
            print(f"  {i}. {row.feature}: IC={row.IC_mean:.4f}, |IC|={row.abs_IC:.4f}, p={row.p_value:.4f}")
        
        # Step 4: Analyze correlations
        print("\n4️⃣ Analyzing feature correlations...")
        corr_matrix, redundant_pairs = analyzer.analyze_correlations(threshold=0.85)
        print(f"✅ Correlation matrix: {corr_matrix.shape}")
        print(f"✅ Found {len(redundant_pairs)} redundant pairs")
        
        if redundant_pairs:
            print("\n🔗 Sample redundant pairs:")
            for pair in redundant_pairs[:5]:
                print(f"  - {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}")
        
        # Step 5: Select features
        print("\n5️⃣ Selecting optimal features...")
        selected_features = analyzer.select_features(ic_threshold=0.01, correlation_threshold=0.85)
        print(f"✅ Selected {len(selected_features)} features")
        print(f"✅ Reduction: {len(analyzer.features)} → {len(selected_features)} ({len(selected_features)/len(analyzer.features)*100:.1f}%)")
        
        # Step 6: Calculate optimal weights
        print("\n6️⃣ Calculating optimal weights...")
        optimal_weights = analyzer.calculate_optimal_weights(method='ic_squared')
        print(f"✅ Calculated weights for {len(optimal_weights)} features")
        
        # Show top 10 weights
        print("\n⚖️ Top 10 Feature Weights:")
        sorted_weights = sorted(optimal_weights.items(), key=lambda x: x[1], reverse=True)[:10]
        for i, (feature, weight) in enumerate(sorted_weights, 1):
            print(f"  {i}. {feature}: {weight:.6f} ({weight*100:.2f}%)")
        
        # Step 7: Test visualizations
        print("\n7️⃣ Testing visualizations...")
        visualizer = MLVisualizer()
        
        figures = visualizer.create_summary_dashboard(
            ic_results=ic_results,
            correlation_matrix=corr_matrix,
            optimal_weights=optimal_weights,
            redundant_pairs=redundant_pairs
        )
        
        print(f"✅ Created {len(figures)} visualizations:")
        for fig_name in figures.keys():
            print(f"  - {fig_name}")
        
        # Step 8: Generate report
        print("\n8️⃣ Generating report...")
        report_path = analyzer.generate_report()
        print(f"✅ Report saved to: {report_path}")
        
        # Final summary
        print("\n" + "=" * 80)
        print("✅ FACTOR ANALYSIS TEST COMPLETE!")
        print("=" * 80)
        
        print("\n📊 SUMMARY:")
        print(f"  • Training samples: {len(df):,}")
        print(f"  • Total features analyzed: {len(analyzer.features)}")
        print(f"  • Features with |IC| > 0.05: {(ic_results['abs_IC'] > 0.05).sum()}")
        print(f"  • Features with |IC| > 0.10: {(ic_results['abs_IC'] > 0.10).sum()}")
        print(f"  • Selected features: {len(selected_features)}")
        print(f"  • Redundant pairs removed: {len(redundant_pairs)}")
        print(f"  • Report location: {report_path}")
        
        print("\n🎯 KEY FEATURES STATUS:")
        for feature in key_features:
            if feature in selected_features:
                ic_value = ic_results[ic_results['feature'] == feature]['IC_mean'].values[0]
                weight = optimal_weights.get(feature, 0)
                print(f"  ✅ {feature}: IC={ic_value:.4f}, Weight={weight:.6f}")
            elif feature in analyzer.features:
                ic_value = ic_results[ic_results['feature'] == feature]['IC_mean'].values[0]
                print(f"  ⚠️ {feature}: IC={ic_value:.4f} (removed - below threshold)")
            else:
                print(f"  ❌ {feature}: Not found in data")
        
        print("\n✅ All tests passed! Phase 2 is ready to use in ML Lab.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_factor_analysis()
    sys.exit(0 if success else 1)
