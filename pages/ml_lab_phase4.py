"""
ML Lab Phase 4: Model Validation UI
Interactive validation dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime


def show_phase4():
    """Display Phase 4: Model Validation UI"""
    
    with st.expander("📊 PHASE 4: Model Validation", expanded=False):
        st.markdown("""
        **Goal:** Validate model performance and determine deployment readiness
        
        **Prerequisites:** 
        - ✅ Phase 1 complete (training data collected)
        - ✅ Phase 2 complete (features selected)
        - ✅ Phase 3 complete (model trained)
        
        **Process:**
        - Walk-forward validation on 2024 quarters
        - Threshold optimization (0.55, 0.60, 0.65, 0.70)
        - Performance stability analysis
        - Deployment recommendation
        """)
        
        # Check prerequisites
        phase1_complete = os.path.exists("data/ml_training/raw/training_data_complete.parquet")
        phase2_complete = os.path.exists("data/ml_training/analysis/optimal_weights.json")
        phase3_complete = os.path.exists("models/production") and len([f for f in os.listdir("models/production") if f.endswith('.pkl') and not f.startswith('scaler')]) > 0
        
        if not all([phase1_complete, phase2_complete, phase3_complete]):
            st.warning("⚠️ Prerequisites not met. Complete Phase 1, 2, and 3 first.")
            
            col_check1, col_check2, col_check3 = st.columns(3)
            with col_check1:
                st.metric("Phase 1", "✅" if phase1_complete else "❌")
            with col_check2:
                st.metric("Phase 2", "✅" if phase2_complete else "❌")
            with col_check3:
                st.metric("Phase 3", "✅" if phase3_complete else "❌")
            
            return
        
        # Create tabs for different Phase 4 functions
        tab1, tab2, tab3 = st.tabs([
            "⚙️ Configuration & Run",
            "📊 Validation Results", 
            "📥 Export & Reports"
        ])
        
        with tab1:
            show_validation_config()
        
        with tab2:
            show_validation_results()
        
        with tab3:
            show_export_section()


def show_validation_config():
    """Configuration and Run Tab"""
    st.markdown("### ⚙️ Validation Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Training Period**")
        train_start = st.date_input(
            "Train Start",
            value=datetime(2023, 1, 1),
            help="Start date for training data"
        )
        train_end = st.date_input(
            "Train End",
            value=datetime(2023, 12, 31),
            help="End date for training data"
        )
    
    with col2:
        st.markdown("**Test Periods (2024)**")
        test_quarters = st.multiselect(
            "Select Quarters",
            options=['Q1', 'Q2', 'Q3', 'Q4'],
            default=['Q1', 'Q2', 'Q3', 'Q4'],
            help="Which 2024 quarters to test on"
        )
        
        confidence_thresholds = st.multiselect(
            "Confidence Thresholds",
            options=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
            default=[0.55, 0.60, 0.65, 0.70],
            help="Thresholds to test for optimal performance"
        )
    
    # Map quarters to date ranges
    quarter_map = {
        'Q1': ('2024-01-01', '2024-03-31'),
        'Q2': ('2024-04-01', '2024-06-30'),
        'Q3': ('2024-07-01', '2024-09-30'),
        'Q4': ('2024-10-01', '2024-12-31')
    }
    
    test_periods = [quarter_map[q] for q in test_quarters]
    
    st.markdown("---")
    st.markdown("### 🚀 Run Validation")
    
    st.info(f"""
    **Configuration Summary:**
    - Training: {train_start} to {train_end}
    - Testing: {len(test_periods)} quarters in 2024
    - Thresholds: {len(confidence_thresholds)} values to test
    - Estimated time: 2-3 minutes
    """)
    
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        if st.button("🔬 Run Walk-Forward Validation", type="primary"):
            run_validation(
                train_start.strftime('%Y-%m-%d'),
                train_end.strftime('%Y-%m-%d'),
                test_periods,
                confidence_thresholds
            )
    
    with col_btn2:
        if st.button("📂 Load Previous Results"):
            load_previous_results()


def run_validation(train_start, train_end, test_periods, confidence_thresholds):
    """Run validation process"""
    from ml.validator import MLValidator
    
    with st.spinner("🔬 Running walk-forward validation... This will take 2-3 minutes."):
        try:
            # Initialize validator
            validator = MLValidator(target='win_3d')
            
            # Run walk-forward validation
            st.info("Step 1/2: Walk-forward validation...")
            walk_forward_results = validator.walk_forward_validation(
                train_start=train_start,
                train_end=train_end,
                test_periods=test_periods
            )
            
            # Run threshold optimization
            st.info("Step 2/2: Threshold optimization...")
            threshold_results = validator.optimize_threshold(
                confidence_thresholds=confidence_thresholds
            )
            
            # Generate summary
            summary = validator.generate_validation_summary(
                walk_forward_results,
                threshold_results
            )
            
            # Store in session state
            st.session_state.validation_complete = True
            st.session_state.walk_forward_results = walk_forward_results
            st.session_state.threshold_results = threshold_results
            st.session_state.validation_summary = summary
            st.session_state.validation_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Save to file
            results_data = {
                'walk_forward_results': walk_forward_results,
                'threshold_results': threshold_results,
                'summary': summary
            }
            
            with open('validation_summary.json', 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            
            st.success("✅ Validation complete!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Validation failed: {e}")
            import traceback
            st.code(traceback.format_exc())


def load_previous_results():
    """Load previously saved validation results"""
    try:
        if os.path.exists('validation_summary.json'):
            with open('validation_summary.json', 'r') as f:
                results_data = json.load(f)
            
            st.session_state.validation_complete = True
            st.session_state.walk_forward_results = results_data['walk_forward_results']
            st.session_state.threshold_results = results_data['threshold_results']
            st.session_state.validation_summary = results_data['summary']
            st.session_state.validation_timestamp = "Loaded from file"
            
            st.success("✅ Previous results loaded!")
            st.rerun()
        else:
            st.warning("No previous results found. Run validation first.")
    except Exception as e:
        st.error(f"❌ Error loading results: {e}")


def show_validation_results():
    """Validation Results Tab"""
    st.markdown("### 📊 Validation Results")
    
    if not st.session_state.get('validation_complete', False):
        st.info("👈 Run validation in the Configuration tab to see results")
        return
    
    # Get results from session state
    walk_forward = st.session_state.walk_forward_results
    threshold = st.session_state.threshold_results
    summary = st.session_state.validation_summary
    
    # Show timestamp
    st.caption(f"Validation completed: {st.session_state.validation_timestamp}")
    
    # Summary metrics
    st.markdown("#### 📈 Overall Performance")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        accuracy = summary['walk_forward']['overall_accuracy']
        st.metric("Accuracy", f"{accuracy:.2%}", 
                 delta="✅" if accuracy >= 0.52 else "❌")
    
    with col2:
        stability = summary['walk_forward']['accuracy_std']
        st.metric("Stability", f"{stability:.4f}",
                 delta="✅" if stability < 0.05 else "❌",
                 help="Lower is better (std dev)")
    
    with col3:
        win_rate = summary['threshold_optimization']['optimal_win_rate']
        st.metric("Win Rate", f"{win_rate:.2%}",
                 delta="✅" if win_rate >= 0.50 else "❌")
    
    with col4:
        profit_factor = summary['threshold_optimization']['optimal_profit_factor']
        st.metric("Profit Factor", f"{profit_factor:.2f}",
                 delta="✅" if profit_factor >= 1.5 else "❌")
    
    with col5:
        optimal_threshold = summary['threshold_optimization']['optimal_threshold']
        st.metric("Optimal Threshold", f"{optimal_threshold:.2f}")
    
    # Deployment recommendation
    st.markdown("---")
    recommendation = summary['recommendation']
    
    if "DEPLOY" in recommendation and "NOT" not in recommendation:
        st.success(f"### 🎯 {recommendation}")
    elif "CAUTION" in recommendation:
        st.warning(f"### ⚠️ {recommendation}")
    else:
        st.error(f"### ❌ {recommendation}")
    
    st.markdown("---")
    
    # Walk-Forward Results
    st.markdown("#### 📊 Walk-Forward Validation Results")
    
    st.info(f"""
    **Training Period:** {walk_forward['train_period']}  
    **Test Periods:** {len(walk_forward['test_periods'])} quarters  
    **Total Test Samples:** {walk_forward.get('n_total_samples', 0):,}
    """)
    
    # Quarter-by-quarter table
    if walk_forward['period_metrics']:
        periods_data = []
        for period in walk_forward['period_metrics']:
            metrics = period['metrics']
            periods_data.append({
                'Period': period['period'],
                'Samples': period['n_samples'],
                'Accuracy': f"{metrics.get('accuracy', 0):.2%}",
                'Precision': f"{metrics.get('precision', 0):.2%}",
                'Recall': f"{metrics.get('recall', 0):.2%}",
                'F1-Score': f"{metrics.get('f1', 0):.2%}",
                'Win Rate': f"{metrics.get('win_rate', 0):.2%}" if 'win_rate' in metrics else "N/A"
            })
        
        periods_df = pd.DataFrame(periods_data)
        st.dataframe(periods_df, width="stretch", hide_index=True)
        
        # Performance chart
        st.markdown("##### Performance by Quarter")
        
        accuracies = [float(p['Accuracy'].strip('%'))/100 for p in periods_data]
        quarters = [p['Period'].split(' to ')[0] for p in periods_data]
        
        chart_data = pd.DataFrame({
            'Quarter': quarters,
            'Accuracy': accuracies
        })
        
        st.bar_chart(chart_data.set_index('Quarter'))
    
    st.markdown("---")
    
    # Threshold Optimization Results
    st.markdown("#### 🎯 Threshold Optimization Results")
    
    if threshold['threshold_results']:
        threshold_data = []
        for result in threshold['threshold_results']:
            threshold_data.append({
                'Threshold': f"{result['threshold']:.2f}",
                'Trades': f"{result['n_trades']:,}",
                'Trade Rate': f"{result['trade_rate']:.1%}",
                'Win Rate': f"{result.get('win_rate', 0):.2%}",
                'Mean Return': f"{result.get('mean_return', 0):.2%}" if 'mean_return' in result else "N/A",
                'Profit Factor': f"{result.get('profit_factor', 0):.2f}" if 'profit_factor' in result else "N/A"
            })
        
        threshold_df = pd.DataFrame(threshold_data)
        
        # Highlight optimal threshold
        optimal_idx = threshold_df[threshold_df['Threshold'] == f"{optimal_threshold:.2f}"].index
        
        st.dataframe(
            threshold_df,
            width="stretch",
            hide_index=True
        )
        
        if len(optimal_idx) > 0:
            st.success(f"⭐ **Optimal Threshold: {optimal_threshold:.2f}** - Best profit factor and win rate balance")
    
    # Overall metrics
    st.markdown("---")
    st.markdown("#### 📋 Detailed Metrics")
    
    overall = walk_forward['overall_metrics']
    
    col_m1, col_m2, col_m3 = st.columns(3)
    
    with col_m1:
        st.markdown("**Classification Metrics**")
        st.write(f"- Accuracy: {overall.get('accuracy', 0):.2%}")
        st.write(f"- Precision: {overall.get('precision', 0):.2%}")
        st.write(f"- Recall: {overall.get('recall', 0):.2%}")
        st.write(f"- F1-Score: {overall.get('f1', 0):.2%}")
        st.write(f"- ROC-AUC: {overall.get('roc_auc', 0):.4f}")
    
    with col_m2:
        st.markdown("**Stability Metrics**")
        period_accs = summary['walk_forward']['period_accuracies']
        st.write(f"- Mean Accuracy: {np.mean(period_accs):.2%}")
        st.write(f"- Std Dev: {np.std(period_accs):.4f}")
        st.write(f"- Min Accuracy: {min(period_accs):.2%}")
        st.write(f"- Max Accuracy: {max(period_accs):.2%}")
        st.write(f"- Range: {(max(period_accs) - min(period_accs)):.2%}")
    
    with col_m3:
        st.markdown("**Trading Metrics**")
        optimal_result = threshold['optimal_threshold']
        st.write(f"- Optimal Threshold: {optimal_result['threshold']:.2f}")
        st.write(f"- Expected Trades: {optimal_result['n_trades']:,}")
        st.write(f"- Win Rate: {optimal_result.get('win_rate', 0):.2%}")
        st.write(f"- Profit Factor: {optimal_result.get('profit_factor', 0):.2f}")
        if 'mean_return' in optimal_result:
            st.write(f"- Mean Return: {optimal_result['mean_return']:.2%}")


def show_export_section():
    """Export and Reports Tab"""
    st.markdown("### 📥 Export & Reports")
    
    if not st.session_state.get('validation_complete', False):
        st.info("👈 Run validation first to export results")
        return
    
    st.markdown("#### 📄 Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Download validation summary JSON
        if os.path.exists('validation_summary.json'):
            with open('validation_summary.json', 'r') as f:
                json_data = f.read()
            
            st.download_button(
                label="📊 Download Summary (JSON)",
                data=json_data,
                file_name="validation_summary.json",
                mime="application/json"
            )
    
    with col2:
        # Download walk-forward results as CSV
        walk_forward = st.session_state.walk_forward_results
        if walk_forward['period_metrics']:
            periods_data = []
            for period in walk_forward['period_metrics']:
                metrics = period['metrics']
                periods_data.append({
                    'period': period['period'],
                    'n_samples': period['n_samples'],
                    **metrics
                })
            
            periods_df = pd.DataFrame(periods_data)
            csv = periods_df.to_csv(index=False)
            
            st.download_button(
                label="📈 Download Walk-Forward (CSV)",
                data=csv,
                file_name="walk_forward_results.csv",
                mime="text/csv"
            )
    
    with col3:
        # Download threshold results as CSV
        threshold = st.session_state.threshold_results
        if threshold['threshold_results']:
            threshold_df = pd.DataFrame(threshold['threshold_results'])
            csv = threshold_df.to_csv(index=False)
            
            st.download_button(
                label="🎯 Download Thresholds (CSV)",
                data=csv,
                file_name="threshold_optimization.csv",
                mime="text/csv"
            )
    
    st.markdown("---")
    st.markdown("#### 📋 Validation Report")
    
    # Show documentation link
    if os.path.exists('docs/PHASE_4_VALIDATION_COMPLETE.md'):
        st.success("✅ Detailed validation report available")
        st.info("📄 See `docs/PHASE_4_VALIDATION_COMPLETE.md` for full analysis and recommendations")
    
    # Show file locations
    st.markdown("#### 📁 Saved Files")
    st.code("""
validation_summary.json          - Full validation results
docs/PHASE_4_VALIDATION_COMPLETE.md - Detailed report
ml/validator.py                  - Validation engine
scripts/test_validation.py       - Command-line script
    """)
    
    st.markdown("---")
    st.markdown("#### 🚀 Next Steps")
    
    summary = st.session_state.validation_summary
    recommendation = summary['recommendation']
    
    if "DEPLOY" in recommendation and "NOT" not in recommendation:
        st.success("""
        **✅ Model is ready for deployment!**
        
        Proceed to Phase 5 to:
        1. Integrate ML predictions with live scanner
        2. Add confidence scores to UI
        3. Display BUY signals
        4. Set up performance monitoring
        """)
    elif "CAUTION" in recommendation:
        st.warning("""
        **⚠️ Model shows edge but needs monitoring**
        
        You can deploy with caution:
        1. Start with paper trading
        2. Monitor performance closely
        3. Set strict risk limits
        4. Consider retraining if performance drops
        """)
    else:
        st.error("""
        **❌ Model not ready for deployment**
        
        Recommended actions:
        1. Collect more training data (Phase 1)
        2. Try different features (Phase 2)
        3. Tune model hyperparameters (Phase 3)
        4. Re-run validation
        """)
