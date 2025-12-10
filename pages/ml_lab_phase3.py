"""
ML Lab Phase 3: Model Training & Predictions UI
Separate module for Phase 3 functionality
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os


def show_phase3():
    """Display Phase 3: Model Training & Predictions UI"""
    
    with st.expander("🧠 PHASE 3: Model Training & Predictions", expanded=False):
        st.markdown("""
        **Goal:** Train ML models and generate predictions
        
        **Prerequisites:** 
        - ✅ Phase 1 complete (training data collected)
        - ✅ Phase 2 complete (features selected)
        - ✅ Models trained (Random Forest saved to production)
        """)
        
        # Check prerequisites
        phase1_complete = os.path.exists("data/ml_training/raw/training_data_complete.parquet")
        phase2_complete = os.path.exists("data/ml_training/analysis/optimal_weights.json")
        models_available = os.path.exists("models/production") and len([f for f in os.listdir("models/production") if f.endswith('.pkl') and not f.startswith('scaler')]) > 0
        
        if not all([phase1_complete, phase2_complete, models_available]):
            st.warning("⚠️ Prerequisites not met. Complete Phase 1 and Phase 2 first, then train models.")
            
            col_check1, col_check2, col_check3 = st.columns(3)
            with col_check1:
                st.metric("Phase 1", "✅" if phase1_complete else "❌")
            with col_check2:
                st.metric("Phase 2", "✅" if phase2_complete else "❌")
            with col_check3:
                st.metric("Models", "✅" if models_available else "❌")
            
            if not models_available:
                st.info("💡 **Tip:** Run `python scripts/test_model_training.py` to train models")
            
            return
        
        # Create tabs for different Phase 3 functions
        tab1, tab2, tab3 = st.tabs([
            "📊 Model Selection",
            "🔮 Live Predictions", 
            "📈 Performance Dashboard"
        ])
        
        with tab1:
            show_model_selection()
        
        with tab2:
            show_live_predictions()
        
        with tab3:
            show_performance_dashboard()


def show_model_selection():
    """Model Selection Tab"""
    st.markdown("### 📊 Model Selection")
    st.markdown("Select and load a trained model for predictions")
    
    try:
        from ml.model_loader import MLModelLoader
        
        # Initialize loader
        if 'model_loader' not in st.session_state:
            st.session_state.model_loader = MLModelLoader()
        
        loader = st.session_state.model_loader
        
        # List available models
        models = loader.list_available_models()
        
        if not models:
            st.warning("No models found in production folder")
            st.info("Run `python scripts/test_model_training.py` to train models")
            return
        
        st.success(f"✅ Found {len(models)} trained model(s)")
        
        # Model selector
        model_options = [f"{m['filename']} ({m.get('model_type', 'Unknown')}, {m['modified']:%Y-%m-%d})" 
                        for m in models]
        selected_idx = st.selectbox(
            "Select Model",
            range(len(models)),
            format_func=lambda i: model_options[i]
        )
        
        selected_model = models[selected_idx]
        
        # Display model info
        st.markdown("#### Model Information")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Model Type", selected_model.get('model_type', 'Unknown'))
        with col2:
            st.metric("Size", f"{selected_model['size_mb']:.2f} MB")
        with col3:
            accuracy = selected_model.get('accuracy')
            if accuracy:
                st.metric("Accuracy", f"{accuracy*100:.2f}%")
            else:
                st.metric("Accuracy", "N/A")
        with col4:
            f1 = selected_model.get('f1')
            if f1:
                st.metric("F1-Score", f"{f1*100:.2f}%")
            else:
                st.metric("F1-Score", "N/A")
        
        # Show metadata if available
        if selected_model.get('metadata'):
            with st.expander("📋 Full Metadata", expanded=False):
                metadata = selected_model['metadata']
                
                st.markdown("**Training Configuration:**")
                st.write(f"- Task: {metadata.get('task', 'N/A')}")
                st.write(f"- Target: {metadata.get('target', 'N/A')}")
                st.write(f"- Features: {metadata.get('n_features', 0)}")
                st.write(f"- Train Samples: {metadata.get('train_samples', 0):,}")
                st.write(f"- Test Samples: {metadata.get('test_samples', 0):,}")
                st.write(f"- Normalization: {metadata.get('normalization', 'N/A')}")
                
                st.markdown("**Features Used:**")
                features = metadata.get('features', [])
                if features:
                    for i, feat in enumerate(features, 1):
                        st.write(f"{i}. {feat}")
                
                st.markdown("**Performance Metrics:**")
                metrics = metadata.get('metrics', {})
                if metrics:
                    metric_df = pd.DataFrame([metrics]).T
                    metric_df.columns = ['Value']
                    st.dataframe(metric_df)
        
        # Load model button
        if st.button("🔄 Load This Model", type="primary"):
            with st.spinner("Loading model..."):
                try:
                    result = loader.load_production_model(selected_model['filename'])
                    st.session_state.loaded_model_info = result
                    st.session_state.model_loaded = True
                    st.success("✅ Model loaded successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error loading model: {e}")
        
        # Show loaded model status
        if st.session_state.get('model_loaded', False):
            st.success("✅ Model is loaded and ready for predictions")
            
            if st.button("🗑️ Unload Model"):
                loader.unload_model()
                st.session_state.model_loaded = False
                st.session_state.loaded_model_info = None
                st.rerun()
    
    except Exception as e:
        st.error(f"Error in model selection: {e}")
        import traceback
        st.code(traceback.format_exc())


def show_live_predictions():
    """Live Predictions Tab"""
    st.markdown("### 🔮 Live Predictions")
    st.markdown("Apply ML model to scanner data and generate trade signals")
    
    # Check if model is loaded
    if not st.session_state.get('model_loaded', False):
        st.warning("⚠️ No model loaded. Go to 'Model Selection' tab first.")
        return
    
    try:
        from ml.prediction_engine import MLPredictionEngine
        
        # Initialize prediction engine
        if 'prediction_engine' not in st.session_state:
            st.session_state.prediction_engine = MLPredictionEngine(st.session_state.model_loader)
        
        engine = st.session_state.prediction_engine
        
        st.markdown("#### Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Date selector (use training data for demo)
            st.info("📊 Using training data for predictions (demo)")
            
            # Load training data
            training_data_path = "data/ml_training/raw/training_data_complete.parquet"
            if not os.path.exists(training_data_path):
                st.error("Training data not found")
                return
            
            df = pd.read_parquet(training_data_path)
            
            # Get unique dates
            if 'entry_date' in df.columns:
                df['entry_date'] = pd.to_datetime(df['entry_date'])
                available_dates = sorted(df['entry_date'].unique())
                
                selected_date = st.selectbox(
                    "Select Date",
                    available_dates,
                    index=len(available_dates)-1,  # Default to latest
                    format_func=lambda x: x.strftime('%Y-%m-%d')
                )
                
                # Filter data for selected date
                df_date = df[df['entry_date'] == selected_date].copy()
                
                st.metric("Stocks on Date", len(df_date))
            else:
                st.error("'entry_date' column not found in training data")
                return
        
        with col2:
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.5,
                max_value=0.9,
                value=0.6,
                step=0.05,
                help="Minimum confidence to generate BUY signal"
            )
            
            top_n = st.number_input(
                "Show Top N",
                min_value=5,
                max_value=100,
                value=20,
                step=5,
                help="Number of top predictions to display"
            )
        
        # Generate predictions button
        if st.button("🔮 Generate Predictions", type="primary"):
            with st.spinner("Generating predictions..."):
                try:
                    # Generate predictions
                    results = engine.predict_with_confidence(
                        df_date,
                        confidence_threshold=confidence_threshold
                    )
                    
                    st.session_state.prediction_results = results
                    st.success(f"✅ Generated predictions for {len(results)} stocks")
                    
                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    return
        
        # Display results
        if 'prediction_results' in st.session_state:
            results = st.session_state.prediction_results
            
            st.markdown("---")
            st.markdown("#### Prediction Results")
            
            # Summary metrics
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            
            with col_m1:
                total_stocks = len(results)
                st.metric("Total Stocks", total_stocks)
            
            with col_m2:
                buy_signals = (results['ml_trade_signal'] == 'BUY').sum()
                st.metric("BUY Signals", buy_signals)
            
            with col_m3:
                avg_confidence = results['ml_confidence'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            
            with col_m4:
                win_predictions = (results['ml_prediction'] == 1).sum()
                st.metric("WIN Predictions", f"{win_predictions} ({win_predictions/total_stocks:.1%})")
            
            # Filter options
            st.markdown("#### Filters")
            col_f1, col_f2 = st.columns(2)
            
            with col_f1:
                show_only_buy = st.checkbox("Show only BUY signals", value=True)
            
            with col_f2:
                sort_by = st.selectbox(
                    "Sort by",
                    ['ml_confidence', 'Ticker'],
                    index=0
                )
            
            # Apply filters
            display_df = results.copy()
            
            if show_only_buy:
                display_df = display_df[display_df['ml_trade_signal'] == 'BUY']
            
            # Sort
            display_df = display_df.sort_values(sort_by, ascending=False)
            
            # Limit to top N
            display_df = display_df.head(top_n)
            
            # Select columns to display
            display_columns = ['Ticker', 'ml_prediction_label', 'ml_confidence', 
                             'ml_trade_signal']
            
            # Add scanner score if available
            if 'Composite_Score' in display_df.columns:
                display_columns.insert(1, 'Composite_Score')
            
            # Format confidence as percentage
            display_df_formatted = display_df[display_columns].copy()
            display_df_formatted['ml_confidence'] = display_df_formatted['ml_confidence'].apply(lambda x: f"{x:.1%}")
            
            # Display table
            st.dataframe(
                display_df_formatted,
                use_container_width=True,
                hide_index=True
            )
            
            # Export button
            csv = display_df[display_columns].to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions (CSV)",
                data=csv,
                file_name=f"ml_predictions_{selected_date:%Y%m%d}.csv",
                mime="text/csv"
            )
    
    except Exception as e:
        st.error(f"Error in live predictions: {e}")
        import traceback
        st.code(traceback.format_exc())


def show_performance_dashboard():
    """Performance Dashboard Tab"""
    st.markdown("### 📈 Performance Dashboard")
    st.markdown("View model performance metrics and visualizations")
    
    # Check if model is loaded
    if not st.session_state.get('model_loaded', False):
        st.warning("⚠️ No model loaded. Go to 'Model Selection' tab first.")
        return
    
    try:
        # Get loaded model info
        model_info = st.session_state.get('loaded_model_info', {})
        metadata = model_info.get('metadata', {})
        
        if not metadata:
            st.warning("No metadata available for loaded model")
            return
        
        # Display metrics
        st.markdown("#### Performance Metrics")
        
        metrics = metadata.get('metrics', {})
        
        if metrics:
            # Classification metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                accuracy = metrics.get('accuracy', 0)
                st.metric("Accuracy", f"{accuracy*100:.2f}%")
            
            with col2:
                precision = metrics.get('precision', 0)
                st.metric("Precision", f"{precision*100:.2f}%")
            
            with col3:
                recall = metrics.get('recall', 0)
                st.metric("Recall", f"{recall*100:.2f}%")
            
            with col4:
                f1 = metrics.get('f1', 0)
                st.metric("F1-Score", f"{f1*100:.2f}%")
            
            with col5:
                roc_auc = metrics.get('roc_auc', 0)
                st.metric("ROC-AUC", f"{roc_auc:.4f}")
            
            # Confusion Matrix
            st.markdown("#### Confusion Matrix")
            
            cm = metrics.get('confusion_matrix', [[0, 0], [0, 0]])
            
            if cm and len(cm) == 2 and len(cm[0]) == 2:
                tn, fp = cm[0]
                fn, tp = cm[1]
                
                col_cm1, col_cm2 = st.columns(2)
                
                with col_cm1:
                    st.markdown("**Predicted Negative**")
                    st.metric("True Negatives", f"{tn:,}")
                    st.metric("False Negatives", f"{fn:,}")
                
                with col_cm2:
                    st.markdown("**Predicted Positive**")
                    st.metric("False Positives", f"{fp:,}")
                    st.metric("True Positives", f"{tp:,}")
                
                # Calculate additional metrics
                total = tn + fp + fn + tp
                st.markdown(f"**Total Samples:** {total:,}")
                st.markdown(f"**Positive Rate:** {(tp + fp)/total:.1%}")
                st.markdown(f"**Actual Positive Rate:** {(tp + fn)/total:.1%}")
        
        # Feature Importance
        st.markdown("---")
        st.markdown("#### Feature Importance")
        
        model = model_info.get('model')
        features = metadata.get('features', [])
        
        if model and hasattr(model, 'feature_importances_') and features:
            importances = model.feature_importances_
            
            # Create dataframe
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            # Display top 10
            st.dataframe(
                importance_df.head(10),
                use_container_width=True,
                hide_index=True
            )
            
            # Simple bar chart
            st.bar_chart(importance_df.set_index('Feature')['Importance'].head(10))
        else:
            st.info("Feature importance not available for this model")
        
        # Training Info
        st.markdown("---")
        st.markdown("#### Training Information")
        
        col_t1, col_t2, col_t3 = st.columns(3)
        
        with col_t1:
            st.metric("Train Samples", f"{metadata.get('train_samples', 0):,}")
        
        with col_t2:
            st.metric("Test Samples", f"{metadata.get('test_samples', 0):,}")
        
        with col_t3:
            st.metric("Features", metadata.get('n_features', 0))
        
        st.write(f"**Target Variable:** {metadata.get('target', 'N/A')}")
        st.write(f"**Normalization:** {metadata.get('normalization', 'N/A')}")
        st.write(f"**Split Method:** {metadata.get('split_method', 'N/A')}")
        
    except Exception as e:
        st.error(f"Error in performance dashboard: {e}")
        import traceback
        st.code(traceback.format_exc())
