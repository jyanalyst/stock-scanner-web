"""
ML Lab - Strategy Optimization Interface
Separate from production scanner
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import os

def show():
    st.set_page_config(page_title="ML Lab", page_icon="🧪", layout="wide")

    st.title("🧪 ML Lab: Strategy Optimization")
    st.markdown("Develop and test machine learning models without affecting production scanner")

    # Define dates at function start (fixes variable scope error)
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)

    # Phase selector
    st.markdown("---")

    # ===== PHASE 1: DATA COLLECTION =====
    with st.expander("🏗️ PHASE 1: Data Collection", expanded=True):
        st.markdown("""
        **Goal:** Build historical dataset with labeled outcomes

        **Process:**
        1. Run scanner on past dates (2023-01-01 to 2024-12-31)
        2. Calculate forward returns (2-day, 3-day, 4-day)
        3. Label win/loss outcomes
        4. Save to training dataset

        **Expected Output:** 15,000-30,000 labeled samples (3x more data!)
        """)

        # ===== PRE-FLIGHT DATA QUALITY CHECK =====
        st.markdown("### 📋 Pre-Flight Data Quality Check")

        # Check if validation has been run
        validation_run = 'validation_results' in st.session_state
        validation_date = st.session_state.get('validation_date', 'Never')

        if validation_run:
            results = st.session_state.validation_results
            summary = results['summary']

            # Show validation summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("✅ Ready", summary['ready_count'])
            with col2:
                st.metric("⚠️ Partial", summary['partial_count'])
            with col3:
                st.metric("❌ Failed", summary['failed_count'])
            with col4:
                est_samples = summary['estimated_samples']['ready_only']
                st.metric("Est. Samples", f"{est_samples:,}")

            st.success(f"✅ Validation completed on {validation_date}")

            # Show issues if any
            issues = results['issues']
            if any(issues.values()):
                with st.expander("🔍 Issues Found", expanded=False):
                    if issues['missing_files']:
                        st.warning(f"**Missing files:** {', '.join(issues['missing_files'])}")
                    if issues['insufficient_data']:
                        st.warning(f"**Insufficient data:** {', '.join(issues['insufficient_data'])}")
                    if issues['data_gaps']:
                        st.info(f"**Data gaps:** {', '.join(issues['data_gaps'])}")
                    if issues['null_values']:
                        st.warning(f"**NULL values:** {', '.join(issues['null_values'])}")
                    if issues['integrity_issues']:
                        st.warning(f"**Integrity issues:** {', '.join(issues['integrity_issues'])}")

            # Download report button
            if st.button("📄 Download Full Report"):
                from ml.data_validator import MLDataValidator
                validator = MLDataValidator(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                report_path = validator.generate_report(results)
                with open(report_path, 'r', encoding='utf-8') as f:
                    st.download_button(
                        label="📄 Download HTML Report",
                        data=f.read(),
                        file_name="ml_data_quality_report.html",
                        mime="text/html"
                    )

            # ===== GAP FILLING SECTION =====
            needs_download = results.get('needs_download', [])
            if needs_download:
                st.markdown("### 🔧 Data Gap Filling")
                st.info(f"Found {len(needs_download)} stocks with gaps or incomplete coverage that could benefit from yfinance data.")

                # Show some examples
                if len(needs_download) <= 5:
                    st.write("**Stocks needing gap filling:**")
                    for stock in needs_download:
                        st.write(f"- **{stock['ticker']}**: {stock['reason']}")
                else:
                    st.write(f"**First 5 stocks needing gap filling:**")
                    for stock in needs_download[:5]:
                        st.write(f"- **{stock['ticker']}**: {stock['reason']}")
                    st.write(f"*... and {len(needs_download) - 5} more*")

                col_gap1, col_gap2 = st.columns(2)
                with col_gap1:
                    if st.button("📥 Fill Gaps with yfinance", type="primary"):
                        fill_data_gaps()
                with col_gap2:
                    if st.button("🔄 Re-validate Only"):
                        run_data_validation()
            else:
                st.success("✅ No stocks need gap filling - all data is complete!")

        else:
            st.warning("⚠️ **Recommended:** Run data quality check before collection to avoid issues!")

        col_val1, col_val2 = st.columns(2)
        with col_val1:
            if st.button("🔍 Validate Data Quality", type="secondary"):
                run_data_validation()
        with col_val2:
            if validation_run and st.button("🔄 Re-validate"):
                run_data_validation()

        st.markdown("---")

        # ===== DATA COLLECTION CONTROLS =====
        col1, col2 = st.columns(2)

        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime(2023, 1, 1)
            )
            forward_days = st.multiselect(
                "Forward Return Periods (days)",
                options=[1, 2, 3, 4, 5],
                default=[2, 3, 4],  # Updated default to include 3-day
                help="2-day: Quick exit, 3-day: Mid-point, 4-day: Full strategy"
            )

        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime(2024, 12, 31)
            )
            save_path = st.text_input(
                "Save Path",
                value="data/ml_training/raw/"
            )

        # Show estimated runtime
        days_to_process = (end_date - start_date).days
        est_runtime_hours = days_to_process * 0.1  # ~6 min per day
        st.info(f"📊 Estimated runtime: {est_runtime_hours:.1f} hours ({days_to_process} trading days)")

        # Collection controls
        col_btn1, col_btn2, col_btn3 = st.columns(3)

        # Enable/disable collection based on validation
        collection_enabled = validation_run and st.session_state.validation_results['summary']['ready_count'] > 0

        with col_btn1:
            if st.button("▶️ Start Collection", type="primary", disabled=not collection_enabled):
                if not collection_enabled:
                    st.error("Please run data validation first!")
                else:
                    start_data_collection(start_date, end_date, forward_days, save_path)

        with col_btn2:
            if st.button("⏸️ Pause"):
                st.session_state.collection_paused = True
                st.warning("Collection paused. Click Resume to continue.")

        with col_btn3:
            if st.button("📊 View Existing Data"):
                show_existing_training_data()

        # Show progress if collection is running
        if 'collection_running' in st.session_state and st.session_state.collection_running:
            show_collection_progress()

    # ===== PHASE 2: FACTOR ANALYSIS =====
    with st.expander("🔬 PHASE 2: Factor Analysis", expanded=False):
        st.markdown("""
        **Goal:** Identify which factors actually predict returns

        **Process:**
        1. Calculate Information Coefficient (IC) for each factor
        2. Analyze feature correlations (remove redundant features)
        3. Select optimal feature set
        4. Calculate optimal feature weights
        5. Optional: PCA dimensionality reduction

        **Prerequisites:** ≥10,000 training samples from Phase 1
        """)

        # Check if data exists
        data_exists = check_training_data_exists()

        if data_exists:
            sample_count = get_training_sample_count()
            st.success(f"✅ Training data available: {sample_count:,} samples")

            # Configuration
            st.markdown("### ⚙️ Analysis Configuration")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                target_var = st.selectbox(
                    "Target Variable",
                    options=['return_3d', 'return_2d', 'return_4d'],
                    index=0,
                    help="Which forward return to predict"
                )
            
            with col2:
                ic_threshold = st.slider(
                    "IC Threshold",
                    min_value=0.001,
                    max_value=0.10,
                    value=0.01,
                    step=0.005,
                    help="Minimum |IC| to keep feature (cross-sectional IC)"
                )
            
            with col3:
                corr_threshold = st.slider(
                    "Correlation Threshold",
                    min_value=0.70,
                    max_value=0.95,
                    value=0.85,
                    step=0.05,
                    help="Max correlation for redundancy"
                )
            
            run_pca = st.checkbox(
                "Run PCA Analysis (optional)",
                value=False,
                help="Dimensionality reduction analysis"
            )

            # Run analysis button
            if st.button("🔬 Run Factor Analysis", type="primary", key="run_factor_analysis"):
                run_factor_analysis(target_var, ic_threshold, corr_threshold, run_pca)
            
            # Show results if available
            if 'factor_analysis_results' in st.session_state:
                show_factor_analysis_results()
        else:
            st.warning("⚠️ No training data found. Complete Phase 1 first.")

    # ===== PHASE 3: MODEL TRAINING & PREDICTIONS =====
    from pages.ml_lab_phase3 import show_phase3
    show_phase3()

    # ===== PHASE 4: MODEL VALIDATION =====
    from pages.ml_lab_phase4 import show_phase4
    show_phase4()

    # ===== PHASE 5: ML SCANNER (STANDALONE) =====
    from pages.ml_lab_phase5 import show_phase5
    show_phase5()


def start_data_collection(start_date, end_date, forward_days, save_path):
    """Start background data collection"""
    from ml.data_collection import MLDataCollector

    st.session_state.collection_running = True
    st.session_state.collection_progress = 0

    with st.spinner("Initializing data collector..."):
        collector = MLDataCollector(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            forward_days=forward_days
        )

        # Run collection (this will take hours)
        try:
            training_data = collector.collect_training_data(save_path=save_path)
            st.success(f"✅ Collection complete! Saved {len(training_data):,} samples")
        except Exception as e:
            st.error(f"❌ Collection failed: {e}")
        finally:
            st.session_state.collection_running = False


def show_collection_progress():
    """Display live progress bar"""
    progress = st.session_state.get('collection_progress', 0)
    current_date = st.session_state.get('current_date', 'N/A')

    st.progress(progress / 100)
    st.write(f"Processing: {current_date}")
    st.write(f"Progress: {progress}%")


def show_existing_training_data():
    """Show summary of existing training data"""
    import os

    data_path = "data/ml_training/raw/"

    if os.path.exists(f"{data_path}/training_data_complete.parquet"):
        try:
            df = pd.read_parquet(f"{data_path}/training_data_complete.parquet")

            st.success(f"✅ Found training dataset: {len(df):,} samples")

            # Show available columns for debugging
            st.info(f"📊 Columns: {', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Samples", f"{len(df):,}")
            with col2:
                # Handle different possible date column names
                date_col = None
                for possible_col in ['entry_date', 'Date', 'date', 'scan_date']:
                    if possible_col in df.columns:
                        date_col = possible_col
                        break
                
                if date_col:
                    # Ensure it's datetime
                    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                        df[date_col] = pd.to_datetime(df[date_col])
                    date_range = f"{df[date_col].min():%Y-%m-%d} to {df[date_col].max():%Y-%m-%d}"
                    st.metric("Date Range", date_range)
                else:
                    st.metric("Date Range", "N/A")
            with col3:
                if 'Ticker' in df.columns:
                    unique_tickers = df['Ticker'].nunique()
                    st.metric("Unique Stocks", unique_tickers)
                else:
                    st.metric("Unique Stocks", "N/A")
            with col4:
                if 'return_2d' in df.columns:
                    avg_return_2d = df['return_2d'].mean() * 100
                    st.metric("Avg 2D Return", f"{avg_return_2d:.2f}%")
                else:
                    st.metric("Avg 2D Return", "N/A")

            st.dataframe(df.head(20))
            
        except Exception as e:
            st.error(f"❌ Error loading training data: {e}")
            st.code(f"File exists but couldn't be read. Error: {str(e)}")
    else:
        st.warning("No training data found. Start Phase 1 collection.")


def check_training_data_exists():
    """Check if training data exists"""
    data_path = "data/ml_training/raw/training_data_complete.parquet"
    return os.path.exists(data_path)


def get_training_sample_count():
    """Get number of training samples"""
    try:
        data_path = "data/ml_training/raw/training_data_complete.parquet"
        df = pd.read_parquet(data_path)
        return len(df)
    except:
        return 0


def run_data_validation():
    """Run comprehensive data quality validation"""
    with st.spinner("🔍 Validating data quality across all stocks..."):
        try:
            from ml.data_validator import MLDataValidator

            # Use the same date range as collection
            start_date = datetime(2023, 1, 1)
            end_date = datetime(2024, 12, 31)

            # Initialize validator
            validator = MLDataValidator(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                min_days=100
            )

            # Run validation
            results = validator.validate_all_stocks()

            # Store results in session state
            st.session_state.validation_results = results
            st.session_state.validation_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Generate HTML report
            report_path = validator.generate_report(results)

            # Show immediate summary
            summary = results['summary']
            st.success(f"✅ Validation complete! Found {summary['ready_count']} ready stocks.")

            if summary['failed_count'] > 0:
                st.warning(f"⚠️ {summary['failed_count']} stocks failed validation - check issues below.")

            # Force UI refresh
            st.rerun()

        except Exception as e:
            st.error(f"❌ Validation failed: {e}")
            import traceback
            st.code(traceback.format_exc())


def fill_data_gaps():
    """Download missing data using yfinance"""
    with st.spinner("📥 Downloading missing data from yfinance..."):
        try:
            from core.local_file_loader import get_local_loader
            from datetime import date

            # Get stocks that need downloading
            if 'validation_results' not in st.session_state:
                st.error("Please run validation first!")
                return

            needs_download = st.session_state.validation_results.get('needs_download', [])
            if not needs_download:
                st.success("✅ No stocks need gap filling!")
                return

            # Initialize loader
            loader = get_local_loader()

            # Set date range for gap filling
            start_date = date(2023, 1, 1)
            end_date = date(2024, 12, 31)

            # Download missing data for each stock
            total_stocks = len(needs_download)
            updated_count = 0
            failed_count = 0

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, stock_info in enumerate(needs_download):
                ticker = stock_info['ticker']
                reason = stock_info['reason']
                yf_ticker = ticker.replace('.SG', '.SI')  # Show yfinance format

                status_text.text(f"📥 Processing {ticker} → {yf_ticker} ({i+1}/{total_stocks}) - {reason}")

                try:
                    # Download missing dates for this stock
                    result = loader._download_and_append_single_stock(
                        ticker, start_date, end_date, force_mode=False
                    )

                    if result['status'] == 'updated':
                        updated_count += 1
                        dates_added = result.get('dates_added', 0)
                        st.info(f"✅ {ticker} → {yf_ticker}: Added {dates_added} dates from yfinance")
                    elif result['status'] == 'skipped':
                        failed_count += 1
                        message = result.get('message', 'No new dates')
                        if 'No data available' in message or 'delisted' in message.lower():
                            st.warning(f"⚠️ {ticker} → {yf_ticker}: No data (possibly newly listed or delisted)")
                        else:
                            st.warning(f"⚠️ {ticker} → {yf_ticker}: {message}")
                    else:
                        failed_count += 1
                        st.warning(f"⚠️ {ticker} → {yf_ticker}: {result.get('message', 'Failed')}")

                except Exception as e:
                    failed_count += 1
                    st.error(f"❌ {ticker} → {yf_ticker}: {e}")

                # Update progress
                progress_bar.progress((i + 1) / total_stocks)

            # Show final results
            status_text.empty()
            progress_bar.empty()

            if updated_count > 0:
                st.success(f"✅ Gap filling complete! Updated {updated_count} stocks.")
                if failed_count > 0:
                    st.warning(f"⚠️ {failed_count} stocks failed to update.")

                # Auto-revalidate after gap filling
                st.info("🔄 Re-validating data quality...")
                run_data_validation()

            else:
                st.warning("⚠️ No stocks were updated. Check yfinance connectivity.")

        except Exception as e:
            st.error(f"❌ Gap filling failed: {e}")
            import traceback
            st.code(traceback.format_exc())


def run_factor_analysis(target_var, ic_threshold, corr_threshold, run_pca):
    """Run complete factor analysis"""
    from ml.factor_analyzer import MLFactorAnalyzer
    from ml.visualizations import MLVisualizer
    
    with st.spinner("🔬 Running factor analysis... This may take a few minutes."):
        try:
            # Initialize analyzer
            analyzer = MLFactorAnalyzer(target=target_var)
            
            # Run full analysis
            results = analyzer.run_full_analysis(
                ic_threshold=ic_threshold,
                correlation_threshold=corr_threshold,
                run_pca=run_pca
            )
            
            # Create visualizations
            visualizer = MLVisualizer()
            
            # Get redundant pairs for visualization
            _, redundant_pairs = analyzer.analyze_correlations(corr_threshold)
            
            figures = visualizer.create_summary_dashboard(
                ic_results=results['ic_results'],
                correlation_matrix=results['correlation_matrix'],
                optimal_weights=results['optimal_weights'],
                redundant_pairs=redundant_pairs
            )
            
            # Store results in session state
            st.session_state.factor_analysis_results = {
                'analyzer': analyzer,
                'results': results,
                'figures': figures,
                'redundant_pairs': redundant_pairs,
                'target_var': target_var,
                'ic_threshold': ic_threshold,
                'corr_threshold': corr_threshold
            }
            
            st.success("✅ Factor analysis complete!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Factor analysis failed: {e}")
            import traceback
            st.code(traceback.format_exc())


def show_factor_analysis_results():
    """Display factor analysis results"""
    st.markdown("---")
    st.markdown("## 📊 Factor Analysis Results")
    
    results_data = st.session_state.factor_analysis_results
    results = results_data['results']
    figures = results_data['figures']
    analyzer = results_data['analyzer']
    
    # Summary metrics
    st.markdown("### 📈 Summary Statistics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Target Variable", results_data['target_var'])
    with col2:
        st.metric("Total Features", len(analyzer.features))
    with col3:
        strong_features = (results['ic_results']['abs_IC'] > 0.10).sum()
        st.metric("Strong Features", strong_features, help="|IC| > 0.10")
    with col4:
        st.metric("Selected Features", len(results['selected_features']))
    with col5:
        reduction_pct = (1 - len(results['selected_features']) / len(analyzer.features)) * 100
        st.metric("Reduction", f"{reduction_pct:.1f}%")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 IC Rankings",
        "🔗 Correlations",
        "🎯 Feature Selection",
        "⚖️ Optimal Weights",
        "📉 PCA (Optional)",
        "📥 Export"
    ])
    
    with tab1:
        st.markdown("### Information Coefficient Rankings")
        st.markdown("*Measures predictive power of each feature*")
        
        # Show IC bar chart
        if 'ic_bar_chart' in figures:
            st.plotly_chart(figures['ic_bar_chart'], width="stretch")
        
        # Show IC distribution
        if 'ic_distribution' in figures:
            st.plotly_chart(figures['ic_distribution'], width="stretch")
        
        # Show IC vs sample size
        if 'ic_vs_sample_size' in figures:
            st.plotly_chart(figures['ic_vs_sample_size'], width="stretch")
        
        # Show top 20 features table
        st.markdown("#### Top 20 Features by |IC|")
        top_20 = results['ic_results'].head(20)[['feature', 'IC_mean', 'abs_IC', 'p_value', 'significant', 'sample_size']]
        st.dataframe(top_20, width="stretch", hide_index=True)
    
    with tab2:
        st.markdown("### Feature Correlations")
        st.markdown("*Identify redundant features*")
        
        # Show correlation heatmap
        if 'correlation_heatmap' in figures:
            st.plotly_chart(figures['correlation_heatmap'], width="stretch")
        
        # Show redundant pairs
        if 'redundant_pairs' in figures:
            st.plotly_chart(figures['redundant_pairs'], width="stretch")
        
        # Show redundant pairs table
        redundant_pairs = results_data['redundant_pairs']
        if redundant_pairs:
            st.markdown(f"#### Redundant Feature Pairs (correlation > {results_data['corr_threshold']})")
            pairs_df = pd.DataFrame(redundant_pairs)[['feature1', 'feature2', 'correlation', 'ic1', 'ic2', 'keep', 'remove']]
            st.dataframe(pairs_df, width="stretch", hide_index=True)
        else:
            st.success("✅ No highly correlated feature pairs found!")
    
    with tab3:
        st.markdown("### Feature Selection Results")
        st.markdown(f"*Selected {len(results['selected_features'])} features based on IC threshold = {results_data['ic_threshold']}*")
        
        # Show selected features
        selected_ic = results['ic_results'][results['ic_results']['feature'].isin(results['selected_features'])]
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("#### ✅ Selected Features")
            st.dataframe(
                selected_ic[['feature', 'IC_mean', 'abs_IC']].head(30),
                width="stretch",
                hide_index=True
            )
        
        with col_b:
            st.markdown("#### ❌ Removed Features")
            removed_features = [f for f in analyzer.features if f not in results['selected_features']]
            removed_ic = results['ic_results'][results['ic_results']['feature'].isin(removed_features)]
            
            if len(removed_ic) > 0:
                st.dataframe(
                    removed_ic[['feature', 'IC_mean', 'abs_IC']].head(30),
                    width="stretch",
                    hide_index=True
                )
            else:
                st.info("No features removed")
    
    with tab4:
        st.markdown("### Optimal Feature Weights")
        st.markdown("*IC²-weighted feature importance*")
        
        # Show feature importance chart
        if 'feature_importance' in figures:
            st.plotly_chart(figures['feature_importance'], width="stretch")
        
        # Show weights table
        st.markdown("#### Top 30 Feature Weights")
        weights_df = pd.DataFrame([
            {'feature': k, 'weight': v, 'weight_pct': f"{v*100:.2f}%"}
            for k, v in results['optimal_weights'].items()
        ]).sort_values('weight', ascending=False).head(30)
        
        st.dataframe(weights_df, width="stretch", hide_index=True)
        
        # Show interpretation
        st.info("""
        **How to use these weights:**
        - Multiply each feature by its weight
        - Sum weighted features to create composite score
        - Higher weights = stronger predictive power
        - Use for Phase 3 model training
        """)
    
    with tab5:
        st.markdown("### PCA Analysis (Optional)")
        
        if results['pca_results']:
            pca_results = results['pca_results']
            
            st.success(f"✅ PCA Complete: {pca_results['n_components_95']} components explain 95% variance")
            
            # Show scree plot
            from ml.visualizations import MLVisualizer
            visualizer = MLVisualizer()
            
            scree_fig = visualizer.plot_pca_scree(pca_results)
            st.plotly_chart(scree_fig, width="stretch")
            
            # Show loadings plot
            loadings_fig = visualizer.plot_pca_loadings(pca_results, component_x=1, component_y=2, top_n=15)
            st.plotly_chart(loadings_fig, width="stretch")
            
            # Show variance explained
            st.markdown("#### Variance Explained by Component")
            variance_df = pd.DataFrame({
                'Component': [f'PC{i+1}' for i in range(min(10, len(pca_results['explained_variance_ratio'])))],
                'Variance (%)': pca_results['explained_variance_ratio'][:10] * 100,
                'Cumulative (%)': pca_results['cumulative_variance'][:10] * 100
            })
            st.dataframe(variance_df, width="stretch", hide_index=True)
        else:
            st.info("PCA analysis was not run. Enable 'Run PCA Analysis' to see results.")
    
    with tab6:
        st.markdown("### 📥 Export Results")
        
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        
        with col_exp1:
            # Download IC results
            ic_csv = results['ic_results'].to_csv(index=False)
            st.download_button(
                label="📊 Download IC Results (CSV)",
                data=ic_csv,
                file_name=f"ic_results_{results_data['target_var']}.csv",
                mime="text/csv"
            )
        
        with col_exp2:
            # Download optimal weights
            import json
            weights_json = json.dumps(results['optimal_weights'], indent=2)
            st.download_button(
                label="⚖️ Download Weights (JSON)",
                data=weights_json,
                file_name=f"optimal_weights_{results_data['target_var']}.json",
                mime="application/json"
            )
        
        with col_exp3:
            # Download selected features
            features_json = json.dumps(results['selected_features'], indent=2)
            st.download_button(
                label="✅ Download Selected Features (JSON)",
                data=features_json,
                file_name=f"selected_features_{results_data['target_var']}.json",
                mime="application/json"
            )
        
        # Download HTML report
        if results['report_path']:
            st.markdown("---")
            st.markdown("#### 📄 Full HTML Report")
            
            try:
                with open(results['report_path'], 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                st.download_button(
                    label="📄 Download Full Report (HTML)",
                    data=html_content,
                    file_name=f"factor_analysis_report_{results_data['target_var']}.html",
                    mime="text/html"
                )
                
                st.success(f"✅ Report saved to: {results['report_path']}")
            except Exception as e:
                st.error(f"Error loading report: {e}")
        
        # Show file locations
        st.markdown("---")
        st.markdown("#### 📁 Saved Files")
        st.code("""
data/ml_training/analysis/
├── ic_results.csv
├── correlation_matrix.csv
├── optimal_weights.json
├── selected_features.json
└── factor_analysis_report.html
        """)


if __name__ == "__main__":
    show()
