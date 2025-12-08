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
        2. Test interaction terms
        3. Discover new factors
        4. Remove redundant features

        **Prerequisites:** ≥10,000 training samples from Phase 1
        """)

        # Check if data exists
        data_exists = check_training_data_exists()

        if data_exists:
            sample_count = get_training_sample_count()
            st.success(f"✅ Training data available: {sample_count:,} samples")

            if st.button("🔬 Calculate Factor ICs", type="primary"):
                run_factor_analysis()
        else:
            st.warning("⚠️ No training data found. Complete Phase 1 first.")

    # ===== PHASE 3-5: COMING SOON =====
    with st.expander("🧠 PHASE 3: Model Training", expanded=False):
        st.info("Coming soon after Phase 2 completion")

    with st.expander("📊 PHASE 4: Validation", expanded=False):
        st.info("Coming soon after Phase 3 completion")

    with st.expander("🚀 PHASE 5: Deployment", expanded=False):
        st.info("Coming soon after Phase 4 completion")


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
        df = pd.read_parquet(f"{data_path}/training_data_complete.parquet")

        st.success(f"✅ Found training dataset: {len(df):,} samples")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", f"{len(df):,}")
        with col2:
            date_range = f"{df['entry_date'].min():%Y-%m-%d} to {df['entry_date'].max():%Y-%m-%d}"
            st.metric("Date Range", date_range)
        with col3:
            unique_tickers = df['Ticker'].nunique()
            st.metric("Unique Stocks", unique_tickers)
        with col4:
            avg_return_2d = df['return_2d'].mean() * 100
            st.metric("Avg 2D Return", f"{avg_return_2d:.2f}%")

        st.dataframe(df.head(20))
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


def run_factor_analysis():
    """Placeholder for factor analysis"""
    st.info("Factor analysis coming in Phase 2 implementation")


if __name__ == "__main__":
    show()
