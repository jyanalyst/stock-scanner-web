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

    # Phase selector
    st.markdown("---")

    # ===== PHASE 1: DATA COLLECTION =====
    with st.expander("🏗️ PHASE 1: Data Collection", expanded=True):
        st.markdown("""
        **Goal:** Build historical dataset with labeled outcomes

        **Process:**
        1. Run scanner on past dates (2023-01-01 to 2024-12-31)
        2. Calculate forward returns (2-day, 4-day)
        3. Label win/loss outcomes
        4. Save to training dataset

        **Expected Output:** 10,000-20,000 labeled samples
        """)

        col1, col2 = st.columns(2)

        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime(2023, 1, 1)
            )
            forward_days = st.multiselect(
                "Forward Return Periods (days)",
                options=[1, 2, 3, 4, 5],
                default=[2, 4]
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

        col_btn1, col_btn2, col_btn3 = st.columns(3)

        with col_btn1:
            if st.button("▶️ Start Collection", type="primary"):
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


def run_factor_analysis():
    """Placeholder for factor analysis"""
    st.info("Factor analysis coming in Phase 2 implementation")


if __name__ == "__main__":
    show()
