# File: app.py
"""
Stock Scanner Web Application
Main Streamlit application with navigation for Scanner and factor analysis
"""

import streamlit as st
import sys

# Page configuration
st.set_page_config(
    page_title="Stock Scanner Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

def show_home_page():
    """Display the home/welcome page"""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Stock Scanner Pro</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Welcome to Stock Scanner Pro! 🚀
    
    Your sophisticated stock scanning application with CRT (Candle Range Theory) analysis and Analyst Report Integration.
    
    ### Available Features:
    
    #### 📈 **Stock Scanner**
    - Focuses on stocks showing Higher High AND Higher Low patterns
    - Identifies trending momentum with expanding range
    - Best for trend continuation plays
    - Flexible filtering for Valid CRT and/or Higher H/L patterns
    - **NEW:** Integrated analyst sentiment analysis
    
    #### 🔬 **Factor Analysis**
    - Validate technical analysis effectiveness with historical data
    - Analyze success rates of MPI trends, IBS levels, and pattern combinations
    - Smart incremental processing with file upload/download workflow
    - Comprehensive performance analytics and factor analysis
    
    #### 📊 **Analyst Reports** (NEW)
    - Automated PDF processing with FinBERT sentiment analysis
    - Display analyst sentiment alongside technical indicators
    - Track multiple reports per stock with history
    - View catalysts, risks, and price targets
    
    ### Key Features:
    - 📊 **Real-time Analysis** - Live scanning of Singapore Exchange stocks
    - 📈 **Technical Analysis** - Volume-weighted range percentiles and CRT levels
    - 🎯 **Dynamic Filtering** - Velocity and pattern-specific filters
    - 📋 **TradingView Export** - Direct export to TradingView watchlists
    - 📥 **CSV Downloads** - Export filtered results for further analysis
    - 🕒 **Historical Analysis** - Scan as of any past trading date
    - 🔬 **Strategy Validation** - Quantitative factor analysis of trading signals
    - 💾 **Local File Storage** - Fast, simple data access from local CSV files
    - 📊 **Sentiment Analysis** - AI-powered analyst report processing (NEW)
    
    ---
    
    **Get Started:**
    👈 Use the sidebar to navigate to the scanner or factor analysis tools!
    """)
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'scan_results' in st.session_state and not st.session_state.scan_results.empty:
            st.metric("Stocks Tracked", len(st.session_state.scan_results), delta="SGX Listed")
        else:
            st.metric("Stocks Tracked", "Ready", delta="SGX Listed")
    
    with col2:
        st.metric("Scanner Types", "1", delta="Higher H/L Focus")
    
    with col3:
        if 'last_scan_time' in st.session_state:
            st.metric("Last Scan", st.session_state.last_scan_time.split()[1][:5], delta="Recent")
        else:
            st.metric("Last Scan", "Ready", delta="Click to start")
    
    with col4:
        # Check if analyst reports exist
        try:
            from utils.analyst_reports import get_cached_reports
            _, latest_reports = get_cached_reports()
            report_count = len(latest_reports) if not latest_reports.empty else 0
            st.metric("Analyst Reports", report_count, delta="Active")
        except:
            st.metric("System Status", "Ready", delta="Operational")

def show_sidebar_stats():
    """Show quick stats in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Quick Stats")
    
    if 'scan_results' in st.session_state and not st.session_state.scan_results.empty:
        results = st.session_state.scan_results
        total_stocks = len(results)
        
        # Count Signal_Bias (new dual-bias system)
        if 'Signal_Bias' in results.columns:
            bullish_count = len(results[results['Signal_Bias'] == '🟢 BULLISH'])
            bearish_count = len(results[results['Signal_Bias'] == '🔴 BEARISH'])
            
            st.sidebar.metric("Stocks Scanned", total_stocks)
            st.sidebar.metric("🟢 Bullish", bullish_count)
            st.sidebar.metric("🔴 Bearish", bearish_count)
        else:
            # Fallback if Signal_Bias not available yet
            st.sidebar.metric("Stocks Scanned", total_stocks)
        
        # Show analyst coverage if available
        if 'sentiment_score' in results.columns:
            with_reports = results['sentiment_score'].notna().sum()
            if with_reports > 0:
                st.sidebar.metric("Analyst Coverage", with_reports)
    
    # Show factor analysis stats if available
    if 'factor_analysis_summary' in st.session_state:
        st.sidebar.markdown("### 🔬 Factor Analysis Stats")
        summary = st.session_state.factor_analysis_summary
        st.sidebar.metric("Total Signals", summary.get('total_signals', 0))
        st.sidebar.metric("Success Rate", f"{summary.get('success_rate', 0):.1f}%")

def main():
    """Main application logic with navigation"""
    
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    
    # Navigation options
    pages = {
        "🏠 Home": "home",
        "📈 Scanner": "scanner",
        "📊 RVOL BackTest": "rvol_backtest",
        "📊 Analysis (Analyst Reports)": "analyst_reports",
        "📊 Analysis (Earnings Reports)": "earnings_reports",
        "🔬 Factor Analysis": "factor_analysis",
        "📊 Historical Analysis": "historical",
        "📋 Watchlist Manager": "watchlist",
        "⚙️ Settings": "settings"
    }
    
    # Page selection
    selected_page = st.sidebar.selectbox(
        "Choose a page:",
        list(pages.keys()),
        index=0
    )
    
    # Show sidebar stats
    show_sidebar_stats()
    
    # Page routing
    page_value = pages[selected_page]
    
    if page_value == "home":
        show_home_page()
    
    elif page_value == "scanner":
        try:
            from pages import scanner
            scanner.show()
        except ImportError as e:
            st.error(f"Scanner module not found: {e}")
            st.info("This scanner focuses on stocks with Higher High AND Higher Low patterns, with integrated analyst sentiment analysis.")

    elif page_value == "rvol_backtest":
        try:
            from pages import rvol_backtest
            rvol_backtest.show()
        except ImportError as e:
            st.error(f"RVOL BackTest module not found: {e}")
            st.info("This module provides Monthly VWAP mean reversion backtesting with realistic retest fill logic.")

    elif page_value == "analyst_reports":
        try:
            from pages import analyst_reports_analysis
            analyst_reports_analysis.show()
        except ImportError as e:
            st.error(f"Analyst Reports Analysis module not found: {e}")
            st.info("This module provides comprehensive analysis of analyst reports with sentiment analysis and price targets.")

    elif page_value == "earnings_reports":
        try:
            from pages import earnings_reports_analysis
            earnings_reports_analysis.show()
        except ImportError as e:
            st.error(f"Earnings Reports Analysis module not found: {e}")
            st.info("This module provides comprehensive earnings analysis combining detailed report viewing with advanced trend analysis.")

    elif page_value == "factor_analysis":
        try:
            from pages import factor_analysis
            factor_analysis.show()
        except ImportError as e:
            st.error(f"Factor analysis module not found: {e}")
            st.info("This module provides historical validation of trading signals and factor effectiveness analysis.")
    
    elif page_value == "historical":
        st.title("📊 Historical Analysis")
        st.warning("🚧 Historical analysis page is not yet implemented")
        st.info("This feature is planned for a future release and will include:")
        st.markdown("""
        - Performance tracking of past signals
        - Win/loss ratios for each scanner type
        - Best performing stocks by pattern
        - Strategy effectiveness over time
        """)

    elif page_value == "watchlist":
        st.title("📋 Watchlist Manager")
        st.warning("🚧 Watchlist management page is not yet implemented")
        st.info("This feature is planned for a future release and will allow:")
        st.markdown("""
        - Add/remove stocks from scanning list
        - Import stocks from CSV
        - Manage stock categories
        - Custom watchlist creation
        """)

    elif page_value == "settings":
        st.title("⚙️ Settings")
        st.warning("🚧 Settings page is not yet implemented")
        st.info("This feature is planned for a future release and will include:")
        st.markdown("""
        - Scanning parameters (days back, rolling windows)
        - Email notification settings
        - Alert thresholds
        - Data source configuration
        - Analyst report processing settings
        """)

if __name__ == "__main__":
    main()
