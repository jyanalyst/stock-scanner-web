# File: app.py
"""
Stock Scanner Web Application
Main Streamlit application with navigation for Higher H/L scanner and factor analysis
"""

# File: app.py
# Add these lines at the very top, before other imports
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Verify they loaded (optional - for debugging)
print(f"Historical Folder ID: {os.getenv('GDRIVE_HISTORICAL_FOLDER_ID')}")
print(f"EOD Folder ID: {os.getenv('GDRIVE_EOD_FOLDER_ID')}")

# Now continue with the rest of your imports
import streamlit as st
import sys
# ... rest of your code

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
    
    Your sophisticated stock scanning application with CRT (Candle Range Theory) analysis.
    
    ### Available Features:
    
    #### 📈 **CRT Higher H/L Scanner**
    - Focuses on stocks showing Higher High AND Higher Low patterns
    - Identifies trending momentum with expanding range
    - Best for trend continuation plays
    - Flexible filtering for Valid CRT and/or Higher H/L patterns
    
    #### 🔬 **Factor Analysis**
    - Validate technical analysis effectiveness with historical data
    - Analyze success rates of MPI trends, IBS levels, and pattern combinations
    - Smart incremental processing with file upload/download workflow
    - Comprehensive performance analytics and factor analysis
    
    ### Key Features:
    - 📊 **Real-time Analysis** - Live scanning of 46 Singapore Exchange stocks
    - 📈 **Technical Analysis** - Volume-weighted range percentiles and CRT levels
    - 🎯 **Dynamic Filtering** - Velocity and pattern-specific filters
    - 📋 **TradingView Export** - Direct export to TradingView watchlists
    - 📥 **CSV Downloads** - Export filtered results for further analysis
    - 🕒 **Historical Analysis** - Scan as of any past trading date
    - 🔬 **Strategy Validation** - Quantitative factor analysis of trading signals
    
    ---
    
    **Get Started:**
    👈 Use the sidebar to navigate to the scanner or factor analysis tools!
    """)
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Stocks Tracked", "46", delta="SGX Listed")
    
    with col2:
        st.metric("Scanner Types", "1", delta="Higher H/L Focus")
    
    with col3:
        if 'last_scan_time' in st.session_state:
            st.metric("Last Scan", st.session_state.last_scan_time.split()[1][:5], delta="Recent")
        else:
            st.metric("Last Scan", "Ready", delta="Click to start")
    
    with col4:
        st.metric("System Status", "Ready", delta="Fully operational")

def show_sidebar_stats():
    """Show quick stats in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Quick Stats")
    
    if 'scan_results' in st.session_state and not st.session_state.scan_results.empty:
        results = st.session_state.scan_results
        total_stocks = len(results)
        valid_crt = len(results[results['Valid_CRT'] == 1])
        higher_hl = len(results[results['Higher_HL'] == 1])
        
        st.sidebar.metric("Stocks Scanned", total_stocks)
        st.sidebar.metric("Valid CRT", valid_crt)
        st.sidebar.metric("Higher H/L", higher_hl)
    
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
        "📈 CRT Higher H/L": "higher_hl",
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
    
    elif page_value == "higher_hl":
        try:
            from pages import scanner_higher_hl
            scanner_higher_hl.show()
        except ImportError as e:
            st.error(f"Scanner module not found: {e}")
            st.info("This scanner focuses on stocks with Higher High AND Higher Low patterns.")
    
    elif page_value == "factor_analysis":
        try:
            from pages import factor_analysis
            factor_analysis.show()
        except ImportError as e:
            st.error(f"Factor analysis module not found: {e}")
            st.info("This module provides historical validation of trading signals and factor effectiveness analysis.")
    
    elif page_value == "historical":
        st.title("📊 Historical Analysis")
        st.info("Historical analysis page coming soon! This will show:")
        st.markdown("""
        - Performance tracking of past signals
        - Win/loss ratios for each scanner type
        - Best performing stocks by pattern
        - Strategy effectiveness over time
        """)
    
    elif page_value == "watchlist":
        st.title("📋 Watchlist Manager") 
        st.info("Watchlist management page coming soon! This will allow:")
        st.markdown("""
        - Add/remove stocks from scanning list
        - Import stocks from CSV
        - Manage stock categories
        - Custom watchlist creation
        """)
    
    elif page_value == "settings":
        st.title("⚙️ Settings")
        st.info("Settings page coming soon! This will include:")
        st.markdown("""
        - Scanning parameters (days back, rolling windows)
        - Email notification settings
        - Alert thresholds
        - Data source configuration
        """)

if __name__ == "__main__":
    main()