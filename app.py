"""
Stock Scanner Web Application
Main Streamlit application with navigation for three specialized scanners
"""

import streamlit as st
import os

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
    
    Your sophisticated stock scanning application with three specialized CRT (Candle Range Theory) scanners.
    
    ### Available Scanners:
    
    #### 📈 **CRT Higher H/L Scanner**
    - Focuses on stocks showing Higher High AND Higher Low patterns
    - Identifies trending momentum with expanding range
    - Best for trend continuation plays
    
    #### 📊 **CRT Wick Below Scanner** 
    - Tracks stocks that test and bounce off CRT Low levels
    - Measures bounce strength and recovery patterns
    - Ideal for support level plays
    
    #### 🚀 **CRT Close Above Scanner**
    - Monitors stocks breaking above CRT High levels
    - Analyzes breakout strength and momentum
    - Perfect for momentum breakout plays
    
    ### Key Features:
    - 📊 **Real-time Analysis** - Live scanning of 46 Singapore Exchange stocks
    - 📈 **Technical Analysis** - Volume-weighted range percentiles and CRT levels
    - 🎯 **Dynamic Filtering** - Velocity and pattern-specific filters
    - 📋 **TradingView Export** - Direct export to TradingView watchlists
    - 📥 **CSV Downloads** - Export filtered results for further analysis
    
    ---
    
    **Get Started:**
    👈 Use the sidebar to navigate to your preferred scanner!
    """)
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Stocks Tracked", "46", delta="SGX Listed")
    
    with col2:
        st.metric("Scanner Types", "3", delta="Specialized")
    
    with col3:
        if 'last_scan_time' in st.session_state:
            st.metric("Last Scan", st.session_state.last_scan_time.split()[1][:5], delta="Recent")
        else:
            st.metric("Last Scan", "Ready", delta="Click to start")
    
    with col4:
        st.metric("System Status", "Ready", delta="Fully operational")

def main():
    """Main application logic with navigation"""
    
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    
    # Navigation options
    pages = {
        "🏠 Home": "home",
        "📈 CRT Higher H/L": "higher_hl",
        "📊 CRT Wick Below": "wick_below",
        "🚀 CRT Close Above": "close_above",
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
    
    # Quick stats in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Quick Stats")
    
    if 'scan_results' in st.session_state:
        total_stocks = len(st.session_state.scan_results)
        valid_crt = len(st.session_state.scan_results[st.session_state.scan_results['Valid_CRT'] == 1])
        
        st.sidebar.metric("Stocks Scanned", total_stocks)
        st.sidebar.metric("Valid CRT", valid_crt)
    
    # Page routing
    page_value = pages[selected_page]
    
    if page_value == "home":
        show_home_page()
    
    elif page_value == "higher_hl":
        try:
            from pages import scanner_higher_hl
            scanner_higher_hl.show()
        except ImportError:
            st.error("CRT Higher H/L scanner module not found. Please ensure pages/scanner_higher_hl.py exists.")
            st.info("This scanner focuses on stocks with Higher High AND Higher Low patterns.")
    
    elif page_value == "wick_below":
        try:
            from pages import scanner_wick_below
            scanner_wick_below.show()
        except ImportError:
            st.error("CRT Wick Below scanner module not found. Please ensure pages/scanner_wick_below.py exists.")
            st.info("This scanner tracks stocks that test and bounce off CRT Low levels.")
    
    elif page_value == "close_above":
        try:
            from pages import scanner_close_above
            scanner_close_above.show()
        except ImportError:
            st.error("CRT Close Above scanner module not found. Please ensure pages/scanner_close_above.py exists.")
            st.info("This scanner monitors stocks breaking above CRT High levels.")
    
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