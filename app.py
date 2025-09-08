"""
Stock Scanner Web Application
Main Streamlit application with navigation
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
    
    This is your sophisticated stock scanning application, migrated from Jupyter notebook to a professional web interface.
    
    ### Features Available:
    - 📊 **Live Stock Scanning** - Real-time analysis of 46 Singapore Exchange stocks
    - 📈 **Technical Analysis** - Volume-weighted range percentiles and CRT levels
    - 🎯 **Buy Signal Detection** - Automated signal generation based on your proven strategy
    - 📧 **Email Notifications** - Professional HTML reports
    - 📋 **Watchlist Management** - Easy stock portfolio management
    
    ### Current Status:
    ✅ Project structure created  
    ✅ Core functions migrated  
    ✅ Live scanner interface ready  
    ⏳ Database setup pending  
    ⏳ Email notifications pending  
    
    ---
    
    **Get Started:**
    👈 Use the sidebar to navigate to **"🔍 Live Scanner"** to start analyzing stocks!
    """)
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Stocks Tracked", "46", delta="Active")
    
    with col2:
        if 'last_scan_time' in st.session_state:
            st.metric("Last Scan", st.session_state.last_scan_time.split()[1][:5], delta="Recent")
        else:
            st.metric("Last Scan", "Ready", delta="Click to start")
    
    with col3:
        if 'scan_results' in st.session_state:
            buy_signals = len(st.session_state.scan_results[st.session_state.scan_results['Buy_Signal'] == 1])
            st.metric("Buy Signals", str(buy_signals), delta="Current scan")
        else:
            st.metric("Buy Signals", "0", delta="No scan yet")
    
    with col4:
        st.metric("System Status", "Ready", delta="Fully operational")

def main():
    """Main application logic with navigation"""
    
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    
    # Navigation options
    pages = {
        "🏠 Home": "home",
        "🔍 Live Scanner": "scanner",
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
        buy_signals = len(st.session_state.scan_results[st.session_state.scan_results['Buy_Signal'] == 1])
        
        st.sidebar.metric("Stocks Scanned", total_stocks)
        st.sidebar.metric("Buy Signals", buy_signals)
        
        if st.sidebar.button("🔄 Quick Scan", help="Run a quick 5-stock test scan"):
            st.session_state.quick_scan = True
            st.rerun()
    
    # Page routing
    page_value = pages[selected_page]
    
    if page_value == "home":
        show_home_page()
    
    elif page_value == "scanner":
        try:
            from pages import scanner
            scanner.show()
        except ImportError:
            st.error("Scanner module not found. Please create pages/scanner.py with the provided code.")
            st.info("Copy the scanner page code into pages/scanner.py to enable live scanning.")
    
    elif page_value == "historical":
        st.title("📊 Historical Analysis")
        st.info("Historical analysis page coming soon! This will show:")
        st.markdown("""
        - Performance tracking of past signals
        - Win/loss ratios
        - Best performing stocks
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
    
    # Handle quick scan trigger
    if st.session_state.get('quick_scan', False):
        st.session_state.quick_scan = False
        if page_value == "scanner":
            st.info("🚀 Quick scan triggered! Use the scanner page to run analysis.")
        else:
            st.info("👈 Navigate to the Live Scanner page to run the quick scan.")

if __name__ == "__main__":
    main()