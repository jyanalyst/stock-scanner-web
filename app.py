# File: app.py
"""
Stock Scanner - Production Application
Main Streamlit application for daily trading scanner
"""

import streamlit as st
import sys

# Page configuration
st.set_page_config(
    page_title="Stock Scanner",
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
    st.markdown('<h1 class="main-header">📈 Stock Scanner - Production</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Welcome to Stock Scanner! 🚀
    
    Your production stock scanning application for daily trading operations.
    
    ### Available Features:
    
    #### 📈 **Stock Scanner**
    - Real-time scanning of Singapore Exchange stocks
    - Technical analysis with CRT (Candle Range Theory)
    - Dynamic filtering and pattern recognition
    - TradingView export capability
    
    #### 📊 **Analyst Reports Analysis**
    - Automated PDF processing with sentiment analysis
    - Track analyst recommendations and price targets
    - View catalysts and risks
    
    #### 📈 **Earnings Reports Analysis**
    - Earnings report tracking and analysis
    - Historical earnings performance
    - Earnings reaction analysis
    
    ### Key Features:
    - 📊 **Real-time Analysis** - Live scanning of SGX stocks
    - 📈 **Technical Analysis** - Volume-weighted percentiles and CRT levels
    - 🎯 **Dynamic Filtering** - Pattern-specific filters
    - 📋 **TradingView Export** - Direct export to TradingView watchlists
    - 📥 **CSV Downloads** - Export results for further analysis
    - 🕒 **Historical Analysis** - Scan as of any past trading date
    - 💾 **Local File Storage** - Fast data access from local files
    
    ---
    
    **Get Started:**
    👈 Use the sidebar to navigate to the scanner or analysis tools!
    """)

# Sidebar navigation
st.sidebar.title("📈 Stock Scanner")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to:",
    [
        "🏠 Home",
        "📈 Scanner",
        "📊 Analyst Reports",
        "📈 Earnings Reports"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Production Version**")
st.sidebar.markdown("Daily Trading Tool")

# Page routing
if page == "🏠 Home":
    show_home_page()

elif page == "📈 Scanner":
    try:
        from pages.scanner import show
        show()
    except Exception as e:
        st.error(f"Error loading Scanner: {str(e)}")

elif page == "📊 Analyst Reports":
    try:
        from pages.analyst_reports_analysis import show
        show()
    except Exception as e:
        st.error(f"Error loading Analyst Reports: {str(e)}")

elif page == "📈 Earnings Reports":
    try:
        from pages.earnings_reports_analysis import show
        show()
    except Exception as e:
        st.error(f"Error loading Earnings Reports: {str(e)}")