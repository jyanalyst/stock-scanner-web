"""
Live Scanner Page
Real-time stock scanning functionality
Fixed function name references
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import time

def show():
    """Main scanner page display"""
    
    st.title("🔍 Live Stock Scanner")
    st.markdown("Real-time analysis of Singapore Exchange stocks using your proven strategy")
    
    # Check if core modules are available
    try:
        from core.data_fetcher import DataFetcher, set_global_data_fetcher, get_company_name
        from core.technical_analysis import add_enhanced_columns
        from utils.watchlist import get_active_watchlist
        modules_available = True
    except ImportError as e:
        st.error(f"❌ Import error: {e}")
        st.info("""
        **Missing modules detected!** Please create the following files:
        
        1. `core/data_fetcher.py` - Copy the enhanced data fetcher code
        2. `core/technical_analysis.py` - Copy the technical analysis code  
        3. `utils/watchlist.py` - Copy the watchlist code
        
        All code was provided in previous messages.
        """)
        
        # Show test mode instead
        if st.button("🧪 Run Test Mode"):
            test_scanner_simple()
        
        if 'scan_results' in st.session_state:
            display_scan_results(st.session_state.scan_results)
        
        return
    
    # Control Panel
    col1, col2, col3 = st.columns(3)
    
    with col1:
        scan_mode = st.selectbox(
            "Scan Mode", 
            ["Test Mode (5 stocks)", "Manual Scan", "Full Scan (46 stocks)"],
            help="Choose how many stocks to analyze"
        )
    
    with col2:
        days_back = st.number_input(
            "Days of Data", 
            min_value=30, 
            max_value=100, 
            value=59,
            help="How many days of historical data to analyze"
        )
    
    with col3:
        auto_refresh = st.checkbox(
            "Auto Refresh (5 min)",
            help="Automatically refresh results every 5 minutes"
        )
    
    # Scan Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Run Stock Scan", type="primary", use_container_width=True):
            if modules_available:
                run_stock_scan(scan_mode, days_back)
            else:
                st.warning("Please install required modules first.")
    
    # Test buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧪 Test Mode (Simple)"):
            test_scanner_simple()
    
    with col2:
        if st.button("🔬 Test Mode (Real yfinance)"):
            test_scanner_with_actual_yfinance()
    
    # Display last scan info if available
    if 'last_scan_time' in st.session_state:
        st.info(f"Last scan completed: {st.session_state.last_scan_time}")
    
    # Display results if available
    if 'scan_results' in st.session_state:
        display_scan_results(st.session_state.scan_results)
    else:
        st.markdown("""
        ### 📋 Ready to Scan
        
        Click the "🚀 Run Stock Scan" button above to start analyzing stocks using your proven strategy:
        
        - **Volume-weighted range analysis**
        - **CRT (Monday reference) levels** 
        - **IBS (Internal Bar Strength) calculations**
        - **Buy signal detection**
        - **Real company names** from yfinance
        
        The scanner will analyze each stock and identify potential buy opportunities based on your criteria.
        
        ---
        
        **🧪 Test Modes Available**: 
        - Simple test with sample data
        - Real yfinance test that fetches actual company names
        """)

def run_stock_scan(scan_mode: str, days_back: int):
    """Execute the stock scanning process"""
    
    # Import modules (already checked in show())
    from core.data_fetcher import DataFetcher, set_global_data_fetcher
    from core.technical_analysis import add_enhanced_columns
    from utils.watchlist import get_active_watchlist
    
    # Determine which stocks to scan
    all_stocks = get_active_watchlist()
    
    if scan_mode == "Test Mode (5 stocks)":
        stocks_to_scan = all_stocks[:5]  # First 5 stocks for testing
    elif scan_mode == "Manual Scan":
        stocks_to_scan = all_stocks
    else:  # Full scan
        stocks_to_scan = all_stocks
    
    st.info(f"🔄 Scanning {len(stocks_to_scan)} stocks... This may take a moment.")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize data fetcher
        fetcher = DataFetcher(days_back=days_back)
        
        # Download stock data (this also fetches company names)
        status_text.text("📥 Downloading stock data and company names...")
        progress_bar.progress(0.2)
        
        stock_data = fetcher.download_stock_data(stocks_to_scan)
        
        # Set global fetcher so company names are available everywhere
        set_global_data_fetcher(fetcher)
        
        if not stock_data:
            st.error("❌ Failed to download stock data. Please check your internet connection.")
            return
        
        # Process each stock
        status_text.text("🔄 Analyzing technical indicators...")
        progress_bar.progress(0.4)
        
        results = []
        for i, (ticker, df_raw) in enumerate(stock_data.items()):
            try:
                # Apply technical analysis
                df_enhanced = add_enhanced_columns(df_raw, ticker)
                
                # Get latest row for results
                latest_row = df_enhanced.iloc[-1]
                
                # Get real company name
                company_name = fetcher.get_company_name(ticker)
                
                # Collect results
                result = {
                    'Ticker': ticker,
                    'Name': company_name,  # Real company name from yfinance
                    'Date': latest_row.name.strftime('%Y-%m-%d'),
                    'Close': round(latest_row['Close'], 2),
                    'IBS': round(latest_row['IBS'], 3) if not pd.isna(latest_row['IBS']) else 0,
                    'Valid_CRT': int(latest_row.get('Valid_CRT', 0)),
                    'Wick_Below': int(latest_row.get('Wick_Below', 0)),
                    'Close_Above': int(latest_row.get('Close_Above', 0)),
                    'Buy_Signal': int(latest_row.get('Buy_Signal', 0)),
                    'Rel_Range_Signal': int(latest_row.get('Rel_Range_Signal', 0)),
                    'VW_Range_Percentile': round(latest_row.get('VW_Range_Percentile', 0), 4)
                }
                results.append(result)
                
                # Update progress
                progress = 0.4 + (0.5 * (i + 1) / len(stock_data))
                progress_bar.progress(progress)
                
            except Exception as e:
                st.warning(f"⚠️ Error processing {ticker}: {str(e)}")
                continue
        
        # Finalize results
        status_text.text("📊 Preparing results...")
        progress_bar.progress(0.9)
        
        results_df = pd.DataFrame(results)
        
        # Store results in session state
        st.session_state.scan_results = results_df
        st.session_state.last_scan_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Complete
        progress_bar.progress(1.0)
        status_text.text("✅ Scan completed!")
        
        # Clear progress indicators after a moment
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"🎉 Scan completed! Analyzed {len(results_df)} stocks with real company names.")
        
        # Auto-refresh the page to show results
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Scan failed: {str(e)}")
        st.info("This might be due to missing modules or network issues.")
        progress_bar.empty()
        status_text.empty()

def display_scan_results(results_df: pd.DataFrame):
    """Display the scanning results in an organized format"""
    
    if results_df.empty:
        st.warning("No results to display.")
        return
    
    # Summary metrics
    st.subheader("📊 Scan Summary")
    
    total_stocks = len(results_df)
    buy_signals = len(results_df[results_df['Buy_Signal'] == 1])
    expansion_signals = len(results_df[results_df['Rel_Range_Signal'] == 1])
    high_ibs = len(results_df[results_df['IBS'] >= 0.5])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Scanned", total_stocks)
    with col2:
        st.metric("Buy Signals", buy_signals, delta=f"{buy_signals/total_stocks*100:.1f}%")
    with col3:
        st.metric("Range Expansion", expansion_signals)
    with col4:
        st.metric("High IBS (≥0.5)", high_ibs)
    
    # Buy Signals Section
    st.subheader("🎯 Buy Signals Detected")
    
    buy_signals_df = results_df[results_df['Buy_Signal'] == 1].copy()
    
    if len(buy_signals_df) > 0:
        # Add signal description
        buy_signals_df['Signal_Type'] = buy_signals_df.apply(
            lambda row: get_signal_description(row), axis=1
        )
        
        # Display buy signals in a nice format
        for _, stock in buy_signals_df.iterrows():
            with st.container():
                st.markdown(f"""
                <div style="background-color: #e8f5e8; padding: 15px; border-radius: 10px; border-left: 4px solid #2E8B57; margin: 10px 0;">
                    <h4 style="margin: 0; color: #2E8B57;">📈 {stock['Ticker']} - {stock['Name']}</h4>
                    <p style="margin: 5px 0; font-size: 18px; font-weight: bold;">${stock['Close']:.2f}</p>
                    <p style="margin: 5px 0;">IBS: {stock['IBS']:.3f} | Signal: {stock['Signal_Type']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Detailed table
        st.subheader("📋 Detailed Buy Signals")
        display_cols = ['Ticker', 'Name', 'Close', 'IBS', 'Signal_Type', 'Valid_CRT']
        st.dataframe(
            buy_signals_df[display_cols],
            use_container_width=True,
            hide_index=True
        )
        
    else:
        st.info("🔍 No buy signals detected in current scan. Keep monitoring for opportunities!")
    
    # Watch List Section
    st.subheader("👀 Range Expansion Watch List")
    
    expansion_df = results_df[
        (results_df['Rel_Range_Signal'] == 1) & (results_df['Buy_Signal'] == 0)
    ].copy()
    
    if len(expansion_df) > 0:
        for _, stock in expansion_df.iterrows():
            st.markdown(f"""
            <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; border-left: 4px solid #ffc107; margin: 5px 0;">
                <strong>📊 {stock['Ticker']} - {stock['Name']}</strong> - ${stock['Close']:.2f}<br>
                <small>IBS: {stock['IBS']:.3f} | Range expansion detected</small>
            </div>
            """, unsafe_allow_html=True)
        
        display_cols = ['Ticker', 'Name', 'Close', 'IBS', 'VW_Range_Percentile']
        st.dataframe(
            expansion_df[display_cols],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No range expansion signals detected.")
    
    # Full Results Table
    with st.expander("📋 Full Scan Results", expanded=False):
        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True
        )

def get_signal_description(row):
    """Get human-readable signal description"""
    signals = []
    if row['Wick_Below']:
        signals.append('Wick Below')
    if row['Close_Above']:
        signals.append('Close Above')
    
    return ' + '.join(signals) if signals else 'None'

def test_scanner_simple():
    """Simple test mode with sample data"""
    st.info("🧪 Running simple test mode...")
    
    # Basic sample data for interface testing
    sample_data = {
        'Ticker': ['A17U.SI', 'BN2.SI', 'C52.SI', 'E28.SI', 'G13.SI'],
        'Name': ['Ascendas REIT', 'Valuetronics', 'ComfortDelGro', 'First Resources', 'Genting Singapore'],
        'Date': ['2025-09-06'] * 5,
        'Close': [2.77, 0.77, 1.48, 1.42, 0.76],
        'IBS': [0.667, 1.000, 1.000, 0.667, 0.750],
        'Valid_CRT': [1, 1, 1, 1, 1],
        'Wick_Below': [0, 1, 0, 0, 0],
        'Close_Above': [1, 1, 1, 1, 1],
        'Buy_Signal': [1, 1, 1, 1, 1],
        'Rel_Range_Signal': [0, 1, 0, 0, 0],
        'VW_Range_Percentile': [0.4523, 0.6789, 0.4123, 0.3241, 0.3456]
    }
    
    results_df = pd.DataFrame(sample_data)
    st.session_state.scan_results = results_df
    st.session_state.last_scan_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    st.success("✅ Simple test data loaded!")
    st.rerun()

def test_scanner_with_actual_yfinance():
    """Test the scanner by actually fetching real company names from yfinance"""
    st.info("🔬 Running test mode with REAL company names from yfinance...")
    
    # Test with a small subset of stocks to get real names
    test_tickers = ['A17U.SI', 'BN2.SI', 'C52.SI', 'E28.SI', 'G13.SI']
    
    try:
        import yfinance as yf
        
        real_names = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, ticker in enumerate(test_tickers):
            try:
                status_text.text(f"Fetching real name for {ticker}...")
                progress_bar.progress((i + 1) / len(test_tickers))
                
                ticker_obj = yf.Ticker(ticker)
                info = ticker_obj.info
                
                # Get real company name
                company_name = (
                    info.get('longName') or 
                    info.get('shortName') or 
                    info.get('displayName') or
                    ticker.replace('.SI', '')
                )
                real_names[ticker] = company_name
                
            except Exception as e:
                real_names[ticker] = ticker.replace('.SI', '')
                st.warning(f"Could not fetch name for {ticker}: {e}")
        
        progress_bar.empty()
        status_text.empty()
        
        # Create sample data with REAL company names
        sample_data = {
            'Ticker': list(real_names.keys()),
            'Name': list(real_names.values()),
            'Date': ['2025-09-06'] * len(real_names),
            'Close': [2.77, 0.77, 1.48, 1.42, 0.76],
            'IBS': [0.667, 1.000, 1.000, 0.667, 0.750],
            'Valid_CRT': [1, 1, 1, 1, 1],
            'Wick_Below': [0, 1, 0, 0, 0],
            'Close_Above': [1, 1, 1, 1, 1],
            'Buy_Signal': [1, 1, 1, 1, 1],
            'Rel_Range_Signal': [0, 1, 0, 0, 0],
            'VW_Range_Percentile': [0.4523, 0.6789, 0.4123, 0.3241, 0.3456]
        }
        
        results_df = pd.DataFrame(sample_data)
        st.session_state.scan_results = results_df
        st.session_state.last_scan_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        st.success("✅ Test data loaded with REAL company names from yfinance!")
        
        # Show what we found
        st.info("**Real company names fetched:**")
        for ticker, name in real_names.items():
            st.write(f"• {ticker} → {name}")
        
        st.rerun()
        
    except ImportError:
        st.error("yfinance not installed. Install with: pip install yfinance")
    except Exception as e:
        st.error(f"Failed to fetch real names: {e}")
        st.info("Using fallback to simple test mode...")
        test_scanner_simple()

if __name__ == "__main__":
    show()