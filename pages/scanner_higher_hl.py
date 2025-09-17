"""
CRT Higher H/L Scanner - PART 1 (Utility Functions)
REDESIGNED WITH MPI STRATEGY FILTERING
Enhanced with MPI (Market Positivity Index) replacing complex momentum system
Column 4 now uses intuitive strategy-based filtering instead of percentiles
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta
import time
import traceback
import logging
import sys
from io import StringIO
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ErrorLogger:
    """Centralized error logging and display for debugging"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.debug_info = []
    
    def log_error(self, component: str, error: Exception, context: dict = None):
        """Log an error with full context"""
        error_info = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'component': component,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {}
        }
        self.errors.append(error_info)
        logger.error(f"{component}: {error_info['error_type']} - {error_info['error_message']}")
        return error_info
    
    def log_warning(self, component: str, message: str, context: dict = None):
        """Log a warning"""
        warning_info = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'component': component,
            'message': message,
            'context': context or {}
        }
        self.warnings.append(warning_info)
        logger.warning(f"{component}: {message}")
        return warning_info
    
    def log_debug(self, component: str, message: str, data: any = None):
        """Log debug information"""
        debug_info = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'component': component,
            'message': message,
            'data': str(data) if data is not None else None
        }
        self.debug_info.append(debug_info)
        logger.info(f"DEBUG {component}: {message}")
        return debug_info
    
    def display_errors_in_streamlit(self):
        """Display all logged errors in Streamlit interface"""
        if self.errors:
            st.error(f"❌ {len(self.errors)} error(s) occurred during execution")
            
            with st.expander("🔍 View Error Details", expanded=True):
                for i, error in enumerate(self.errors, 1):
                    st.markdown(f"**Error {i}: {error['component']}**")
                    st.code(f"Type: {error['error_type']}\nMessage: {error['error_message']}")
                    
                    if error['context']:
                        st.json(error['context'])
                    
                    with st.expander(f"Full Traceback - Error {i}"):
                        st.code(error['traceback'], language='python')
                    
                    st.markdown("---")
        
        if self.warnings:
            st.warning(f"⚠️ {len(self.warnings)} warning(s) occurred")
            
            with st.expander("📋 View Warnings"):
                for warning in self.warnings:
                    st.write(f"**{warning['component']}**: {warning['message']}")
                    if warning['context']:
                        st.json(warning['context'])
    
    def get_summary(self):
        """Get a summary of logged issues"""
        return {
            'errors': len(self.errors),
            'warnings': len(self.warnings),
            'debug_entries': len(self.debug_info)
        }

# Initialize global error logger
if 'error_logger' not in st.session_state:
    st.session_state.error_logger = ErrorLogger()

def get_price_format(price: float) -> str:
    """
    Get the appropriate price format string based on price level
    Returns format string for st.column_config.NumberColumn
    """
    if price < 1.00:
        return "$%.3f"  # 3 decimal places for stocks under $1.00
    else:
        return "$%.2f"  # 2 decimal places for regular stocks

def get_dynamic_column_config(df: pd.DataFrame, display_cols: list, base_config: dict) -> dict:
    """
    Create dynamic column configuration with price-based decimal formatting
    """
    config = base_config.copy()
    
    # Identify price columns that need dynamic formatting
    price_columns = ['Close', 'High', 'Low', 'CRT_High', 'CRT_Low', 'Weekly_Open']
    
    for col in price_columns:
        if col in display_cols and col in df.columns:
            # Check if we have price data to determine format
            if len(df) > 0:
                # Use the minimum price in the column to determine format
                min_price = df[col].min()
                if min_price < 1.00:
                    # At least one stock is under $1, use 3 decimals for all
                    config[col] = st.column_config.NumberColumn(
                        config[col].title if col in config and hasattr(config[col], 'title') else col.replace('_', ' ').title(),
                        format="$%.3f"
                    )
                else:
                    # All stocks are $1.00 or above, use 2 decimals
                    config[col] = st.column_config.NumberColumn(
                        config[col].title if col in config and hasattr(config[col], 'title') else col.replace('_', ' ').title(),
                        format="$%.2f"
                    )
    
    return config

def format_mpi_visual(mpi_value: float) -> str:
    """
    Convert MPI to visual blocks for intuitive display
    """
    if pd.isna(mpi_value):
        return "░░░░░░░░░░"
    
    blocks = max(0, min(10, int(mpi_value * 10)))  # Ensure 0-10 range
    return "█" * blocks + "░" * (10 - blocks)

def get_mpi_strategy_zone(mpi_value: float) -> dict:
def get_mpi_strategy_zone(mpi_value: float) -> Dict[str, str]:
    """
    Determine MPI strategy zone and characteristics
    UPDATED: Now maps to MPI_State values for consistency
    """
    if pd.isna(mpi_value):
        return {
            'zone': 'Unknown',
            'emoji': '❓',
            'color': 'gray',
            'interpretation': 'Insufficient data',
            'action': 'Wait for more data'
        }
    
    # Map to MPI_State categories for consistency
    if mpi_value >= 0.70:
        return {
            'zone': 'Strong Bull',
            'emoji': '🚀',
            'color': 'darkgreen',
            'interpretation': f'{mpi_value:.0%} green days - Strong momentum',
            'action': 'Buy on dips, ride the trend'
        }
    elif mpi_value >= 0.50:
        return {
            'zone': 'Bull Trend', 
            'emoji': '📈',
            'color': 'green',
            'interpretation': f'{mpi_value:.0%} green days - Positive momentum',
            'action': 'Buy breakouts, hold positions'
        }
    elif mpi_value >= 0.30:
        return {
            'zone': 'Neutral',
            'emoji': '➖',
            'color': 'orange', 
            'interpretation': f'{mpi_value:.0%} green days - Mixed signals',
            'action': 'Wait for clearer direction'
        }
    else:
        return {
            'zone': 'Bear Trend',
            'emoji': '📉',
            'color': 'red',
            'interpretation': f'{mpi_value:.0%} green days - Weak momentum', 
            'action': 'Avoid longs, consider shorts'
        }

def apply_mpi_strategy_filter(base_stocks: pd.DataFrame, selected_states: list) -> pd.DataFrame:
    """
    Apply MPI strategy-based filtering using advanced MPI State Classification
    
    Args:
        base_stocks: Input DataFrame with MPI_State column
        selected_states: List of selected MPI states
    
    Returns:
        Filtered DataFrame sorted appropriately
    """
    if base_stocks.empty:
        return base_stocks
    
    if 'MPI_State' not in base_stocks.columns:
        st.warning("MPI_State column not found - using all stocks")
        return base_stocks
    
    # If no states selected, return all stocks
    if not selected_states:
        filtered = base_stocks.copy()
        return filtered.sort_values('MPI_Base', ascending=False)
    
    # Filter by selected MPI states using exact string matching
    filtered = base_stocks[base_stocks['MPI_State'].isin(selected_states)].copy()
    
    # Smart sorting based on selected states
    bear_states = ['Bear Market']
    neutral_states = ['Neutral Zone']
    bull_states = ['Strong Bull Rising', 'Strong Bull Slowing', 'Bull Acceleration', 'Bull Deceleration']
    
    selected_categories = []
    if any(state in bear_states for state in selected_states):
        selected_categories.append('bear')
    if any(state in neutral_states for state in selected_states):
        selected_categories.append('neutral')
    if any(state in bull_states for state in selected_states):
        selected_categories.append('bull')
    
    # Sort based on predominant selection
    if selected_categories == ['bear']:
        # For bear-only selections, sort weakest first (for shorting)
        filtered = filtered.sort_values('MPI_Base', ascending=True)
    else:
        # For all other combinations, sort strongest first
        # Secondary sort by acceleration for same MPI levels
        filtered = filtered.sort_values(['MPI_Base', 'MPI_Acceleration'], ascending=[False, False])
    
    return filtered

def apply_dynamic_filters(base_stocks, results_df):
    """
    Apply dynamic filtering with CRT Velocity, IBS, Higher H/L, and NEW MPI Strategy filters
    UPDATED: Column 4 now uses MPI Strategy Filter instead of complex momentum percentiles
    Returns filtered stocks
    """
    
    if base_stocks.empty:
        st.warning("No stocks available for filtering")
        return base_stocks
    
    st.subheader("🎯 Dynamic Filtering")
    
    # Create four columns for filters
    col1, col2, col3, col4 = st.columns(4)
    
    # Initialize filtered stocks
    filtered_stocks = base_stocks.copy()
    
    # COL 1: CRT VELOCITY PERCENTILE FILTER (UNCHANGED)
    with col1:
        st.markdown("**CRT Velocity Filter:**")
        
        # Get velocity statistics - handle both Valid CRT and non-Valid CRT stocks
        if 'CRT_Velocity' in base_stocks.columns:
            velocities = base_stocks['CRT_Velocity']
        else:
            velocities = base_stocks.index.map(lambda idx: results_df.loc[idx, 'CRT_Velocity'] if idx in results_df.index else 0)
            velocities = pd.Series(velocities, index=base_stocks.index)
        
        # Remove any zero velocities for percentile calculation
        non_zero_velocities = velocities[velocities != 0]
        
        if len(non_zero_velocities) > 0:
            percentile_options = {
                "Top 25%": 75,
                "Top 50%": 50, 
                "Top 75%": 25,
                "No Filter": None
            }
            
            selected_percentile = st.radio(
                "Select velocity filter:",
                list(percentile_options.keys()),
                key="crt_velocity_percentile_radio"
            )
            
            # Apply percentile filtering
            if selected_percentile != "No Filter":
                percentile_val = percentile_options[selected_percentile]
                threshold_value = np.percentile(non_zero_velocities, percentile_val)
                filtered_stocks = filtered_stocks[velocities >= threshold_value]
                st.info(f"CRT Velocity ≥ {threshold_value:+.4f} pp")
            else:
                st.info("All velocities included")
            
            # Show velocity statistics
            with st.expander("CRT Velocity Statistics", expanded=False):
                stats_data = {
                    "Metric": ["Count", "Min", "25th %ile", "Median", "75th %ile", "Max"],
                    "Value": [
                        f"{len(non_zero_velocities)}",
                        f"{non_zero_velocities.min():+.4f}",
                        f"{np.percentile(non_zero_velocities, 25):+.4f}",
                        f"{non_zero_velocities.median():+.4f}", 
                        f"{np.percentile(non_zero_velocities, 75):+.4f}",
                        f"{non_zero_velocities.max():+.4f}"
                    ],
                    "Unit": ["stocks", "pp", "pp", "pp", "pp", "pp"]
                }
                st.dataframe(pd.DataFrame(stats_data), hide_index=True, use_container_width=True)
        else:
            st.info("No CRT Velocity data available for filtering")
    
    # COL 2: IBS FILTER (UNCHANGED)
    with col2:
        st.markdown("**IBS Filter:**")
        
        # Get IBS values
        ibs_values = filtered_stocks['IBS']
        
        # Percentile options (standardized)
        ibs_percentile_options = {
            "Top 25%": 75,
            "Top 50%": 50,
            "Top 75%": 25,
            "Custom": "custom",
            "No Filter": None
        }
        
        selected_ibs_option = st.radio(
            "Select IBS filter:",
            list(ibs_percentile_options.keys()),
            key="ibs_percentile_radio"
        )
        
        # Handle custom input
        if selected_ibs_option == "Custom":
            custom_ibs_value = st.number_input(
                "Enter minimum IBS value:",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                format="%.2f",
                key="custom_ibs_input"
            )
            filtered_stocks = filtered_stocks[filtered_stocks['IBS'] >= custom_ibs_value]
            st.info(f"IBS ≥ {custom_ibs_value:.2f}")
        elif selected_ibs_option != "No Filter":
            # Percentile-based filtering
            percentile_val = ibs_percentile_options[selected_ibs_option]
            threshold_value = np.percentile(ibs_values, percentile_val)
            filtered_stocks = filtered_stocks[filtered_stocks['IBS'] >= threshold_value]
            st.info(f"IBS ≥ {threshold_value:.3f} ({selected_ibs_option})")
        else:
            st.info("All IBS values included")
        
        # Show IBS statistics
        with st.expander("IBS Statistics", expanded=False):
            ibs_stats = {
                "Metric": ["Count", "Min", "25th %ile", "Median", "75th %ile", "Max"],
                "Value": [
                    f"{len(ibs_values)}",
                    f"{ibs_values.min():.3f}",
                    f"{np.percentile(ibs_values, 25):.3f}",
                    f"{ibs_values.median():.3f}",
                    f"{np.percentile(ibs_values, 75):.3f}",
                    f"{ibs_values.max():.3f}"
                ],
                "Unit": ["stocks", "", "", "", "", ""]
            }
            st.dataframe(pd.DataFrame(ibs_stats), hide_index=True, use_container_width=True)
    
    # COL 3: HIGHER H/L FILTER (UNCHANGED)
    with col3:
        # Check if we need this filter
        base_filter_type = st.session_state.get('base_filter_type', 'All Stocks')
        
        if base_filter_type not in ["Valid CRT + Higher H/L", "Higher H/L Only"]:
            st.markdown("**Higher H/L Filter:**")
            
            higher_hl_options = {
                "Higher H/L Only": 1,
                "No Filter": None
            }
            
            selected_higher_hl = st.radio(
                "Select Higher H/L filter:",
                list(higher_hl_options.keys()),
                key="higher_hl_pattern_radio"
            )
            
            # Apply Higher_HL filtering
            if selected_higher_hl != "No Filter":
                filtered_stocks = filtered_stocks[filtered_stocks['Higher_HL'] == 1]
                st.info("Only showing Higher High/Low patterns")
            else:
                st.info("All patterns included")
            
            # Show Higher_HL statistics
            with st.expander("Higher H/L Statistics", expanded=False):
                higher_hl_count = (base_stocks['Higher_HL'] == 1).sum()
                total_count = len(base_stocks)
                higher_hl_pct = (higher_hl_count / total_count * 100) if total_count > 0 else 0
                
                hl_stats = {
                    "Pattern": ["Higher H/L", "Not Higher H/L", "Total"],
                    "Count": [
                        f"{higher_hl_count}",
                        f"{total_count - higher_hl_count}",
                        f"{total_count}"
                    ],
                    "Percentage": [
                        f"{higher_hl_pct:.1f}%",
                        f"{100 - higher_hl_pct:.1f}%",
                        "100.0%"
                    ]
                }
                st.dataframe(pd.DataFrame(hl_stats), hide_index=True, use_container_width=True)
        else:
            st.markdown("**Additional Filters:**")
            st.info("Higher H/L filter not needed - already filtered by base selection")
            selected_higher_hl = "No Filter"
    
    # COL 4: MULTI-SELECT MPI STATE FILTER (ADVANCED CLASSIFICATION)
    with col4:
        st.markdown("**MPI Strategy Filter:**")
        
        # Check if MPI data is available
        if 'MPI_State' in filtered_stocks.columns:
            
            # MPI State options with advanced classification
            strategy_options = [
                "🚀 Strong Bull Rising",      # MPI≥70% + Acceleration>0.1
                "📈 Strong Bull Slowing",     # MPI≥70% + Acceleration≤0.1  
                "⚡ Bull Acceleration",       # MPI≥50% + Acceleration>0.1
                "📊 Bull Deceleration",       # MPI≥50% + Acceleration≤0.1
                "➖ Neutral Zone",            # MPI 30-50%
                "📉 Bear Market"              # MPI<30%
            ]
            
            # Multi-select checkboxes with advanced states
            st.markdown("Select momentum states:")
            selected_strategies = []
            
            for i, strategy in enumerate(strategy_options):
                # Map display names to internal MPI_State values
                state_mapping = {
                    "🚀 Strong Bull Rising": "Strong Bull Rising",
                    "📈 Strong Bull Slowing": "Strong Bull Slowing", 
                    "⚡ Bull Acceleration": "Bull Acceleration",
                    "📊 Bull Deceleration": "Bull Deceleration",
                    "➖ Neutral Zone": "Neutral Zone",
                    "📉 Bear Market": "Bear Market"
                }
                
                if st.checkbox(strategy, key=f"mpi_state_checkbox_{i}"):
                    selected_strategies.append(state_mapping[strategy])
            
            # Apply MPI state filtering with advanced classification
            if selected_strategies:
                filtered_stocks = apply_mpi_strategy_filter(filtered_stocks, selected_strategies)
                
                # Show info about selected strategies with advanced details
                display_names = [k for k, v in state_mapping.items() if v in selected_strategies]
                st.info(f"Selected: {', '.join([name.split(' ')[0] for name in display_names])} ({len(filtered_stocks)} stocks)")
                
                # Show detailed explanations for selected states
                with st.expander("Selected State Details", expanded=False):
                    for strategy in selected_strategies:
                        if strategy == "Strong Bull Rising":
                            st.write("🚀 **Strong Bull Rising:** MPI ≥ 70% + Acceleration > 10% (Breakout momentum)")
                        elif strategy == "Strong Bull Slowing":
                            st.write("📈 **Strong Bull Slowing:** MPI ≥ 70% + Acceleration ≤ 10% (Strong but cooling)")
                        elif strategy == "Bull Acceleration":
                            st.write("⚡ **Bull Acceleration:** MPI ≥ 50% + Acceleration > 10% (Building momentum)")
                        elif strategy == "Bull Deceleration":
                            st.write("📊 **Bull Deceleration:** MPI ≥ 50% + Acceleration ≤ 10% (Slowing bull trend)")
                        elif strategy == "Neutral Zone":
                            st.write("➖ **Neutral Zone:** MPI 30-50% (Mixed signals)")
                        elif strategy == "Bear Market":
                            st.write("📉 **Bear Market:** MPI < 30% (Weak momentum)")
            else:
                st.info("No states selected - showing all MPI levels")
            
            # Show MPI State statistics and distribution
            with st.expander("MPI State Statistics", expanded=False):
                if 'MPI_State' in base_stocks.columns:
                    state_counts = base_stocks['MPI_State'].value_counts()
                    
                    # Create comprehensive state statistics
                    state_stats_data = []
                    state_order = [
                        'Strong Bull Rising', 'Strong Bull Slowing', 
                        'Bull Acceleration', 'Bull Deceleration',
                        'Neutral Zone', 'Bear Market'
                    ]
                    
                    for state in state_order:
                        count = state_counts.get(state, 0)
                        if count > 0:
                            state_stocks = base_stocks[base_stocks['MPI_State'] == state]
                            avg_mpi = state_stocks['MPI_Base'].mean()
                            avg_accel = state_stocks['MPI_Acceleration'].mean()
                            
                            state_stats_data.append({
                                "MPI State": state,
                                "Count": f"{count}",
                                "Avg MPI": f"{avg_mpi:.1%}",
                                "Avg Accel": f"{avg_accel:+.3f}",
                                "Description": get_state_description(state)
                            })
                    
                    if state_stats_data:
                        st.dataframe(pd.DataFrame(state_stats_data), hide_index=True, use_container_width=True)
                else:
                    st.warning("MPI_State column not available")
        else:
            st.warning("MPI state filtering not available - MPI_State data missing")
            selected_strategies = []
    
    # Show combined filter summary
    filter_summary = []
    if 'selected_percentile' in locals() and selected_percentile != "No Filter" and len(non_zero_velocities) > 0:
        filter_summary.append(f"CRT Velocity {selected_percentile}")
    if selected_ibs_option == "Custom":
        filter_summary.append(f"IBS ≥ {custom_ibs_value:.2f}")
    elif selected_ibs_option != "No Filter":
        filter_summary.append(f"IBS {selected_ibs_option}")
    if 'selected_higher_hl' in locals() and selected_higher_hl != "No Filter":
        filter_summary.append("Higher H/L Only")
    if selected_strategies:
        # Show count of selected MPI states
        filter_summary.append(f"MPI: {len(selected_strategies)} states")
    
    if filter_summary:
        st.success(f"Active filters: {' + '.join(filter_summary)} → {len(filtered_stocks)} stocks")
    else:
        st.info(f"No additional filters applied → {len(filtered_stocks)} stocks")
    
    # Show distribution charts
    with st.expander("📊 Distribution Analysis", expanded=False):
        charts_to_show = []
        
        # Velocity chart (if available)
        if 'non_zero_velocities' in locals() and len(non_zero_velocities) > 0:
            charts_to_show.append(('velocity', non_zero_velocities, selected_percentile if 'selected_percentile' in locals() else None))
        
        # IBS chart
        charts_to_show.append(('ibs', filtered_stocks['IBS'] if not filtered_stocks.empty else pd.Series(), selected_ibs_option))
        
        # MPI chart (if available)
        if 'MPI_Base' in base_stocks.columns:
            charts_to_show.append(('mpi', base_stocks['MPI_Base'], selected_strategies))
        
        # Create charts in rows of 2
        num_charts = len(charts_to_show)
        for i in range(0, num_charts, 2):
            cols = st.columns(2)
            for j, (chart_type, data, filter_selection) in enumerate(charts_to_show[i:i+2]):
                with cols[j]:
                    if chart_type == 'velocity' and len(data) > 0:
                        fig = px.histogram(
                            x=data,
                            nbins=20,
                            title="CRT Velocity Distribution",
                            labels={'x': 'CRT Velocity (pp)', 'y': 'Count'}
                        )
                        if filter_selection and filter_selection != "No Filter":
                            percentile_val = percentile_options[filter_selection]
                            threshold_value = np.percentile(non_zero_velocities, percentile_val)
                            fig.add_vline(x=threshold_value, line_dash="dash", line_color="red", 
                                         annotation_text=f"{filter_selection} threshold")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif chart_type == 'ibs' and len(data) > 0:
                        fig = px.histogram(
                            x=data,
                            nbins=20,
                            title="IBS Distribution",
                            labels={'x': 'IBS', 'y': 'Count'}
                        )
                        if filter_selection == "Custom":
                            if 'custom_ibs_value' in locals():
                                fig.add_vline(x=custom_ibs_value, line_dash="dash", line_color="red",
                                            annotation_text=f"Custom ≥ {custom_ibs_value:.2f}")
                        elif filter_selection != "No Filter" and filter_selection in ibs_percentile_options:
                            percentile_val = ibs_percentile_options[filter_selection]
                            if percentile_val is not None:
                                threshold_value = np.percentile(data, percentile_val)
                                fig.add_vline(x=threshold_value, line_dash="dash", line_color="red",
                                            annotation_text=f"{filter_selection} threshold")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif chart_type == 'mpi' and len(data) > 0:
                        fig = px.histogram(
                            x=data,
                            nbins=20,
                            title="MPI Strategy Distribution",
                            labels={'x': 'MPI Base (Market Positivity Index)', 'y': 'Count'}
                        )
                        
                        # Add strategy zone shading
                        fig.add_vrect(x0=0.70, x1=1.00, fillcolor="green", opacity=0.2, annotation_text="Strong Bull")
                        fig.add_vrect(x0=0.50, x1=0.70, fillcolor="lightgreen", opacity=0.2, annotation_text="Bull Trend")  
                        fig.add_vrect(x0=0.30, x1=0.50, fillcolor="orange", opacity=0.2, annotation_text="Neutral")
                        fig.add_vrect(x0=0.00, x1=0.30, fillcolor="red", opacity=0.2, annotation_text="Bear Trend")
                        
                        # Highlight selected MPI states on distribution chart
                        if selected_strategies:
                            # Add different colored lines for each selected state
                            colors = ['darkgreen', 'green', 'blue', 'orange', 'gray', 'red']
                            for i, strategy in enumerate(selected_strategies):
                                color = colors[i % len(colors)]
                                
                                if strategy == "Strong Bull Rising":
                                    fig.add_vline(x=0.70, line_dash="solid", line_color=color, line_width=3,
                                                annotation_text="Strong Bull Rising")
                                elif strategy == "Strong Bull Slowing":
                                    fig.add_vline(x=0.70, line_dash="dash", line_color=color, line_width=2,
                                                annotation_text="Strong Bull Slowing")
                                elif strategy == "Bull Acceleration":
                                    fig.add_vline(x=0.50, line_dash="solid", line_color=color, line_width=2,
                                                annotation_text="Bull Acceleration")
                                elif strategy == "Bull Deceleration":
                                    fig.add_vline(x=0.50, line_dash="dash", line_color=color, line_width=2,
                                                annotation_text="Bull Deceleration")
                                elif strategy == "Neutral Zone":
                                    fig.add_vline(x=0.40, line_dash="dot", line_color=color, line_width=2,
                                                annotation_text="Neutral Zone")
                                elif strategy == "Bear Market":
                                    fig.add_vline(x=0.30, line_dash="solid", line_color=color, line_width=3,
                                                annotation_text="Bear Market")
                        
                        st.plotly_chart(fig, use_container_width=True)
    
    return filtered_stocks

def show():
    """Main scanner page display for Higher H/L patterns with MPI Strategy filtering"""
    
    st.title("📈 CRT Higher H/L Scanner")
    st.markdown("Enhanced with **MPI Strategy Filtering** - Market Positivity Index for intuitive momentum analysis")
    
    # Clear previous errors for new scan
    if st.button("🗑️ Clear Error Log"):
        st.session_state.error_logger = ErrorLogger()
        st.success("Error log cleared!")
        st.rerun()
    
    # Check if core modules are available
    try:
        from core.data_fetcher import DataFetcher, set_global_data_fetcher, get_company_name
        from core.technical_analysis import add_enhanced_columns
        from utils.watchlist import get_active_watchlist
        modules_available = True
        st.session_state.error_logger.log_debug("Module Check", "All core modules imported successfully")
    except ImportError as e:
        st.session_state.error_logger.log_error("Module Import", e, {
            'missing_modules': ['core.data_fetcher', 'core.technical_analysis', 'utils.watchlist'],
            'python_path': sys.path
        })
        st.error(f"❌ Import error: {e}")
        modules_available = False
    
    # Display any existing errors
    st.session_state.error_logger.display_errors_in_streamlit()
    
    # Scanning Configuration Panel
    st.subheader("🎯 Scanning Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Scan Scope**")
        scan_scope = st.radio(
            "Choose scan scope:",
            ["Single Stock", "Full Watchlist"],
            help="Select whether to scan one stock or the entire watchlist"
        )
        
        if scan_scope == "Single Stock":
            try:
                if modules_available:
                    watchlist = get_active_watchlist()
                    st.session_state.error_logger.log_debug("Watchlist", f"Loaded {len(watchlist)} stocks", watchlist)
                else:
                    watchlist = ['A17U.SI', 'BN2.SI', 'C52.SI', 'E28.SI', 'G13.SI']
                    st.session_state.error_logger.log_warning("Watchlist", "Using fallback watchlist due to missing modules")
                
                selected_stock = st.selectbox(
                    "Select Stock:",
                    options=watchlist,
                    help="Choose which stock to analyze"
                )
                st.session_state.error_logger.log_debug("Stock Selection", f"Selected: {selected_stock}")
                
            except Exception as e:
                st.session_state.error_logger.log_error("Watchlist Loading", e)
                selected_stock = "A17U.SI"
        
    with col2:
        st.markdown("**📅 Analysis Date**")
        scan_date_type = st.radio(
            "Choose analysis date:",
            ["Current Date", "Historical Date"],
            help="Scan as of current date or specify a historical date"
        )
        
        if scan_date_type == "Historical Date":
            try:
                default_date = date.today() - timedelta(days=7)
                historical_date = st.date_input(
                    "Analysis Date:",
                    value=default_date,
                    max_value=date.today() - timedelta(days=1),
                    help="Choose the historical date for analysis (must be a past trading day)"
                )
                st.session_state.error_logger.log_debug("Date Selection", f"Historical date: {historical_date}")
            except Exception as e:
                st.session_state.error_logger.log_error("Date Selection", e)
                historical_date = date.today() - timedelta(days=7)
    
    # Advanced Settings
    with st.expander("⚙️ Advanced Settings"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            days_back = st.number_input(
                "Days of Historical Data", 
                min_value=30, 
                max_value=250, 
                value=59,
                help="How many days of data to download for analysis"
            )
        
        with col2:
            rolling_window = st.number_input(
                "Rolling Window", 
                min_value=10, 
                max_value=50, 
                value=20,
                help="Rolling window for technical calculations"
            )
        
        with col3:
            debug_mode = st.checkbox(
                "Enable Debug Mode",
                value=False,
                help="Show detailed debug information during scanning"
            )
    
    # Scan Execution
    st.subheader("🚀 Execute Scan")
    
    if st.button("🚀 Execute Scan", type="primary", use_container_width=True):
        st.session_state.error_logger = ErrorLogger()
        
        if modules_available:
            try:
                if scan_scope == "Single Stock":
                    stocks_to_scan = [selected_stock]
                else:
                    stocks_to_scan = get_active_watchlist()
                
                if scan_date_type == "Historical Date":
                    analysis_date = historical_date
                else:
                    analysis_date = None
                
                run_enhanced_stock_scan(
                    stocks_to_scan=stocks_to_scan,
                    analysis_date=analysis_date,
                    days_back=days_back,
                    rolling_window=rolling_window,
                    debug_mode=debug_mode
                )
                
            except Exception as e:
                st.session_state.error_logger.log_error("Scan Execution", e, {
                    "scan_scope": scan_scope,
                    "scan_date_type": scan_date_type,
                    "selected_stock": selected_stock if scan_scope == "Single Stock" else None
                })
                st.error("❌ Failed to execute scan - check error details above")
        else:
            st.warning("Cannot execute scan - required modules are not available")
    
    # Display last scan info if available
    if 'last_scan_time' in st.session_state:
        st.info(f"📊 Last scan completed: {st.session_state.last_scan_time}")
        
        if 'last_scan_config' in st.session_state:
            config = st.session_state.last_scan_config
            st.caption(f"Scope: {config['scope']} | Date: {config['date']} | Stocks: {config['stock_count']}")
    
    # Display results if available
    if 'scan_results' in st.session_state:
        display_scan_results(st.session_state.scan_results)

def run_enhanced_stock_scan(stocks_to_scan, analysis_date=None, days_back=59, rolling_window=20, debug_mode=False):
    """Execute the enhanced stock scanning process with MPI system integration"""
    
    error_logger = st.session_state.error_logger
    
    try:
        from core.data_fetcher import DataFetcher, set_global_data_fetcher
        from core.technical_analysis import add_enhanced_columns
        
        error_logger.log_debug("Scan Start", "Starting enhanced stock scan with MPI system", {
            "stocks_count": len(stocks_to_scan),
            "stocks": stocks_to_scan,
            "analysis_date": str(analysis_date) if analysis_date else "Current",
            "days_back": days_back,
            "rolling_window": rolling_window
        })
        
        is_historical = analysis_date is not None
        is_single_stock = len(stocks_to_scan) == 1
        
        if is_single_stock:
            scope_text = f"single stock ({stocks_to_scan[0]})"
        else:
            scope_text = f"{len(stocks_to_scan)} stocks"
        
        if is_historical:
            date_text = f"historical analysis (as of {analysis_date})"
        else:
            date_text = "current data analysis"
        
        st.info(f"🔄 Scanning {scope_text} with {date_text}... Calculating MPI indicators...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔧 Initializing data fetcher...")
        try:
            fetcher = DataFetcher(days_back=days_back)
            error_logger.log_debug("Data Fetcher", f"Initialized with {days_back} days back")
        except Exception as e:
            error_logger.log_error("Data Fetcher Initialization", e, {"days_back": days_back})
            raise
        
        progress_bar.progress(0.1)
        
        status_text.text("📥 Downloading stock data and company names...")
        try:
            stock_data = fetcher.download_stock_data(stocks_to_scan)
            error_logger.log_debug("Data Download", f"Downloaded data for {len(stock_data)} stocks", {
                "requested_stocks": len(stocks_to_scan),
                "successful_downloads": len(stock_data),
                "success_rate": f"{len(stock_data)/len(stocks_to_scan)*100:.1f}%"
            })
            
            missing_stocks = [stock for stock in stocks_to_scan if stock not in stock_data]
            if missing_stocks:
                error_logger.log_warning("Data Download", f"Failed to download {len(missing_stocks)} stocks", {
                    "missing_stocks": missing_stocks
                })
                
        except Exception as e:
            error_logger.log_error("Data Download", e, {
                "stocks_to_scan": stocks_to_scan,
                "fetcher_days_back": days_back
            })
            raise
        
        set_global_data_fetcher(fetcher)
        progress_bar.progress(0.3)
        
        if not stock_data:
            error_msg = "No stock data was downloaded successfully"
            error_logger.log_error("Data Validation", Exception(error_msg), {
                "stocks_requested": stocks_to_scan,
                "data_received": stock_data
            })
            st.error("❌ Failed to download stock data. Check error log for details.")
            return
        
        status_text.text("🔄 Calculating MPI indicators and technical analysis...")
        progress_bar.progress(0.4)
        
        results = []
        processing_errors = []
        
        for i, (ticker, df_raw) in enumerate(stock_data.items()):
            try:
                error_logger.log_debug("Stock Processing", f"Processing {ticker} with MPI system", {
                    "data_shape": df_raw.shape,
                    "date_range": f"{df_raw.index[0]} to {df_raw.index[-1]}" if len(df_raw) > 0 else "No data",
                    "columns": list(df_raw.columns)
                })
                
                if df_raw.empty:
                    error_logger.log_warning("Stock Processing", f"Empty dataframe for {ticker}")
                    continue
                
                try:
                    df_enhanced = add_enhanced_columns(df_raw, ticker, rolling_window)
                    error_logger.log_debug("Technical Analysis", f"Enhanced columns with MPI added for {ticker}", {
                        "enhanced_shape": df_enhanced.shape,
                        "mpi_columns": [col for col in df_enhanced.columns if 'MPI' in col],
                        "has_mpi_base": 'MPI_Base' in df_enhanced.columns
                    })
                except Exception as e:
                    error_logger.log_error("Technical Analysis", e, {
                        "ticker": ticker,
                        "raw_data_shape": df_raw.shape,
                        "rolling_window": rolling_window
                    })
                    processing_errors.append(f"{ticker}: Technical analysis failed")
                    continue
                
                if is_historical:
                    try:
                        target_date = pd.to_datetime(analysis_date)
                        available_dates = df_enhanced.index
                        
                        if hasattr(available_dates, 'tz') and available_dates.tz is not None:
                            if target_date.tz is None:
                                target_date = target_date.tz_localize('Asia/Singapore')
                            else:
                                target_date = target_date.tz_convert(available_dates.tz)
                        else:
                            if target_date.tz is not None:
                                target_date = target_date.tz_localize(None)
                        
                        valid_dates = available_dates[available_dates <= target_date]
                        
                        error_logger.log_debug("Historical Date Processing", f"Date comparison for {ticker}", {
                            "target_date": str(target_date),
                            "available_dates_count": len(available_dates),
                            "valid_dates_count": len(valid_dates)
                        })
                        
                        if len(valid_dates) == 0:
                            error_logger.log_warning("Historical Analysis", f"No data for {ticker} on or before {analysis_date}", {
                                "target_date": str(target_date),
                                "available_date_range": f"{available_dates[0]} to {available_dates[-1]}" if len(available_dates) > 0 else "No dates"
                            })
                            continue
                        
                        analysis_row = df_enhanced.loc[valid_dates[-1]]
                        actual_date = valid_dates[-1]
                        
                    except Exception as e:
                        error_logger.log_error("Historical Date Processing", e, {
                            "ticker": ticker,
                            "target_date": str(analysis_date)
                        })
                        continue
                else:
                    analysis_row = df_enhanced.iloc[-1]
                    actual_date = analysis_row.name
                
                try:
                    company_name = fetcher.get_company_name(ticker)
                    error_logger.log_debug("Company Name", f"Retrieved name for {ticker}: {company_name}")
                except Exception as e:
                    error_logger.log_warning("Company Name", f"Failed to get name for {ticker}: {e}")
                    company_name = ticker.replace('.SI', '')
                
                # Dynamic decimal formatting based on price level
                close_price = float(analysis_row['Close'])
                high_price = float(analysis_row['High'])
                low_price = float(analysis_row['Low'])
                
                if close_price < 1.00:
                    price_decimals = 3
                else:
                    price_decimals = 2
                
                def safe_round(value, decimals):
                    try:
                        return round(float(value), decimals) if not pd.isna(value) else 0
                    except:
                        return 0
                
                # Get MPI strategy zone information
                mpi_base = float(analysis_row.get('MPI_Base', 0.5)) if not pd.isna(analysis_row.get('MPI_Base', 0.5)) else 0.5
                mpi_zone_info = get_mpi_strategy_zone(mpi_base)
                
                # Collect results with MPI-ENHANCED data
                try:
                    result = {
                        'Ticker': ticker,
                        'Name': company_name,
                        'Analysis_Date': actual_date.strftime('%Y-%m-%d') if hasattr(actual_date, 'strftime') else str(actual_date),
                        'Close': round(close_price, price_decimals),
                        'High': round(high_price, price_decimals),
                        'Low': round(low_price, price_decimals),
                        'IBS': round(float(analysis_row['IBS']), 3) if not pd.isna(analysis_row['IBS']) else 0,
                        'Valid_CRT': int(analysis_row.get('Valid_CRT', 0)),
                        'Higher_HL': int(analysis_row.get('Higher_HL', 0)) if not pd.isna(analysis_row.get('Higher_HL', 0)) else 0,
                        'CRT_Velocity': round(float(analysis_row.get('CRT_Qualifying_Velocity', 0)), 4) if not pd.isna(analysis_row.get('CRT_Qualifying_Velocity', 0)) else 0,
                        'Weekly_Open': safe_round(analysis_row.get('Weekly_Open', 0), price_decimals),
                        'CRT_High': safe_round(analysis_row.get('CRT_High', 0), price_decimals),
                        'CRT_Low': safe_round(analysis_row.get('CRT_Low', 0), price_decimals),
                        'VW_Range_Percentile': round(float(analysis_row.get('VW_Range_Percentile', 0)), 4) if not pd.isna(analysis_row.get('VW_Range_Percentile', 0)) else 0,
                        'Rel_Range_Signal': int(analysis_row.get('Rel_Range_Signal', 0)),
                        
                        # NEW MPI SYSTEM COLUMNS (replacing old complex momentum)
                        'MPI_Base': round(mpi_base, 4),
                        'MPI_Fast': round(float(analysis_row.get('MPI_Fast', 0.5)), 4) if not pd.isna(analysis_row.get('MPI_Fast', 0.5)) else 0.5,
                        'MPI_Slow': round(float(analysis_row.get('MPI_Slow', 0.5)), 4) if not pd.isna(analysis_row.get('MPI_Slow', 0.5)) else 0.5,
                        'MPI_Acceleration': round(float(analysis_row.get('MPI_Acceleration', 0.0)), 4) if not pd.isna(analysis_row.get('MPI_Acceleration', 0.0)) else 0.0,
                        'MPI_State': str(analysis_row.get('MPI_State', 'Unknown')),
                        'MPI_Zone': mpi_zone_info['zone'],
                        'MPI_Zone_Emoji': mpi_zone_info['emoji'],
                        'MPI_Visual': format_mpi_visual(mpi_base),
                        
                        # MPI Strategy Signals
                        'Signal_Breakout': int(analysis_row.get('Signal_Breakout', 0)) if not pd.isna(analysis_row.get('Signal_Breakout', 0)) else 0,
                        'Signal_Pullback': int(analysis_row.get('Signal_Pullback', 0)) if not pd.isna(analysis_row.get('Signal_Pullback', 0)) else 0,
                        'Signal_Short': int(analysis_row.get('Signal_Short', 0)) if not pd.isna(analysis_row.get('Signal_Short', 0)) else 0,
                        
                        # Store price level for display formatting
                        'Price_Decimals': price_decimals
                    }
                    
                    results.append(result)
                    
                    if debug_mode:
                        error_logger.log_debug("Result Collection", f"Collected MPI result for {ticker}", {
                            "mpi_base": mpi_base,
                            "mpi_zone": mpi_zone_info['zone'],
                            "mpi_visual": format_mpi_visual(mpi_base),
                            "price_decimals": price_decimals
                        })
                        
                except Exception as e:
                    error_logger.log_error("Result Collection", e, {
                        "ticker": ticker,
                        "analysis_row_index": str(actual_date)
                    })
                    processing_errors.append(f"{ticker}: Result collection failed")
                    continue
                
                progress = 0.4 + (0.5 * (i + 1) / len(stock_data))
                progress_bar.progress(progress)
                
            except Exception as e:
                error_logger.log_error("Stock Processing", e, {
                    "ticker": ticker,
                    "processing_step": "overall"
                })
                processing_errors.append(f"{ticker}: {str(e)}")
                continue
        
        if processing_errors:
            error_logger.log_warning("Processing Summary", f"{len(processing_errors)} stocks had processing errors", {
                "errors": processing_errors,
                "success_rate": f"{len(results)}/{len(stock_data)} ({len(results)/len(stock_data)*100:.1f}%)"
            })
        
        status_text.text("📊 Preparing MPI-enhanced results...")
        progress_bar.progress(0.9)
        
        try:
            results_df = pd.DataFrame(results)
            error_logger.log_debug("Results Finalization", f"Created MPI-enhanced results dataframe", {
                "results_count": len(results_df),
                "mpi_columns": [col for col in results_df.columns if 'MPI' in col] if not results_df.empty else []
            })
        except Exception as e:
            error_logger.log_error("Results DataFrame Creation", e, {
                "results_data_sample": results[:3] if results else []
            })
            raise
        
        st.session_state.scan_results = results_df
        st.session_state.last_scan_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        st.session_state.last_scan_config = {
            'scope': 'Single Stock' if is_single_stock else 'Full Watchlist',
            'date': f'Historical ({analysis_date})' if is_historical else 'Current',
            'stock_count': len(results_df)
        }
        
        progress_bar.progress(1.0)
        status_text.text("✅ MPI scan completed!")
        
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        success_message = f"🎉 MPI Scan completed! Analyzed {len(results_df)} stocks successfully"
        if processing_errors:
            success_message += f" ({len(processing_errors)} errors - check log for details)"
        if is_historical:
            success_message += f" as of {analysis_date}"
        
        st.success(success_message)
        error_logger.log_debug("Scan Completion", "MPI scan completed successfully", {
            "successful_stocks": len(results_df),
            "total_requested": len(stocks_to_scan),
            "processing_errors": len(processing_errors)
        })
        
        st.rerun()
        
    except Exception as e:
        error_logger.log_error("Scan Execution", e, {
            "scan_parameters": {
                "stocks_to_scan": stocks_to_scan,
                "analysis_date": str(analysis_date) if analysis_date else None,
                "days_back": days_back,
                "rolling_window": rolling_window
            }
        })
        st.error("❌ MPI scan failed with critical error - check error log for full details")
        
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.empty()

def display_scan_results(results_df: pd.DataFrame):
    """Display scanning results with MPI Strategy filtering and intuitive momentum display"""
    
    error_logger = st.session_state.error_logger
    
    try:
        if results_df.empty:
            st.warning("No results to display.")
            error_logger.log_warning("Results Display", "Empty results dataframe")
            return
        
        # Summary metrics with MPI STATISTICS
        st.subheader("📊 Scan Summary with MPI Analysis")
        
        total_stocks = len(results_df)
        higher_hl_count = len(results_df[results_df['Higher_HL'] == 1])
        valid_crt_count = len(results_df[results_df['Valid_CRT'] == 1])
        higher_hl_with_crt = len(results_df[(results_df['Higher_HL'] == 1) & (results_df['Valid_CRT'] == 1)])
        
        # MPI statistics
        if 'MPI_Base' in results_df.columns:
            strong_bull_count = len(results_df[results_df['MPI_Base'] >= 0.70])
            bull_trend_count = len(results_df[(results_df['MPI_Base'] >= 0.50) & (results_df['MPI_Base'] < 0.70)])
            bear_trend_count = len(results_df[results_df['MPI_Base'] < 0.30])
            avg_mpi = results_df['MPI_Base'].mean()
        else:
            strong_bull_count = bull_trend_count = bear_trend_count = 0
            avg_mpi = 0.5
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("Total Analyzed", total_stocks)
        with col2:
            st.metric("Higher H/L", higher_hl_count, delta=f"{higher_hl_count/total_stocks*100:.1f}%")
        with col3:
            st.metric("Valid CRT", valid_crt_count)
        with col4:
            st.metric("Both Patterns", higher_hl_with_crt)
        with col5:
            st.metric("🚀 Strong Bull", strong_bull_count, delta="MPI ≥70%")
        with col6:
            st.metric("📈 Bull Trend", bull_trend_count, delta="MPI 50-70%")
        
        # Analysis Date Info with MPI summary
        if len(results_df) > 0:
            analysis_dates = results_df['Analysis_Date'].unique()
            if len(analysis_dates) == 1:
                st.info(f"📅 Analysis date: **{analysis_dates[0]}** | Average MPI: **{avg_mpi:.1%}** | Strong Bulls: **{strong_bull_count}** | Bears: **{bear_trend_count}**")
            else:
                st.info(f"📅 Analysis dates: **{', '.join(analysis_dates)}** | Average MPI: **{avg_mpi:.1%}**")
        
        # Base Filter Selection (UNCHANGED)
        st.subheader("🎯 Pattern Analysis")
        
        base_filter_options = {
            "Valid CRT + Higher H/L": "Stocks with both Valid CRT AND Higher H/L patterns",
            "Valid CRT Only": "All stocks with Valid CRT (Monday range expansion)",
            "Higher H/L Only": "All stocks with Higher High AND Higher Low patterns",
            "All Stocks": "Complete scan results without pattern filtering"
        }
        
        selected_base_filter = st.radio(
            "Select base pattern filter:",
            list(base_filter_options.keys()),
            help="Choose which pattern combination to analyze",
            key="base_filter_radio"
        )
        
        # Store in session state for filter logic
        st.session_state.base_filter_type = selected_base_filter
        
        # Apply base filter
        if selected_base_filter == "Valid CRT + Higher H/L":
            base_stocks = results_df[(results_df['Valid_CRT'] == 1) & (results_df['Higher_HL'] == 1)].copy()
            st.info(f"Showing {len(base_stocks)} stocks with both Valid CRT and Higher H/L patterns")
        
        elif selected_base_filter == "Valid CRT Only":
            base_stocks = results_df[results_df['Valid_CRT'] == 1].copy()
            st.info(f"Showing {len(base_stocks)} stocks with Valid CRT (Monday range expansion)")
        
        elif selected_base_filter == "Higher H/L Only":
            base_stocks = results_df[results_df['Higher_HL'] == 1].copy()
            st.info(f"Showing {len(base_stocks)} stocks with Higher High AND Higher Low patterns")
        
        else:  # All Stocks
            base_stocks = results_df.copy()
            st.info(f"Showing all {len(base_stocks)} scanned stocks")
        
        if len(base_stocks) > 0:
            # Apply dynamic filters (now includes MPI Strategy Filter)
            filtered_stocks = apply_dynamic_filters(base_stocks, results_df)
            
            # Display filtered results
            st.subheader(f"📋 MPI-Filtered Results ({len(filtered_stocks)} stocks)")
            
            if len(filtered_stocks) > 0:
                # Display columns in FILTER SEQUENCE ORDER: Price Context → Col1 → Col2 → Col3 → Col4
                # ALWAYS include Higher_HL column for all base filters
                display_cols = ['Ticker', 'Name', 'Close', 'CRT_High', 'CRT_Low',
                               'CRT_Velocity', 'IBS', 'Higher_HL', 'MPI_Zone_Emoji', 'MPI_Base', 'MPI_Visual']
                
                base_column_config = {
                    'Ticker': st.column_config.TextColumn('Ticker', width='small'),
                    'Name': st.column_config.TextColumn('Company Name', width='medium'),
                    'CRT_Velocity': st.column_config.NumberColumn('CRT Vel', format='%+.4f'),
                    'IBS': st.column_config.NumberColumn('IBS', format='%.3f'),
                    'Higher_HL': st.column_config.NumberColumn('H/L', width='small'),
                    'MPI_Zone_Emoji': st.column_config.TextColumn('📊', width='small', help='MPI Strategy Zone'),
                    'MPI_Base': st.column_config.NumberColumn('MPI', format='%.1%', help='Market Positivity Index'),
                    'MPI_Visual': st.column_config.TextColumn('MPI Visual', width='medium', help='Visual MPI representation')
                }
                
                # Apply dynamic price formatting
                column_config = get_dynamic_column_config(filtered_stocks, display_cols, base_column_config)
                
                # Filter display_cols to only include existing columns
                display_cols = [col for col in display_cols if col in filtered_stocks.columns]
                
                # Verify we have all required columns before displaying
                missing_cols = [col for col in display_cols if col not in filtered_stocks.columns]
                if missing_cols:
                    st.error(f"❌ Missing columns in filtered results: {missing_cols}")
                    st.write("Available columns:", list(filtered_stocks.columns))
                else:
                    try:
                        st.dataframe(
                            filtered_stocks[display_cols],
                            column_config=column_config,
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # TradingView Export (ENHANCED with MPI info)
                        st.subheader("📋 TradingView Export (MPI-Filtered)")
                        tv_tickers = [f"SGX:{ticker.replace('.SI', '')}" for ticker in filtered_stocks['Ticker'].tolist()]
                        tv_string = ','.join(tv_tickers)
                        
                        # Calculate MPI summary for export description
                        if 'MPI_Base' in filtered_stocks.columns:
                            mpi_summary = f"Avg MPI: {filtered_stocks['MPI_Base'].mean():.1%}"
                            strong_bulls_in_export = len(filtered_stocks[filtered_stocks['MPI_Base'] >= 0.70])
                            if strong_bulls_in_export > 0:
                                mpi_summary += f" | {strong_bulls_in_export} Strong Bulls"
                        else:
                            mpi_summary = ""
                        
                        st.text_area(
                            f"Singapore Exchange (SGX) - {selected_base_filter} ({len(tv_tickers)} stocks) {mpi_summary}:",
                            value=tv_string,
                            height=100,
                            help="Copy and paste into TradingView watchlist. Sorted by MPI strategy zones."
                        )
                        
                        # Export filtered data with MPI
                        csv_data = filtered_stocks.to_csv(index=False)
                        filename_prefix = selected_base_filter.lower().replace(' ', '_').replace('+', 'and')
                        st.download_button(
                            label="📥 Download MPI-Enhanced Data (CSV)",
                            data=csv_data,
                            file_name=f"mpi_{filename_prefix}_{len(filtered_stocks)}_stocks.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error displaying filtered results: {str(e)}")
                        error_logger.log_error("Filtered Results Display", e, {
                            "filtered_stocks_shape": filtered_stocks.shape,
                            "display_cols": display_cols,
                            "missing_cols": missing_cols
                        })
            
            else:
                st.warning("No stocks match the current filter criteria")
        
        else:
            st.warning(f"No stocks found for pattern: {selected_base_filter}")
        
        # Full Results Table with MPI SYSTEM (same column order as filtered results)
        with st.expander("📋 Full MPI Analysis Results", expanded=False):
            try:
                full_results_cols = [
                    'Analysis_Date', 'Ticker', 'Name', 'Close', 'CRT_High', 'CRT_Low',
                    'CRT_Velocity', 'IBS', 'Higher_HL', 'Valid_CRT', 
                    'MPI_Zone_Emoji', 'MPI_Zone', 'MPI_Base', 'MPI_Acceleration', 'MPI_Visual', 
                    'Signal_Breakout', 'Signal_Pullback', 'Signal_Short'
                ]
                
                base_full_results_config = {
                    'Analysis_Date': st.column_config.TextColumn('Date', width='small'),
                    'Ticker': st.column_config.TextColumn('Ticker', width='small'),
                    'Name': st.column_config.TextColumn('Company Name', width='medium'),
                    'CRT_Velocity': st.column_config.NumberColumn('CRT Vel', format='%+.4f'),
                    'IBS': st.column_config.NumberColumn('IBS', format='%.3f'),
                    'Higher_HL': st.column_config.NumberColumn('H/L', width='small'),
                    'Valid_CRT': st.column_config.NumberColumn('CRT', width='small'),
                    'MPI_Zone_Emoji': st.column_config.TextColumn('📊', width='small'),
                    'MPI_Zone': st.column_config.TextColumn('MPI Zone', width='medium'),
                    'MPI_Base': st.column_config.NumberColumn('MPI Base', format='%.1%'),
                    'MPI_Acceleration': st.column_config.NumberColumn('MPI Accel', format='%+.3f'),
                    'MPI_Visual': st.column_config.TextColumn('MPI Visual', width='medium'),
                    'Signal_Breakout': st.column_config.NumberColumn('🚀', width='small'),
                    'Signal_Pullback': st.column_config.NumberColumn('🎯', width='small'),
                    'Signal_Short': st.column_config.NumberColumn('📉', width='small')
                }
                
                # Apply dynamic price formatting for full results
                full_results_column_config = get_dynamic_column_config(results_df, full_results_cols, base_full_results_config)
                
                # Filter to only existing columns
                full_results_cols = [col for col in full_results_cols if col in results_df.columns]
                
                # Check for missing columns
                missing_cols = [col for col in ['Ticker', 'Name', 'Close', 'MPI_Base'] if col not in results_df.columns]
                if missing_cols:
                    st.error(f"❌ Missing essential columns in results: {missing_cols}")
                    st.write("Available columns:", list(results_df.columns))
                else:
                    st.dataframe(
                        results_df[full_results_cols],
                        column_config=full_results_column_config,
                        use_container_width=True,
                        hide_index=True
                    )
                    
            except Exception as e:
                st.error(f"❌ Error displaying full results: {str(e)}")
                error_logger.log_error("Full Results Display", e, {
                    "results_df_shape": results_df.shape if not results_df.empty else "Empty",
                    "attempted_columns": full_results_cols if 'full_results_cols' in locals() else "Unknown"
                })
        
        # MPI Strategy Insights
        if 'MPI_Base' in results_df.columns:
            with st.expander("📈 MPI Strategy Insights", expanded=False):
                
                # Create strategy zone summary
                zone_summary = []
                for zone in ['Strong Bull', 'Bull Trend', 'Neutral', 'Bear Trend']:
                    zone_stocks = results_df[results_df['MPI_Zone'] == zone]
                    if len(zone_stocks) > 0:
                        avg_mpi = zone_stocks['MPI_Base'].mean()
                        zone_summary.append({
                            'Zone': zone,
                            'Count': len(zone_stocks),
                            'Avg MPI': f"{avg_mpi:.1%}",
                            'Top Stock': zone_stocks.iloc[zone_stocks['MPI_Base'].idxmax()]['Name'] if len(zone_stocks) > 0 else 'N/A'
                        })
                
                if zone_summary:
                    st.dataframe(pd.DataFrame(zone_summary), hide_index=True, use_container_width=True)
                
                # MPI distribution chart
                if len(results_df) > 1:
                    fig = px.histogram(
                        results_df, 
                        x='MPI_Base',
                        nbins=20,
                        title="MPI Distribution Across All Stocks",
                        labels={'MPI_Base': 'Market Positivity Index', 'count': 'Number of Stocks'}
                    )
                    
                    # Add strategy zone shading
                    fig.add_vrect(x0=0.70, x1=1.00, fillcolor="green", opacity=0.2, annotation_text="🚀 Strong Bull")
                    fig.add_vrect(x0=0.50, x1=0.70, fillcolor="lightgreen", opacity=0.2, annotation_text="📈 Bull Trend")
                    fig.add_vrect(x0=0.30, x1=0.50, fillcolor="orange", opacity=0.2, annotation_text="➖ Neutral")
                    fig.add_vrect(x0=0.00, x1=0.30, fillcolor="red", opacity=0.2, annotation_text="📉 Bear Trend")
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        error_logger.log_debug("Results Display", "Successfully displayed MPI-enhanced results", {
            "total_displayed": len(results_df),
            "base_filter": selected_base_filter,
            "mpi_available": 'MPI_Base' in results_df.columns,
            "strong_bull_count": strong_bull_count
        })
        
    except Exception as e:
        error_logger.log_error("Results Display", e, {
            "results_shape": results_df.shape if not results_df.empty else "Empty DataFrame",
            "has_mpi_columns": 'MPI_Base' in results_df.columns if not results_df.empty else False
        })
        st.error("❌ Error displaying MPI results - check error log for details")

if __name__ == "__main__":
    show()