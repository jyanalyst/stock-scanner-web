"""
Common UI components and utilities used across pages
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import date, timedelta
import plotly.graph_objects as go
import plotly.express as px


def create_progress_container() -> Tuple[Any, Any]:
    """Create progress bar and status text containers"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    return progress_bar, status_text


def update_progress(progress_bar: Any, status_text: Any, progress: float, message: str):
    """Update progress bar and status message"""
    progress_bar.progress(progress)
    status_text.text(message)


def clear_progress(progress_bar: Any, status_text: Any):
    """Clear progress indicators"""
    progress_bar.empty()
    status_text.empty()


def display_metric_grid(metrics: List[Dict[str, Any]], columns: int = 4):
    """
    Display metrics in a grid layout

    Args:
        metrics: List of metric dictionaries with 'label', 'value', 'delta' keys
        columns: Number of columns in grid
    """
    rows = (len(metrics) + columns - 1) // columns

    for row in range(rows):
        cols = st.columns(columns)
        for col_idx in range(columns):
            metric_idx = row * columns + col_idx
            if metric_idx < len(metrics):
                metric = metrics[metric_idx]
                with cols[col_idx]:
                    st.metric(
                        metric['label'],
                        metric['value'],
                        metric.get('delta'),
                        help=metric.get('help')
                    )


def create_dynamic_column_config(df: pd.DataFrame, display_cols: List[str],
                               base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create dynamic column configuration with price-based decimal formatting"""
    config = base_config.copy()

    price_columns = ['Close', 'High', 'Low', 'CRT_High', 'CRT_Low', 'Weekly_Open']

    for col in price_columns:
        if col in display_cols and col in df.columns and len(df) > 0:
            min_price = df[col].min()
            format_str = "$%.3f" if min_price < 1.00 else "$%.2f"

            config[col] = st.column_config.NumberColumn(
                config[col].title if col in config and hasattr(config[col], 'title') else col.replace('_', ' ').title(),
                format=format_str
            )

    return config


def format_mpi_visual(mpi_value: float) -> str:
    """Convert MPI to visual blocks for intuitive display"""
    if pd.isna(mpi_value):
        return "░░░░░░░░░░"

    blocks = max(0, min(10, int(mpi_value * 10)))
    return "█" * blocks + "░" * (10 - blocks)


def get_price_format(price: float) -> str:
    """Get the appropriate price format string based on price level"""
    return "$%.3f" if price < 1.00 else "$%.2f"


def create_date_range_selector(default_days_back: int = 365) -> Tuple[date, date]:
    """
    Create a date range selector with sensible defaults

    Args:
        default_days_back: Default number of days to look back

    Returns:
        Tuple of (start_date, end_date)
    """
    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input(
            "Start Date:",
            value=date.today() - timedelta(days=default_days_back),
            max_value=date.today() - timedelta(days=1),
            help="Start date for analysis"
        )

    with col2:
        end_date = st.date_input(
            "End Date:",
            value=date.today() - timedelta(days=1),
            min_value=start_date,
            max_value=date.today() - timedelta(days=1),
            help="End date for analysis"
        )

    # Validate date range
    if end_date <= start_date:
        st.error("End date must be after start date")
        return None, None

    return start_date, end_date


def create_advanced_settings_expander():
    """Create advanced settings expander with common options"""
    with st.expander("⚙️ Advanced Settings"):
        col1, col2 = st.columns(2)

        with col1:
            days_back = st.number_input(
                "Days of Historical Data",
                min_value=30,
                max_value=500,
                value=100,
                help="Number of days to load (loads all available from local files)"
            )

        with col2:
            rolling_window = st.number_input(
                "Rolling Window",
                min_value=10,
                max_value=50,
                value=20,
                help="Rolling window for technical calculations"
            )

    return days_back, rolling_window


def display_data_table(df: pd.DataFrame, title: str = None, height: int = 400,
                      use_container_width: bool = True):
    """
    Display a data table with consistent styling

    Args:
        df: DataFrame to display
        title: Optional title for the table
        height: Height of the table in pixels
        use_container_width: Whether to use full container width
    """
    if title:
        st.subheader(title)

    if df.empty:
        st.warning("No data to display")
        return

    # Show basic info
    st.info(f"Showing {len(df):,} rows × {len(df.columns)} columns")

    # Display the table
    st.dataframe(
        df,
        use_container_width=use_container_width,
        height=height,
        hide_index=False
    )


def create_download_button(data: Any, filename: str, label: str = "📥 Download",
                          mime_type: str = None, help_text: str = None):
    """
    Create a download button with consistent styling

    Args:
        data: Data to download
        filename: Filename for download
        label: Button label
        mime_type: MIME type of the data
        help_text: Help text for the button
    """
    st.download_button(
        label=label,
        data=data,
        file_name=filename,
        mime=mime_type,
        help=help_text,
        width="stretch"
    )


def display_chart(fig: go.Figure, title: str = None, height: int = 400):
    """
    Display a Plotly chart with consistent styling

    Args:
        fig: Plotly figure to display
        title: Optional title
        height: Chart height in pixels
    """
    if title:
        st.subheader(title)

    fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch")


def create_info_box(message: str, icon: str = "ℹ️"):
    """Create an info box with consistent styling"""
    st.info(f"{icon} {message}")


def create_success_box(message: str, icon: str = "✅"):
    """Create a success box with consistent styling"""
    st.success(f"{icon} {message}")


def create_warning_box(message: str, icon: str = "⚠️"):
    """Create a warning box with consistent styling"""
    st.warning(f"{icon} {message}")


def create_error_box(message: str, icon: str = "❌"):
    """Create an error box with consistent styling"""
    st.error(f"{icon} {message}")


def create_section_header(title: str, description: str = None):
    """
    Create a section header with optional description

    Args:
        title: Section title
        description: Optional description text
    """
    st.subheader(title)
    if description:
        st.markdown(description)


def create_sidebar_section(title: str):
    """Create a sidebar section with consistent styling"""
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### {title}")


def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None) -> bool:
    """
    Validate that a DataFrame has required structure

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Returns:
        True if valid, False otherwise
    """
    if df is None or df.empty:
        st.error("No data available")
        return False

    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            return False

    return True