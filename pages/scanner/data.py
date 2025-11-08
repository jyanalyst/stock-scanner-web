"""
Scanner Data Processing - Data preparation and filtering logic
Extracted from scanner.py for better organization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import streamlit as st

from pages.common.data_utils import safe_float, safe_int, safe_string
from pages.common.constants import MPI_TRENDS

logger = logging.getLogger(__name__)


def apply_mpi_expansion_filter(base_stocks: pd.DataFrame, selected_trends: List[str]) -> pd.DataFrame:
    """Apply pure MPI expansion-based filtering with smart sorting"""
    if base_stocks.empty or 'MPI_Trend' not in base_stocks.columns:
        logger.warning("MPI_Trend column not found - returning all stocks")
        return base_stocks

    if not selected_trends:
        return base_stocks.sort_values('MPI', ascending=False)

    # Filter by selected MPI trends
    filtered = base_stocks[base_stocks['MPI_Trend'].isin(selected_trends)].copy()

    # Smart sorting based on selected trends
    expansion_trends = ['Strong Expansion', 'Expanding']
    contraction_trends = ['Mild Contraction', 'Strong Contraction']

    selected_categories = []
    if any(trend in expansion_trends for trend in selected_trends):
        selected_categories.append('expansion')
    if 'Flat' in selected_trends:
        selected_categories.append('flat')
    if any(trend in contraction_trends for trend in selected_trends):
        selected_categories.append('contraction')

    # Sort based on predominant selection
    if selected_categories == ['contraction']:
        filtered = filtered.sort_values('MPI_Velocity', ascending=True)
    elif selected_categories == ['expansion']:
        filtered = filtered.sort_values('MPI_Velocity', ascending=False)
    else:
        filtered = filtered.sort_values('MPI', ascending=False)

    return filtered


def create_filter_statistics_dataframe(data: pd.Series, metric_name: str) -> pd.DataFrame:
    """Create standardized statistics dataframe for filters"""
    if len(data) == 0:
        return pd.DataFrame()

    stats_data = {
        "Metric": ["Count", "Min", "25th %ile", "Median", "75th %ile", "Max"],
        "Value": [
            f"{len(data)}",
            f"{data.min():.4f}",
            f"{np.percentile(data, 25):.4f}",
            f"{data.median():.4f}",
            f"{np.percentile(data, 75):.4f}",
            f"{data.max():.4f}"
        ],
        "Unit": ["stocks", "pp", "pp", "pp", "pp", "pp"] if "Velocity" in metric_name else ["stocks", "", "", "", "", ""]
    }
    return pd.DataFrame(stats_data)


def apply_velocity_filter(filtered_stocks: pd.DataFrame, results_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series], str]:
    """Apply VW Range Velocity percentile filter for daily momentum"""
    if filtered_stocks.empty:
        return filtered_stocks, None, "No stocks available"

    # Get velocity data
    if 'VW_Range_Velocity' in filtered_stocks.columns:
        velocities = filtered_stocks['VW_Range_Velocity']
    else:
        velocities = filtered_stocks.index.map(
            lambda idx: results_df.loc[idx, 'VW_Range_Velocity'] if idx in results_df.index else 0
        )
        velocities = pd.Series(velocities, index=filtered_stocks.index)

    # Remove zero velocities for percentile calculation
    non_zero_velocities = velocities[velocities != 0]

    if len(non_zero_velocities) == 0:
        return filtered_stocks, None, "No VW Range Velocity data available"

    # Percentile options
    percentile_options = {
        "Top 25%": 75,
        "Top 50%": 50,
        "Top 75%": 25,
        "Custom": "custom",
        "No Filter": None
    }

    selected_percentile = st.radio(
        "Select velocity filter:",
        list(percentile_options.keys()),
        index=4,
        key="vw_range_velocity_percentile_radio"
    )

    info_message = "All velocities included"

    if selected_percentile == "Custom":
        filter_type = st.radio(
            "Filter by:",
            ["Minimum", "Maximum"],
            horizontal=True,
            key="velocity_minmax_radio"
        )

        if filter_type == "Minimum":
            custom_velocity_value = st.number_input(
                "Enter minimum VW Range Velocity:",
                min_value=-1.0,
                max_value=1.0,
                value=0.05,
                step=0.01,
                format="%.4f",
                key="custom_velocity_min_input",
                help="Filter stocks with velocity >= this value"
            )
            filtered_stocks = filtered_stocks[velocities >= custom_velocity_value]
            info_message = f"VW Range Velocity ≥ {custom_velocity_value:+.4f} pp"
        else:
            custom_velocity_value = st.number_input(
                "Enter maximum VW Range Velocity:",
                min_value=-1.0,
                max_value=1.0,
                value=0.0,
                step=0.01,
                format="%.4f",
                key="custom_velocity_max_input",
                help="Filter stocks with velocity <= this value"
            )
            filtered_stocks = filtered_stocks[velocities <= custom_velocity_value]
            info_message = f"VW Range Velocity ≤ {custom_velocity_value:+.4f} pp"

    elif selected_percentile in ["Top 25%", "Top 50%", "Top 75%"]:
        percentile_val = percentile_options[selected_percentile]
        threshold_value = np.percentile(non_zero_velocities, percentile_val)
        filtered_stocks = filtered_stocks[velocities >= threshold_value]
        info_message = f"VW Range Velocity ≥ {threshold_value:+.4f} pp"

    return filtered_stocks, non_zero_velocities, info_message


def apply_ibs_filter(filtered_stocks: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Apply IBS percentile filter"""
    if filtered_stocks.empty:
        return filtered_stocks, "No stocks available"

    ibs_values = filtered_stocks['IBS']

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

    if selected_ibs_option == "Custom":
        filter_type = st.radio(
            "Filter by:",
            ["Minimum", "Maximum"],
            horizontal=True,
            key="ibs_minmax_radio"
        )

        if filter_type == "Minimum":
            custom_ibs_value = st.number_input(
                "Enter minimum IBS value:",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                format="%.2f",
                key="custom_ibs_min_input"
            )
            filtered_stocks = filtered_stocks[filtered_stocks['IBS'] >= custom_ibs_value]
            info_message = f"IBS ≥ {custom_ibs_value:.2f}"
        else:
            custom_ibs_value = st.number_input(
                "Enter maximum IBS value:",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                format="%.2f",
                key="custom_ibs_max_input"
            )
            filtered_stocks = filtered_stocks[filtered_stocks['IBS'] <= custom_ibs_value]
            info_message = f"IBS ≤ {custom_ibs_value:.2f}"

    elif selected_ibs_option != "No Filter":
        percentile_val = ibs_percentile_options[selected_ibs_option]
        threshold_value = np.percentile(ibs_values, percentile_val)
        filtered_stocks = filtered_stocks[filtered_stocks['IBS'] >= threshold_value]
        info_message = f"IBS ≥ {threshold_value:.3f} ({selected_ibs_option})"
    else:
        info_message = "All IBS values included"

    return filtered_stocks, info_message


def apply_relative_volume_filter(filtered_stocks: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Apply Relative Volume percentile filter"""
    if filtered_stocks.empty:
        return filtered_stocks, "No stocks available"

    rel_volume_values = filtered_stocks['Relative_Volume']

    rel_volume_percentile_options = {
        "Top 25%": 75,
        "Top 50%": 50,
        "Top 75%": 25,
        "Custom": "custom",
        "No Filter": None
    }

    selected_rel_volume_option = st.radio(
        "Select Relative Volume filter:",
        list(rel_volume_percentile_options.keys()),
        key="rel_volume_percentile_radio"
    )

    if selected_rel_volume_option == "Custom":
        filter_type = st.radio(
            "Filter by:",
            ["Minimum", "Maximum"],
            horizontal=True,
            key="rel_volume_minmax_radio"
        )

        if filter_type == "Minimum":
            custom_rel_volume_value = st.number_input(
                "Enter minimum Relative Volume %:",
                min_value=50.0,
                max_value=500.0,
                value=70.0,
                step=10.0,
                format="%.1f",
                key="custom_rel_volume_min_input"
            )
            filtered_stocks = filtered_stocks[filtered_stocks['Relative_Volume'] >= custom_rel_volume_value]
            info_message = f"Rel Volume ≥ {custom_rel_volume_value:.1f}%"
        else:
            custom_rel_volume_value = st.number_input(
                "Enter maximum Relative Volume %:",
                min_value=50.0,
                max_value=500.0,
                value=50.0,
                step=10.0,
                format="%.1f",
                key="custom_rel_volume_max_input"
            )
            filtered_stocks = filtered_stocks[filtered_stocks['Relative_Volume'] <= custom_rel_volume_value]
            info_message = f"Rel Volume ≤ {custom_rel_volume_value:.1f}%"

    elif selected_rel_volume_option != "No Filter":
        percentile_val = rel_volume_percentile_options[selected_rel_volume_option]
        threshold_value = np.percentile(rel_volume_values, percentile_val)
        filtered_stocks = filtered_stocks[filtered_stocks['Relative_Volume'] >= threshold_value]
        info_message = f"Rel Volume ≥ {threshold_value:.1f}% ({selected_rel_volume_option})"
    else:
        info_message = "All Relative Volume levels included"

    return filtered_stocks, info_message


def apply_higher_hl_filter(filtered_stocks: pd.DataFrame, base_filter_type: str) -> Tuple[pd.DataFrame, str, str]:
    """Apply Break H/L pattern filter"""
    hl_filter_options = {
        "Break H/L Only": "All stocks with Valid CRT (Monday range expansion)",
        "Break H Only": "Any break high (BH or BHL)",
        "No Filter": "All patterns"
    }

    st.markdown("**Break H/L Filter:**")

    selected_hl_filter = st.radio(
        "Select Break H/L filter:",
        list(hl_filter_options.keys()),
        key="higher_hl_pattern_radio"
    )

    if selected_hl_filter == "Break H/L Only":
        filtered_stocks = filtered_stocks[filtered_stocks['Higher_HL'] == 1]
        info_message = f"Break H/L Only (BHL) - {len(filtered_stocks)} stocks"
    elif selected_hl_filter == "Break H Only":
        filtered_stocks = filtered_stocks[filtered_stocks['Higher_H'] == 1]
        info_message = f"Break H Only (BH + BHL) - {len(filtered_stocks)} stocks"
    else:
        info_message = f"All patterns - {len(filtered_stocks)} stocks"

    return filtered_stocks, selected_hl_filter, info_message


def apply_mpi_filter(filtered_stocks: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], str]:
    """Apply MPI expansion trend filter"""
    if 'MPI_Trend' not in filtered_stocks.columns:
        st.warning("MPI expansion filtering not available - MPI_Trend data missing")
        return filtered_stocks, [], "MPI data unavailable"

    use_mpi_filter = st.checkbox("Enable MPI Filtering", value=True, key="mpi_filter_toggle")

    if not use_mpi_filter:
        return filtered_stocks, [], "MPI filter disabled"

    trend_options = [
        "📈 Expanding",
        "➖ Flat",
        "📉 Contracting"
    ]

    st.markdown("Select momentum trends:")
    selected_trends = []

    trend_mapping = {
        "📈 Expanding": "Expanding",
        "➖ Flat": "Flat",
        "📉 Contracting": "Contracting"
    }

    for i, trend in enumerate(trend_options):
        if st.checkbox(trend, key=f"mpi_trend_checkbox_{i}"):
            selected_trends.append(trend_mapping[trend])

    if selected_trends:
        filtered_stocks = apply_mpi_expansion_filter(filtered_stocks, selected_trends)
        display_names = [k for k, v in trend_mapping.items() if v in selected_trends]
        info_message = f"Selected: {', '.join([name.split(' ')[0] for name in display_names])} ({len(filtered_stocks)} stocks)"
    else:
        info_message = "No trends selected - showing all MPI levels"

    return filtered_stocks, selected_trends, info_message


def show_filter_statistics(component_name: str, data: Optional[pd.Series], base_stocks: Optional[pd.DataFrame] = None):
    """Show statistics for a filter component"""
    with st.expander(f"{component_name} Statistics", expanded=False):
        if component_name == "VW Range Velocity" and data is not None and len(data) > 0:
            stats_df = create_filter_statistics_dataframe(data, component_name)
            st.dataframe(stats_df, hide_index=True, use_container_width=True)

        elif component_name == "IBS" and data is not None and len(data) > 0:
            stats_df = create_filter_statistics_dataframe(data, component_name)
            st.dataframe(stats_df, hide_index=True, use_container_width=True)

        elif component_name == "Relative Volume" and data is not None and len(data) > 0:
            stats_df = create_filter_statistics_dataframe(data, component_name)
            st.dataframe(stats_df, hide_index=True, use_container_width=True)

        elif component_name == "Break H/L" and base_stocks is not None:
            higher_h_count = (base_stocks['Higher_H'] == 1).sum()
            higher_hl_count = (base_stocks['Higher_HL'] == 1).sum()
            hh_only_count = ((base_stocks['Higher_H'] == 1) & (base_stocks['Higher_HL'] == 0)).sum()
            total_count = len(base_stocks)

            hl_stats = pd.DataFrame({
                "Pattern": ["BHL (H/L)", "BH Only", "Total Break H", "Neither", "Total"],
                "Count": [
                    f"{higher_hl_count}",
                    f"{hh_only_count}",
                    f"{higher_h_count}",
                    f"{total_count - higher_h_count}",
                    f"{total_count}"
                ],
                "Percentage": [
                    f"{higher_hl_count/total_count*100:.1f}%",
                    f"{hh_only_count/total_count*100:.1f}%",
                    f"{higher_h_count/total_count*100:.1f}%",
                    f"{(total_count - higher_h_count)/total_count*100:.1f}%",
                    "100.0%"
                ],
                "Description": [
                    "Both break high AND break low",
                    "Break high only (not break low)",
                    "Any break high (BH + BHL)",
                    "No break high pattern",
                    "All stocks"
                ]
            })
            st.dataframe(hl_stats, hide_index=True, use_container_width=True)

        elif component_name == "MPI Trend" and base_stocks is not None and 'MPI_Trend' in base_stocks.columns:
            trend_counts = base_stocks['MPI_Trend'].value_counts()
            trend_order = ['Expanding', 'Flat', 'Contracting']

            trend_stats_data = []
            for trend in trend_order:
                count = trend_counts.get(trend, 0)
                if count > 0:
                    trend_stocks = base_stocks[base_stocks['MPI_Trend'] == trend]
                    avg_mpi = trend_stocks['MPI'].mean()
                    avg_velocity = trend_stocks['MPI_Velocity'].mean()

                    trend_stats_data.append({
                        "MPI Trend": trend,
                        "Count": f"{count}",
                        "Avg MPI": f"{avg_mpi:.1%}",
                        "Avg Velocity": f"{avg_velocity:+.3f}",
                        "Description": get_mpi_expansion_description(trend)
                    })

            if trend_stats_data:
                st.dataframe(pd.DataFrame(trend_stats_data), hide_index=True, use_container_width=True)


def apply_dynamic_filters(base_stocks: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    """Apply dynamic filtering with optimized component structure"""
    if base_stocks.empty:
        st.warning("No stocks available for filtering")
        return base_stocks

    st.subheader("🎯 Dynamic Filtering")

    col1, col2, col3, col4, col5 = st.columns(5)

    filtered_stocks = base_stocks.copy()
    filter_summary = []

    # COL 1: BREAK H/L FILTER
    with col1:
        filtered_stocks, selected_hl_filter, hl_info = apply_higher_hl_filter(filtered_stocks, st.session_state.get('base_filter_type', 'All Stocks'))
        st.info(hl_info)
        show_filter_statistics("Break H/L", None, base_stocks)
        if selected_hl_filter != "No Filter":
            filter_summary.append(f"H/L: {selected_hl_filter}")

    # COL 2: VW RANGE VELOCITY FILTER
    with col2:
        st.markdown("**VW Range Velocity Filter:**")
        filtered_stocks, velocity_data, velocity_info = apply_velocity_filter(filtered_stocks, results_df)
        st.info(velocity_info)
        if velocity_data is not None and len(velocity_data) > 0:
            show_filter_statistics("VW Range Velocity", velocity_data)
            if "≥" in velocity_info or "≤" in velocity_info:
                filter_summary.append("VW Range Velocity filtered")

    # COL 3: IBS FILTER
    with col3:
        st.markdown("**IBS Filter:**")
        filtered_stocks, ibs_info = apply_ibs_filter(filtered_stocks)
        st.info(ibs_info)
        if len(filtered_stocks) > 0:
            show_filter_statistics("IBS", filtered_stocks['IBS'])
            if "≥" in ibs_info or "≤" in ibs_info:
                filter_summary.append("IBS filtered")

    # COL 4: RELATIVE VOLUME FILTER
    with col4:
        st.markdown("**Relative Volume Filter:**")
        filtered_stocks, rel_volume_info = apply_relative_volume_filter(filtered_stocks)
        st.info(rel_volume_info)
        if len(filtered_stocks) > 0:
            show_filter_statistics("Relative Volume", filtered_stocks['Relative_Volume'])
            if "≥" in rel_volume_info or "≤" in rel_volume_info:
                filter_summary.append("Relative Volume filtered")

    # COL 5: MPI EXPANSION FILTER
    with col5:
        st.markdown("**MPI Expansion Filter:**")
        filtered_stocks, selected_trends, mpi_info = apply_mpi_filter(filtered_stocks)
        st.info(mpi_info)
        show_filter_statistics("MPI Trend", None, base_stocks)
        if selected_trends:
            filter_summary.append(f"MPI: {len(selected_trends)} trends")
        elif "disabled" in mpi_info:
            filter_summary.append("MPI: Disabled")

    if filter_summary:
        st.success(f"Active filters: {' + '.join(filter_summary)} → {len(filtered_stocks)} stocks")
    else:
        st.info(f"No additional filters applied → {len(filtered_stocks)} stocks")

    return filtered_stocks


def get_mpi_expansion_description(trend: str) -> str:
    """Get trading description for MPI expansion trends"""
    descriptions = {
        'Strong Expansion': 'Strong momentum building',
        'Expanding': 'Positive momentum',
        'Flat': 'No momentum change',
        'Mild Contraction': 'Momentum weakening',
        'Strong Contraction': 'Momentum declining'
    }
    return descriptions.get(trend, 'Unknown')