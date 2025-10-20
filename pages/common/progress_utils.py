"""
Progress Update Utilities
Reusable functions for progress bar updates with consistent formatting
"""

import time
from typing import Dict, List, Callable, Any
import streamlit as st
from pages.common.constants import PROGRESS_UPDATE_INTERVAL


def update_progress_from_stats(
    progress_bar,
    status_text,
    stats: Dict[str, Any],
    start_progress: float = 0.0,
    end_progress: float = 1.0,
    status_emoji_map: Dict[str, str] = None
) -> None:
    """
    Generic progress updater from stats dictionary with 'details' list
    
    Args:
        progress_bar: Streamlit progress bar element
        status_text: Streamlit text element for status messages
        stats: Dictionary containing 'total_stocks' and 'details' list
        start_progress: Starting progress value (0.0-1.0)
        end_progress: Ending progress value (0.0-1.0)
        status_emoji_map: Custom emoji mapping for statuses (optional)
    
    Example stats format:
        {
            'total_stocks': 10,
            'details': [
                {'ticker': 'A17U.SG', 'status': 'updated', 'message': 'Added 1 row'},
                {'ticker': 'C38U.SG', 'status': 'skipped', 'message': 'Date exists'},
                ...
            ]
        }
    """
    if not stats.get('details') or stats.get('total_stocks', 0) == 0:
        return
    
    # Default emoji mapping
    if status_emoji_map is None:
        status_emoji_map = {
            'updated': '✅',
            'created': '🆕',
            'skipped': '⏭️',
            'error': '❌',
            'failed': '❌'
        }
    
    progress_range = end_progress - start_progress
    
    for i, detail in enumerate(stats['details']):
        progress = start_progress + (progress_range * (i + 1) / stats['total_stocks'])
        status = detail.get('status', 'unknown')
        ticker = detail.get('ticker', 'Unknown')
        message = detail.get('message', '')
        
        emoji = status_emoji_map.get(status, '❓')
        status_message = f"{emoji} {ticker} - {message}"
        
        # Update progress bar and status text
        from pages.common.ui_components import update_progress
        update_progress(progress_bar, status_text, progress, status_message)
        
        time.sleep(PROGRESS_UPDATE_INTERVAL)


def create_progress_tracker(
    total_items: int,
    start_progress: float = 0.0,
    end_progress: float = 1.0
) -> Callable[[int, str], float]:
    """
    Create a progress tracker function for iterative operations
    
    Args:
        total_items: Total number of items to process
        start_progress: Starting progress value (0.0-1.0)
        end_progress: Ending progress value (0.0-1.0)
    
    Returns:
        Function that takes (current_index, message) and returns progress value
    
    Example:
        tracker = create_progress_tracker(total_items=100, start_progress=0.2, end_progress=0.8)
        for i, item in enumerate(items):
            progress = tracker(i, f"Processing {item}")
            update_progress(bar, text, progress, f"Processing {item}")
    """
    progress_range = end_progress - start_progress
    
    def calculate_progress(current_index: int, message: str = "") -> float:
        """Calculate current progress value"""
        if total_items == 0:
            return end_progress
        return start_progress + (progress_range * (current_index + 1) / total_items)
    
    return calculate_progress


def format_progress_message(
    operation: str,
    current: int,
    total: int,
    item_name: str = "items",
    show_percentage: bool = False
) -> str:
    """
    Format a standard progress message
    
    Args:
        operation: Operation being performed (e.g., "Processing", "Updating")
        current: Current item number
        total: Total items
        item_name: Name of items being processed
        show_percentage: Whether to show percentage
    
    Returns:
        Formatted message string
    
    Examples:
        >>> format_progress_message("Processing", 5, 10, "stocks")
        'Processing 5/10 stocks...'
        >>> format_progress_message("Updating", 3, 10, "files", show_percentage=True)
        'Updating 3/10 files (30%)...'
    """
    msg = f"{operation} {current}/{total} {item_name}"
    
    if show_percentage and total > 0:
        pct = (current / total) * 100
        msg += f" ({pct:.0f}%)"
    
    msg += "..."
    return msg


def batch_progress_update(
    progress_bar,
    status_text,
    items: List[Any],
    process_fn: Callable[[Any], Dict[str, Any]],
    start_progress: float = 0.0,
    end_progress: float = 1.0,
    operation_name: str = "Processing"
) -> List[Dict[str, Any]]:
    """
    Process items with automatic progress updates
    
    Args:
        progress_bar: Streamlit progress bar element
        status_text: Streamlit text element
        items: List of items to process
        process_fn: Function that processes each item and returns result dict
        start_progress: Starting progress value
        end_progress: Ending progress value
        operation_name: Name of operation for status messages
    
    Returns:
        List of results from process_fn
    
    Example:
        def process_stock(ticker):
            # ... processing logic
            return {'ticker': ticker, 'status': 'success'}
        
        results = batch_progress_update(
            bar, text, stock_list, process_stock,
            operation_name="Analyzing"
        )
    """
    from pages.common.ui_components import update_progress
    
    results = []
    tracker = create_progress_tracker(len(items), start_progress, end_progress)
    
    for i, item in enumerate(items):
        # Process item
        result = process_fn(item)
        results.append(result)
        
        # Update progress
        progress = tracker(i, "")
        msg = format_progress_message(operation_name, i + 1, len(items), "items")
        update_progress(progress_bar, status_text, progress, msg)
        
        time.sleep(PROGRESS_UPDATE_INTERVAL)
    
    return results
