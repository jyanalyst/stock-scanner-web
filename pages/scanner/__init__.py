# Scanner module package
# Re-export main functions for backward compatibility

try:
    from pages.scanner.ui import show_update_prompt, show_scanning_configuration, show_advanced_settings, execute_scan_button, display_scan_summary, show_base_pattern_filter, display_filtered_results, show_full_results_table, show_mpi_insights
    from pages.scanner.logic import run_enhanced_stock_scan
    from pages.scanner.data import apply_dynamic_filters
except ImportError as e:
    # Handle case where streamlit is not available (for testing)
    print(f"Warning: Some scanner UI components not available: {e}")
    # Define dummy functions for testing
    def show_update_prompt(): pass
    def show_scanning_configuration(): return None, None, None, None
    def show_advanced_settings(): return 100, 20
    def execute_scan_button(*args): pass
    def display_scan_summary(*args): pass
    def show_base_pattern_filter(*args): return None
    def display_filtered_results(*args): pass
    def show_full_results_table(*args): pass
    def show_mpi_insights(*args): pass
    def run_enhanced_stock_scan(*args): pass
    def apply_dynamic_filters(*args): return None