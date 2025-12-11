"""
Clean ML Data Collection Script
- Suppresses Streamlit warnings
- Shows only essential progress
- Better error visibility
- More frequent checkpoints (every 10 days)

Usage:
    python scripts/run_ml_collection_clean.py
"""

import sys
import os
from datetime import datetime
import logging
import warnings
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# COMPREHENSIVE STREAMLIT WARNING SUPPRESSION
# Set environment variables to quiet Streamlit BEFORE any imports
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
os.environ['STREAMLIT_LOGGER_LEVEL'] = 'ERROR'

# Suppress ALL Streamlit warnings and logs BEFORE importing anything
warnings.filterwarnings('ignore', message='.*ScriptRunContext.*')
warnings.filterwarnings('ignore', message='.*Session state does not function.*')
warnings.filterwarnings('ignore', message='.*No runtime found.*')
warnings.filterwarnings('ignore', message='.*Thread.*missing ScriptRunContext.*')
warnings.filterwarnings('ignore', message='.*missing ScriptRunContext.*')
warnings.filterwarnings('ignore', message='.*Warning: to view this Streamlit app.*')
warnings.filterwarnings('ignore', category=UserWarning, module='streamlit')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='streamlit')

# Set ALL streamlit-related loggers to CRITICAL level
for logger_name in [
    'streamlit',
    'streamlit.runtime',
    'streamlit.runtime.scriptrunner_utils',
    'streamlit.runtime.state',
    'streamlit.runtime.caching',
    'streamlit.runtime.scriptrunner_utils.script_run_context',
    'streamlit.runtime.state.session_state_proxy',
    'streamlit.runtime.caching.cache_data_api'
]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

from ml.data_collection import MLDataCollector
from tqdm import tqdm
import time
import sys
import threading
import io
from contextlib import redirect_stderr

# AGGRESSIVE LOGGING SUPPRESSION - SILENCE ALL MODULES
# Set root logger to ERROR level to suppress INFO/WARNING from all modules
logging.getLogger().setLevel(logging.ERROR)

# SUPPRESS ALL PYTHON WARNINGS GLOBALLY
warnings.filterwarnings('ignore')

# Silence ALL known noisy modules
noisy_modules = [
    # Core scanner modules
    'pages.scanner.logic',
    'pages.scanner.data',
    'pages.scanner.ui',
    'pages.scanner.ui_flow_analysis',
    'pages.scanner.config',
    'pages.scanner.constants',
    'pages.scanner',

    # Data loading modules
    'core.local_file_loader',
    'core.data_fetcher',
    'core.technical_analysis',
    'core',

    # Analysis modules
    'pages.common.performance',
    'pages.common.data_utils',
    'pages.common.data_validation',
    'pages.common.error_handler',
    'pages.common.progress_utils',
    'pages.common.ui_components',
    'pages.common.constants',
    'pages.common',

    # ML modules
    'ml.data_collection',
    'ml.data_validator',
    'ml',

    # Utility modules
    'utils.analyst_reports',
    'utils.date_utils',
    'utils.earnings_reports',
    'utils.helpers',
    'utils.paths',
    'utils.watchlist',
    'utils',

    # All streamlit modules (already done above but being explicit)
    'streamlit',
    'streamlit.runtime',
    'streamlit.runtime.scriptrunner_utils',
    'streamlit.runtime.state',
    'streamlit.runtime.caching',
    'streamlit.runtime.scriptrunner_utils.script_run_context',
    'streamlit.runtime.state.session_state_proxy',
    'streamlit.runtime.caching.cache_data_api',

    # Specific noisy loggers
    'CrossStockRankings',
    'AnalystReports',
    'EarningsReports',
    'EarningsReaction',
    'Scanner',
    'Unknown'
]

for module in noisy_modules:
    logging.getLogger(module).setLevel(logging.ERROR)

# Configure OUR logging - only show important messages
logging.basicConfig(
    level=logging.INFO,  # Our script's messages
    format='%(message)s',  # Simple format, no timestamps
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data/ml_training/collection.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)
# Ensure our logger is not silenced
logger.setLevel(logging.INFO)


def load_ml_config():
    """Load ML configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs', 'ml_config.yaml')

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"❌ Error loading config from {config_path}: {e}")
        print("💡 Using default settings...")
        return {
            'data_collection': {
                'start_date': '2023-01-01',
                'end_date': '2024-12-31',
                'forward_days': [2, 3, 4],
                'checkpoint_frequency': 20,
                'max_workers': 1
            }
        }


class StreamlitWarningFilter:
    """Smart stderr filter that suppresses Streamlit warnings but preserves real errors"""

    def __init__(self):
        self.original_stderr = None
        self.captured_output = []
        self.filter_thread = None

    def __enter__(self):
        self.original_stderr = sys.stderr
        self.captured_output = []

        # Create a pipe to capture stderr
        from io import StringIO
        self.string_buffer = StringIO()

        # Replace stderr with our buffer
        sys.stderr = self.string_buffer
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original stderr
        sys.stderr = self.original_stderr

        # Process captured output
        captured = self.string_buffer.getvalue()

        if captured:
            # Filter out Streamlit warnings but keep real errors
            lines = captured.split('\n')
            filtered_lines = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Filter out Streamlit warnings
                if any(phrase in line.lower() for phrase in [
                    'missing scriptRunContext',
                    'session state does not function',
                    'no runtime found',
                    'warning: to view this streamlit app',
                    'thread \'mainthread\': missing scriptRunContext'
                ]):
                    continue  # Suppress this warning

                # Keep real errors (but these should be rare)
                filtered_lines.append(line)

            # Print any remaining stderr output (real errors)
            if filtered_lines:
                print("STDERR:", file=self.original_stderr)
                for line in filtered_lines:
                    print(line, file=self.original_stderr)


def main():
    """Run ML data collection with enhanced visual progress bar"""

    # Load configuration from YAML
    config = load_ml_config()
    data_config = config.get('data_collection', {})

    # Extract settings from config
    start_date = data_config.get('start_date', '2023-01-01')
    end_date = data_config.get('end_date', '2024-12-31')
    forward_days = data_config.get('forward_days', [2, 3, 4])
    checkpoint_freq = data_config.get('checkpoint_frequency', 20)
    max_workers = data_config.get('max_workers', 1)

    # Calculate estimated time (rough approximation: ~0.1 hours per trading day)
    from datetime import datetime
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    days_diff = (end_dt - start_dt).days
    trading_days = max(1, int(days_diff * 0.7))  # Rough estimate of trading days
    estimated_hours = trading_days * 0.1  # ~6 minutes per day with parallel processing

    print("=" * 80)
    print("🚀 ML DATA COLLECTION - ENHANCED MODE")
    print("=" * 80)
    print()
    print(f"📅 Date Range: {start_date} to {end_date}")
    print(f"📊 Forward Returns: {', '.join([f'{d}-day' for d in forward_days])}")
    print(f"⏱️  Estimated Time: ~{estimated_hours:.1f} hours ({trading_days} trading days)")
    print("📊 Visual Progress: Real-time with ETA")
    print(f"💾 Checkpoints: Every {checkpoint_freq} days")
    print(f"⚡ Parallel Workers: {max_workers}")
    print()
    print("=" * 80)
    print()

    # Confirm before starting
    response = input(f"Ready to start? This will take ~{estimated_hours:.1f} hours. (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("❌ Collection cancelled.")
        return

    print()
    print("🏁 Starting collection...")
    print()

    start_time = datetime.now()
    total_samples = 0

    try:
        # REDIRECT STDERR TO SUPPRESS REMAINING WARNINGS
        import sys
        from contextlib import redirect_stderr
        import os

        # Initialize collector with stderr redirected
        with open(os.devnull, 'w') as devnull:
            with redirect_stderr(devnull):
                collector = MLDataCollector(
                    start_date=start_date,
                    end_date=end_date,
                    forward_days=forward_days
                )

                # Get trading dates for progress tracking
                trading_dates = collector._get_trading_dates()

        total_dates = len(trading_dates)

        print(f"📅 Found {total_dates} trading dates to process")
        print()

        # Create enhanced progress bar
        with tqdm(total=total_dates,
                  desc="Processing",
                  unit="dates",
                  bar_format='{desc}: {percentage:3.1f}%|{bar}| {n_fmt}/{total_fmt} dates | {rate_fmt} | ETA: {remaining}',
                  ncols=100) as pbar:

            # Custom progress tracking
            processed_dates = 0
            samples_collected = 0
            last_checkpoint = 0

            # Override the collection method to track progress
            original_collect = collector.collect_training_data

            def enhanced_collect(*args, **kwargs):
                nonlocal processed_dates, samples_collected, pbar

                # Get the original save_checkpoint method
                original_save_checkpoint = collector._save_checkpoint

                def enhanced_checkpoint(samples, current_date):
                    nonlocal processed_dates, samples_collected, last_checkpoint, pbar

                    # Update counters
                    processed_dates = list(trading_dates).index(current_date) + 1
                    samples_collected = len(samples)

                    # Update progress bar with enhanced info
                    pbar.n = processed_dates
                    pbar.set_postfix({
                        'samples': f'{samples_collected:,}',
                        'date': current_date.strftime('%Y-%m-%d'),
                        'rate': f'{processed_dates/(time.time()-start_time.timestamp())*3600:.1f}d/h'
                    })
                    pbar.refresh()

                    # Save checkpoint every N dates (from config)
                    if processed_dates - last_checkpoint >= checkpoint_freq:
                        original_save_checkpoint(samples, current_date)
                        last_checkpoint = processed_dates
                        print(f"\n💾 Checkpoint saved: {samples_collected:,} samples at {current_date.strftime('%Y-%m-%d')}")

                # Override checkpoint method
                collector._save_checkpoint = enhanced_checkpoint

                # Run original collection
                result = original_collect(*args, **kwargs)

                # Final checkpoint
                if hasattr(result, '__len__'):
                    final_samples = len(result)
                    print(f"\n💾 Final checkpoint: {final_samples:,} samples")

                return result

            # Run collection with enhanced progress tracking and stderr filtering
            with StreamlitWarningFilter():
                training_data = enhanced_collect(
                    save_path="data/ml_training/raw/",
                    resume_from=None,
                    use_validation=True
                )

            # Complete the progress bar
            pbar.n = total_dates
            pbar.set_postfix({
                'samples': f'{len(training_data):,}',
                'status': 'Complete!'
            })
            pbar.close()

        # Calculate final stats
        end_time = datetime.now()
        duration = end_time - start_time
        hours = duration.total_seconds() / 3600

        print()
        print("=" * 80)
        print("✅ COLLECTION COMPLETE!")
        print("=" * 80)
        print()
        print(f"📊 Total Samples: {len(training_data):,}")
        print(f"📈 Unique Stocks: {training_data['Ticker'].nunique()}")
        print(f"📅 Date Range: {training_data['entry_date'].min()} to {training_data['entry_date'].max()}")
        print(f"⏱️  Duration: {hours:.1f} hours")
        print(f"💾 Saved to: data/ml_training/raw/training_data_complete.parquet")
        print()
        print("🎯 Next Steps:")
        print("   1. Open ML Lab in Streamlit")
        print("   2. Click 'View Existing Data' to verify")
        print("   3. Proceed to Phase 2: Factor Analysis")
        print()
        print("=" * 80)

    except KeyboardInterrupt:
        print()
        print("⚠️  Collection interrupted by user (Ctrl+C)")
        print(f"💾 Collected {total_samples:,} samples before interruption")
        print("💡 Run resume script to continue:")
        print("   python scripts/resume_ml_collection.py")

    except Exception as e:
        print()
        print("=" * 80)
        print("❌ ERROR OCCURRED")
        print("=" * 80)
        print()
        print(f"Error: {e}")
        print()
        print("📋 Full traceback:")
        import traceback
        traceback.print_exc()
        print()
        print("💡 Check collection.log for details")


if __name__ == "__main__":
    main()
