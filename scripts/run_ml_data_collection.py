"""
Standalone ML Data Collection Script
Run this separately from Streamlit app - takes ~73 hours

Usage:
    python scripts/run_ml_data_collection.py

Features:
- Runs independently of Streamlit
- Shows real-time progress
- Saves checkpoints every 20 days
- Can be resumed if interrupted
- Logs all activity to console and file
"""

import sys
import os
from datetime import datetime
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.data_collection import MLDataCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data/ml_training/collection.log')
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Run ML data collection"""
    
    print("=" * 80)
    print("🚀 ML DATA COLLECTION - STANDALONE MODE")
    print("=" * 80)
    print()
    print("📅 Date Range: 2023-01-01 to 2024-12-31")
    print("📊 Forward Returns: 2-day, 3-day, 4-day")
    print("⏱️  Estimated Time: ~73 hours (730 trading days)")
    print("💾 Save Path: data/ml_training/raw/")
    print()
    print("💡 TIP: Keep this terminal open and prevent computer from sleeping")
    print("💡 TIP: Checkpoints saved every 20 days - can resume if interrupted")
    print()
    print("=" * 80)
    print()
    
    # Confirm before starting
    response = input("Ready to start? This will take ~73 hours. (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("❌ Collection cancelled.")
        return
    
    print()
    print("🏁 Starting collection...")
    print()
    
    start_time = datetime.now()
    
    try:
        # Initialize collector
        collector = MLDataCollector(
            start_date="2023-01-01",
            end_date="2024-12-31",
            forward_days=[2, 3, 4]
        )
        
        # Run collection
        training_data = collector.collect_training_data(
            save_path="data/ml_training/raw/",
            resume_from=None,  # Set to date string to resume (e.g., "2023-06-15")
            use_validation=True  # Use validated stocks only
        )
        
        # Calculate stats
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
        print("💡 You can resume by setting resume_from parameter in this script")
        print("   Example: resume_from='2023-06-15'")
        
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
        print("💡 You may be able to resume from last checkpoint")


if __name__ == "__main__":
    main()
