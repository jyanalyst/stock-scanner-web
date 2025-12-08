"""
Resume ML Data Collection from Last Checkpoint
Automatically finds the last checkpoint and continues from there

Usage:
    python scripts/resume_ml_collection.py
"""

import sys
import os
from datetime import datetime
import logging
import glob

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.data_collection import MLDataCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data/ml_training/collection.log', mode='a')  # Append mode
    ]
)

logger = logging.getLogger(__name__)


def find_last_checkpoint():
    """Find the most recent checkpoint file"""
    checkpoint_dir = "data/ml_training/checkpoints"
    
    if not os.path.exists(checkpoint_dir):
        logger.warning(f"No checkpoint directory found at {checkpoint_dir}")
        return None
    
    # Find all checkpoint files
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.parquet"))
    
    if not checkpoint_files:
        logger.warning("No checkpoint files found")
        return None
    
    # Extract dates from filenames and find the latest
    checkpoints = []
    for filepath in checkpoint_files:
        filename = os.path.basename(filepath)
        # Extract date from filename like "checkpoint_20230302.parquet"
        date_str = filename.replace("checkpoint_", "").replace(".parquet", "")
        try:
            date_obj = datetime.strptime(date_str, "%Y%m%d")
            checkpoints.append((date_obj, date_str))
        except ValueError:
            logger.warning(f"Could not parse date from {filename}")
            continue
    
    if not checkpoints:
        return None
    
    # Sort by date and get the latest
    checkpoints.sort(reverse=True)
    latest_date_obj, latest_date_str = checkpoints[0]
    
    # Format as YYYY-MM-DD for resume_from parameter
    resume_date = latest_date_obj.strftime("%Y-%m-%d")
    
    return resume_date, latest_date_str, len(checkpoints)


def main():
    """Resume ML data collection from last checkpoint"""
    
    print("=" * 80)
    print("🔄 RESUME ML DATA COLLECTION")
    print("=" * 80)
    print()
    
    # Find last checkpoint
    checkpoint_info = find_last_checkpoint()
    
    if checkpoint_info is None:
        print("❌ No checkpoints found!")
        print()
        print("Options:")
        print("  1. Start fresh: python scripts/run_ml_data_collection.py")
        print("  2. Check if checkpoints exist: dir data\\ml_training\\checkpoints")
        print()
        return
    
    resume_date, checkpoint_file, total_checkpoints = checkpoint_info
    
    print(f"📁 Found {total_checkpoints} checkpoint(s)")
    print(f"📅 Last checkpoint: {checkpoint_file}")
    print(f"🔄 Will resume from: {resume_date}")
    print()
    print("📊 Collection will continue from this date to 2024-12-31")
    print("⏱️  Estimated remaining time depends on how far you got")
    print()
    print("=" * 80)
    print()
    
    # Confirm before starting
    response = input("Resume collection from this checkpoint? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("❌ Resume cancelled.")
        return
    
    print()
    print("🏁 Resuming collection...")
    print()
    
    start_time = datetime.now()
    
    try:
        # Initialize collector
        collector = MLDataCollector(
            start_date="2023-01-01",
            end_date="2024-12-31",
            forward_days=[2, 3, 4]
        )
        
        # Run collection with resume
        training_data = collector.collect_training_data(
            save_path="data/ml_training/raw/",
            resume_from=resume_date,
            use_validation=True
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
        print("💡 Run this script again to resume from the latest checkpoint")
        
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
        print("💡 Run this script again to resume from the latest checkpoint")


if __name__ == "__main__":
    main()
