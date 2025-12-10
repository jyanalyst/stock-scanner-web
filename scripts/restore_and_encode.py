"""
Restore backup and re-run categorical encoding
"""

import shutil
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("Restoring backup...")
backup_path = "data/ml_training/raw/training_data_complete_backup.parquet"
data_path = "data/ml_training/raw/training_data_complete.parquet"

if os.path.exists(backup_path):
    shutil.copy(backup_path, data_path)
    print(f"✅ Restored {data_path} from backup")
else:
    print(f"⚠️ Backup not found: {backup_path}")
    print("Proceeding with current data...")

print("\nRunning categorical encoding...")
print("=" * 80)

# Import and run the encoding script
from add_categorical_encoding import main

success = main()
sys.exit(0 if success else 1)
