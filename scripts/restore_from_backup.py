#!/usr/bin/env python3
"""
Restore Historical Data Files from Backup
=========================================

Restores CSV files from the most recent backup in data/Historical_Data/backups/

USAGE:
    python scripts/restore_from_backup.py --dry-run    # Show what would be restored
    python scripts/restore_from_backup.py --restore    # Restore all files from backup

AUTHOR: jyanalyst
DATE: 2025-11-22
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scripts/restore_report.txt', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class BackupRestorer:
    """Handles restoration of files from backup"""

    def __init__(self, data_dir: str = "data/Historical_Data"):
        self.data_dir = Path(data_dir)
        self.backup_dir = self.data_dir / "backups"

        # Statistics
        self.stats = {
            'backups_found': 0,
            'files_to_restore': 0,
            'files_restored': 0,
            'files_skipped': 0
        }

    def find_latest_backups(self) -> Dict[str, Path]:
        """Find the latest backup for each original file"""
        latest_backups = {}

        if not self.backup_dir.exists():
            logger.error(f"Backup directory not found: {self.backup_dir}")
            return latest_backups

        # Get all backup files
        backup_files = list(self.backup_dir.glob("*.csv"))
        logger.info(f"Found {len(backup_files)} backup files")

        # Group by original filename
        backup_groups = {}
        for backup_file in backup_files:
            # Extract original name (remove .backup_TIMESTAMP)
            name_parts = backup_file.stem.split('.backup_')
            if len(name_parts) == 2:
                original_name = name_parts[0] + '.csv'
                timestamp_str = name_parts[1]

                if original_name not in backup_groups:
                    backup_groups[original_name] = []

                try:
                    # Parse timestamp
                    timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                    backup_groups[original_name].append((backup_file, timestamp))
                except ValueError:
                    logger.warning(f"Could not parse timestamp for {backup_file.name}")

        # Find latest backup for each file
        for original_name, backups in backup_groups.items():
            if backups:
                # Sort by timestamp (newest first)
                backups.sort(key=lambda x: x[1], reverse=True)
                latest_backup, latest_timestamp = backups[0]
                latest_backups[original_name] = latest_backup

                logger.info(f"Latest backup for {original_name}: {latest_backup.name} ({latest_timestamp})")

        self.stats['backups_found'] = len(latest_backups)
        return latest_backups

    def restore_files(self, latest_backups: Dict[str, Path], dry_run: bool = False) -> bool:
        """Restore files from latest backups"""
        if not latest_backups:
            logger.info("No backups found to restore")
            return True

        success = True
        self.stats['files_to_restore'] = len(latest_backups)

        for original_name, backup_file in latest_backups.items():
            target_file = self.data_dir / original_name

            if dry_run:
                logger.info(f"[DRY RUN] Would restore {original_name} from {backup_file.name}")
                continue

            try:
                # Create backup of current file before restoring
                if target_file.exists():
                    current_backup_name = f"{target_file.stem}.pre_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}{target_file.suffix}"
                    current_backup_path = self.backup_dir / current_backup_name
                    shutil.copy2(target_file, current_backup_path)
                    logger.info(f"Backed up current {original_name} to {current_backup_name}")

                # Restore from backup
                shutil.copy2(backup_file, target_file)
                logger.info(f"Restored {original_name} from backup")
                self.stats['files_restored'] += 1

            except Exception as e:
                logger.error(f"Failed to restore {original_name}: {e}")
                success = False

        return success

    def print_summary(self, latest_backups: Dict[str, Path]):
        """Print restoration summary"""
        print("\n" + "="*60)
        print("🔄 HISTORICAL DATA BACKUP RESTORATION - SUMMARY")
        print("="*60)

        print(f"\n🔍 BACKUP SCAN RESULTS:")
        print(f"   Backup files found: {self.stats['backups_found']}")
        print(f"   Files to restore: {self.stats['files_to_restore']}")

        if latest_backups:
            print(f"\n📁 FILES TO BE RESTORED:")
            for original_name, backup_file in latest_backups.items():
                print(f"   • {original_name} ← {backup_file.name}")

            print(f"\n🔄 RESTORATION RESULTS:")
            print(f"   Files restored: {self.stats['files_restored']}")
            print(f"   Files skipped: {self.stats['files_skipped']}")

        print(f"\n📄 Detailed log saved to: scripts/restore_report.txt")
        print("="*60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Restore Historical Data files from backup")
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be restored without actually restoring')
    parser.add_argument('--restore', action='store_true',
                       help='Restore all files from their latest backups')
    parser.add_argument('--data-dir', type=str, default='data/Historical_Data',
                       help='Directory containing CSV files (default: data/Historical_Data)')

    args = parser.parse_args()

    # Validate arguments
    if not any([args.dry_run, args.restore]):
        print("❌ Please specify --dry-run or --restore")
        parser.print_help()
        return 1

    # Initialize restorer
    restorer = BackupRestorer(args.data_dir)

    print("🔄 Historical Data Backup Restorer")
    print("="*40)

    try:
        # Find latest backups
        print("🔍 Scanning for latest backups...")
        latest_backups = restorer.find_latest_backups()

        if not latest_backups:
            print("❌ No backups found!")
            return 1

        # Show summary
        restorer.print_summary(latest_backups)

        if args.dry_run:
            print("🔍 DRY RUN - No files were actually restored")
            return 0

        if args.restore:
            print(f"\n🔄 Restoring {len(latest_backups)} files...")
            success = restorer.restore_files(latest_backups, dry_run=False)

            if success:
                print("✅ All files restored successfully!")
                restorer.print_summary(latest_backups)
                return 0
            else:
                print("❌ Some files failed to restore")
                return 1

    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
