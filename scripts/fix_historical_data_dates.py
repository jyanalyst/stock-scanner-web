#!/usr/bin/env python3
"""
Historical Data Date Format Fixer
==================================

Scans and fixes date format inconsistencies in Historical_Data CSV files.

PROBLEM:
- Some CSV files contain dates in DD/M/YYYY format (e.g., "13/1/2025")
- Code expects MM/DD/YYYY format (e.g., "01/13/2025")
- This causes parsing errors when trying to get price data for earnings dates

SOLUTION:
- Scan all CSV files in data/Historical_Data/
- Detect dates in DD/M/YYYY format
- Convert to MM/DD/YYYY format
- Create backups before changes
- Generate detailed reports

USAGE:
    python scripts/fix_historical_data_dates.py --scan-only    # Just report issues
    python scripts/fix_historical_data_dates.py --fix          # Fix all issues
    python scripts/fix_historical_data_dates.py --file J69U.csv  # Fix specific file
    python scripts/fix_historical_data_dates.py --dry-run       # Show what would be changed

AUTHOR: jyanalyst
DATE: 2025-11-22
"""

import os
import sys
import csv
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scripts/date_fix_report.txt', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DateFormatFixer:
    """Handles detection and fixing of date format issues in CSV files"""

    def __init__(self, data_dir: str = "data/Historical_Data"):
        self.data_dir = Path(data_dir)
        self.backup_dir = self.data_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)

        # Statistics
        self.stats = {
            'files_scanned': 0,
            'files_with_issues': 0,
            'total_dates_fixed': 0,
            'files_backed_up': 0,
            'files_fixed': 0
        }

    def scan_file(self, csv_file: Path) -> Tuple[bool, List[Dict], int]:
        """
        Scan a CSV file for date format issues

        Returns:
            (has_issues, problematic_rows, total_rows)
        """
        problematic_rows = []
        total_rows = 0

        try:
            with open(csv_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row_num, row in enumerate(reader, start=2):  # Start at 2 (header + 1)
                    total_rows += 1
                    date_str = row.get('Date', '').strip()

                    if self._is_dd_mm_yyyy_format(date_str):
                        problematic_rows.append({
                            'row': row_num,
                            'original_date': date_str,
                            'fixed_date': self._convert_date_format(date_str),
                            'full_row': row
                        })

        except Exception as e:
            logger.error(f"Error scanning {csv_file.name}: {e}")
            return False, [], 0

        has_issues = len(problematic_rows) > 0
        return has_issues, problematic_rows, total_rows

    def _is_dd_mm_yyyy_format(self, date_str: str) -> bool:
        """Check if date is in DD/M/YYYY format (problematic)"""
        if not date_str:
            return False

        try:
            # Split by '/'
            parts = date_str.split('/')
            if len(parts) != 3:
                return False

            day, month, year = map(int, parts)

            # Check if day > 12 (indicates DD/MM/YYYY format)
            # Also validate ranges
            if day > 12 and 1 <= month <= 12 and year >= 1900:
                return True

        except (ValueError, IndexError):
            pass

        return False

    def _convert_date_format(self, date_str: str) -> str:
        """Convert DD/M/YYYY to M/D/YYYY format"""
        try:
            parts = date_str.split('/')
            if len(parts) == 3:
                day, month, year = map(int, parts)
                # Convert to MM/DD/YYYY
                return "02d"
        except (ValueError, IndexError):
            pass

        return date_str  # Return original if conversion fails

    def create_backup(self, csv_file: Path) -> bool:
        """Create backup of the file"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"{csv_file.stem}.backup_{timestamp}{csv_file.suffix}"
            backup_path = self.backup_dir / backup_name

            shutil.copy2(csv_file, backup_path)
            logger.info(f"Created backup: {backup_path}")
            self.stats['files_backed_up'] += 1
            return True

        except Exception as e:
            logger.error(f"Failed to create backup for {csv_file.name}: {e}")
            return False

    def fix_file(self, csv_file: Path, problematic_rows: List[Dict]) -> bool:
        """Fix date format issues in a CSV file"""
        try:
            # Read all rows
            rows = []
            with open(csv_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                rows.append(dict(zip(fieldnames, fieldnames)))  # Header row

                for row in reader:
                    rows.append(row)

            # Fix problematic dates
            for issue in problematic_rows:
                row_idx = issue['row'] - 1  # Convert to 0-based index
                if row_idx < len(rows):
                    rows[row_idx]['Date'] = issue['fixed_date']
                    self.stats['total_dates_fixed'] += 1

            # Write back to file
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                for row in rows:
                    writer.writerow(row)

            logger.info(f"Fixed {len(problematic_rows)} dates in {csv_file.name}")
            self.stats['files_fixed'] += 1
            return True

        except Exception as e:
            logger.error(f"Failed to fix {csv_file.name}: {e}")
            return False

    def scan_all_files(self) -> Dict[str, List[Dict]]:
        """Scan all CSV files and return issues found"""
        issues_found = {}

        if not self.data_dir.exists():
            logger.error(f"Data directory not found: {self.data_dir}")
            return issues_found

        csv_files = list(self.data_dir.glob("*.csv"))
        csv_files = [f for f in csv_files if not f.name.startswith('.') and 'backup' not in f.name.lower()]

        logger.info(f"Found {len(csv_files)} CSV files to scan")

        for csv_file in csv_files:
            self.stats['files_scanned'] += 1
            logger.info(f"Scanning {csv_file.name}...")

            has_issues, problematic_rows, total_rows = self.scan_file(csv_file)

            if has_issues:
                issues_found[csv_file.name] = {
                    'file_path': csv_file,
                    'total_rows': total_rows,
                    'problematic_rows': problematic_rows,
                    'issue_count': len(problematic_rows)
                }
                self.stats['files_with_issues'] += 1

                logger.warning(f"Found {len(problematic_rows)} date issues in {csv_file.name}")

        return issues_found

    def fix_all_files(self, issues_found: Dict[str, List[Dict]], dry_run: bool = False) -> bool:
        """Fix all files with issues"""
        if not issues_found:
            logger.info("No files need fixing")
            return True

        success = True

        for file_name, file_data in issues_found.items():
            csv_file = file_data['file_path']
            problematic_rows = file_data['problematic_rows']

            logger.info(f"Processing {file_name} ({len(problematic_rows)} issues)...")

            if not dry_run:
                # Create backup
                if not self.create_backup(csv_file):
                    logger.error(f"Skipping {file_name} due to backup failure")
                    success = False
                    continue

                # Fix the file
                if not self.fix_file(csv_file, problematic_rows):
                    logger.error(f"Failed to fix {file_name}")
                    success = False
            else:
                logger.info(f"[DRY RUN] Would fix {len(problematic_rows)} dates in {file_name}")

        return success

    def print_summary(self, issues_found: Dict):
        """Print detailed summary"""
        print("\n" + "="*60)
        print("📅 HISTORICAL DATA DATE FORMAT FIXER - SUMMARY")
        print("="*60)

        print(f"\n🔍 SCAN RESULTS:")
        print(f"   Files scanned: {self.stats['files_scanned']}")
        print(f"   Files with issues: {self.stats['files_with_issues']}")
        print(f"   Total dates to fix: {sum(len(data['problematic_rows']) for data in issues_found.values())}")

        if issues_found:
            print(f"\n⚠️  FILES WITH DATE ISSUES:")
            for file_name, data in issues_found.items():
                print(f"   • {file_name}: {len(data['problematic_rows'])} problematic dates")

            print(f"\n💾 BACKUP RESULTS:")
            print(f"   Backups created: {self.stats['files_backed_up']}")

            print(f"\n🔧 FIX RESULTS:")
            print(f"   Files fixed: {self.stats['files_fixed']}")
            print(f"   Total dates fixed: {self.stats['total_dates_fixed']}")

        print(f"\n📄 Detailed log saved to: scripts/date_fix_report.txt")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Fix date format issues in Historical_Data CSV files")
    parser.add_argument('--scan-only', action='store_true',
                       help='Only scan and report issues, do not fix')
    parser.add_argument('--fix', action='store_true',
                       help='Scan and fix all issues')
    parser.add_argument('--file', type=str,
                       help='Fix only the specified file (e.g., J69U.csv)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be changed without actually changing files')
    parser.add_argument('--data-dir', type=str, default='data/Historical_Data',
                       help='Directory containing CSV files (default: data/Historical_Data)')

    args = parser.parse_args()

    # Validate arguments
    if not any([args.scan_only, args.fix, args.file]):
        print("❌ Please specify --scan-only, --fix, or --file")
        parser.print_help()
        return 1

    if args.dry_run and not (args.fix or args.file):
        print("❌ --dry-run must be used with --fix or --file")
        return 1

    # Initialize fixer
    fixer = DateFormatFixer(args.data_dir)

    print("📅 Historical Data Date Format Fixer")
    print("="*50)

    try:
        if args.file:
            # Fix specific file
            csv_file = Path(args.data_dir) / args.file
            if not csv_file.exists():
                print(f"❌ File not found: {csv_file}")
                return 1

            print(f"🔍 Scanning {args.file}...")
            has_issues, problematic_rows, total_rows = fixer.scan_file(csv_file)

            if not has_issues:
                print(f"✅ No issues found in {args.file}")
                return 0

            print(f"⚠️  Found {len(problematic_rows)} date issues in {args.file}")

            if args.dry_run:
                print("🔍 DRY RUN - Would fix the following dates:")
                for issue in problematic_rows[:5]:  # Show first 5
                    print(f"   Row {issue['row']}: {issue['original_date']} → {issue['fixed_date']}")
                if len(problematic_rows) > 5:
                    print(f"   ... and {len(problematic_rows) - 5} more")
            else:
                # Create backup and fix
                if fixer.create_backup(csv_file):
                    if fixer.fix_file(csv_file, problematic_rows):
                        print(f"✅ Successfully fixed {args.file}")
                        fixer.print_summary({args.file: {
                            'file_path': csv_file,
                            'problematic_rows': problematic_rows,
                            'total_rows': total_rows,
                            'issue_count': len(problematic_rows)
                        }})
                    else:
                        print(f"❌ Failed to fix {args.file}")
                        return 1
                else:
                    print(f"❌ Failed to create backup for {args.file}")
                    return 1

        else:
            # Scan all files
            print("🔍 Scanning all CSV files...")
            issues_found = fixer.scan_all_files()

            if args.scan_only:
                fixer.print_summary(issues_found)
                return 0

            if args.fix:
                if issues_found:
                    print(f"\n🔧 Fixing {len(issues_found)} files...")
                    success = fixer.fix_all_files(issues_found, dry_run=args.dry_run)
                    fixer.print_summary(issues_found)

                    if success:
                        print("✅ All files processed successfully!")
                        return 0
                    else:
                        print("❌ Some files failed to process")
                        return 1
                else:
                    print("✅ No files need fixing!")
                    return 0

    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
