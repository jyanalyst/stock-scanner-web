"""
Backup all data in Codespace to a zip file for download.
Run this weekly to backup your data.
"""
import zipfile
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.paths import DATA_DIR, BACKUP_DIR


def create_backup():
    """Create a backup zip file of all data"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_filename = f"scanner_backup_{timestamp}.zip"
    backup_path = BACKUP_DIR / backup_filename
    
    print(f"Creating backup: {backup_filename}")
    print(f"Source: {DATA_DIR}")
    print(f"Destination: {backup_path}")
    print()
    
    # Create zip file
    with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through data directory
        file_count = 0
        for file_path in DATA_DIR.rglob('*'):
            if file_path.is_file():
                # Add file to zip with relative path
                arcname = file_path.relative_to(DATA_DIR.parent)
                zipf.write(file_path, arcname)
                file_count += 1
                
                # Progress indicator
                if file_count % 10 == 0:
                    print(f"  Backed up {file_count} files...", end='\r')
        
        print(f"  ✓ Backed up {file_count} files successfully")
    
    # Get file size
    size_mb = backup_path.stat().st_size / (1024 * 1024)
    
    print()
    print("=" * 60)
    print(f"✓ Backup created successfully!")
    print(f"  File: {backup_filename}")
    print(f"  Size: {size_mb:.2f} MB")
    print(f"  Location: {backup_path}")
    print("=" * 60)
    print()
    print("To download:")
    print("  1. Right-click the file in VS Code Explorer")
    print("  2. Select 'Download'")
    print("  3. Save to your local machine")
    
    return backup_path


if __name__ == "__main__":
    create_backup()