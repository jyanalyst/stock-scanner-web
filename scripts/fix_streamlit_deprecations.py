#!/usr/bin/env python3
"""
Fix Streamlit use_container_width deprecation warnings
Replaces use_container_width=True with width="stretch"
Replaces use_container_width=False with width="content"
"""

import re
import os

def fix_file(file_path):
    """Fix a single file"""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return 0

    # Read file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # Replace use_container_width=True with width="stretch"
    content = re.sub(r'use_container_width=True', 'width="stretch"', content)

    # Replace use_container_width=False with width="content"
    content = re.sub(r'use_container_width=False', 'width="content"', content)

    # Write back if changed
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        # Count changes
        changes = (original_content.count('use_container_width=True') - content.count('use_container_width=True') +
                  original_content.count('use_container_width=False') - content.count('use_container_width=False'))
        print(f"✅ Fixed {changes} instances in {file_path}")
        return changes
    else:
        print(f"ℹ️ No changes needed in {file_path}")
        return 0

def main():
    """Main function"""
    # List of files to fix
    files_to_fix = [
        'pages/scanner/ui.py',
        'pages/scanner/data.py',
        'pages/scanner/ui_flow_analysis.py',
        'pages/scanner/feature_lab/ui_components.py',
        'pages/factor_analysis.py',
        'pages/earnings_trend_analyzer.py',
        'pages/earnings_reports_analysis.py',
        'pages/common/ui_components.py',
        'pages/analyst_reports_analysis.py',
        'pages/rvol_backtest/ui.py',
        'pages/reit_analysis.py',
        'pages/ml_lab.py',
        'pages/ml_lab_phase3.py',
        'pages/ml_lab_phase4.py'
    ]

    total_fixed = 0

    print("🔧 Fixing Streamlit use_container_width deprecations...")
    print("=" * 60)

    for file_path in files_to_fix:
        changes = fix_file(file_path)
        total_fixed += changes

    print("=" * 60)
    print(f"🎉 Total fixed: {total_fixed} use_container_width deprecations across all files")
    print("\n📋 Summary:")
    print("  - use_container_width=True  → width=\"stretch\"")
    print("  - use_container_width=False → width=\"content\"")
    print("\n✅ All Streamlit deprecation warnings should now be resolved!")

if __name__ == "__main__":
    main()
