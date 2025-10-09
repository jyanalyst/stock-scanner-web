```python
# File: scripts/import_analyst_txt.py
"""
Import analyst report data from Claude-generated .txt files
Filename format: YYYYMMDD_Company_Ticker.txt
Each .txt file should contain a JSON object with analyst report data
"""
import json
from pathlib import Path
from datetime import datetime
import sys
import re

sys.path.append(str(Path(__file__).parent.parent))
from utils.paths import ANALYST_REPORTS_DIR, ANALYST_PDF_DIR


def parse_filename(filename: str) -> dict:
    """
    Parse filename to extract date and ticker
    
    Expected format: YYYYMMDD_Company_Ticker.txt
    Example: 20251008_Wilmar_F34.txt
    
    Returns:
        dict with 'date', 'company', 'ticker' or None if invalid
    """
    # Remove .txt extension
    basename = filename.replace('.txt', '')
    
    # Pattern: YYYYMMDD_Company_Ticker
    # Date: 8 digits
    # Company: any characters
    # Ticker: uppercase letters/numbers
    pattern = r'^(\d{8})_(.+?)_([A-Z0-9]+)$'
    
    match = re.match(pattern, basename)
    
    if not match:
        return None
    
    date_str, company, ticker = match.groups()
    
    # Validate date
    try:
        date_obj = datetime.strptime(date_str, '%Y%m%d')
        date_formatted = date_obj.strftime('%Y-%m-%d')
    except ValueError:
        return None
    
    return {
        'date': date_formatted,
        'company': company,
        'ticker': ticker,
        'date_raw': date_str
    }


def validate_json_data(data: dict, filename_info: dict) -> tuple:
    """
    Validate JSON data against filename
    
    Returns:
        (is_valid, error_message)
    """
    # Check required fields
    required_fields = ['ticker', 'ticker_sgx', 'report_date', 'sentiment_score', 'sentiment_label']
    missing_fields = [f for f in required_fields if f not in data]
    
    if missing_fields:
        return False, f"Missing required fields: {missing_fields}"
    
    # Validate ticker matches filename
    if data['ticker'] != filename_info['ticker']:
        return False, f"Ticker mismatch: JSON has '{data['ticker']}' but filename has '{filename_info['ticker']}'"
    
    # Validate date matches filename
    if data['report_date'] != filename_info['date']:
        return False, f"Date mismatch: JSON has '{data['report_date']}' but filename has '{filename_info['date']}'"
    
    # Validate ticker_sgx format
    if not data['ticker_sgx'].endswith('.SG'):
        return False, f"ticker_sgx must end with .SG, got: {data['ticker_sgx']}"
    
    if data['ticker_sgx'] != f"{data['ticker']}.SG":
        return False, f"ticker_sgx should be '{data['ticker']}.SG', got: {data['ticker_sgx']}"
    
    return True, None


def import_txt_file(txt_path: Path):
    """Import a single .txt file with JSON data"""
    print(f"Processing: {txt_path.name}")
    
    # Parse filename
    filename_info = parse_filename(txt_path.name)
    
    if not filename_info:
        print(f"  ✗ Error: Invalid filename format")
        print(f"    Expected: YYYYMMDD_Company_Ticker.txt")
        print(f"    Example: 20251008_Wilmar_F34.txt")
        return False
    
    print(f"  Parsed filename: {filename_info['date']} | {filename_info['company']} | {filename_info['ticker']}")
    
    # Read the file with UTF-8 encoding
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        print(f"  ✗ Error: File is not UTF-8 encoded")
        print(f"    Please save the file with UTF-8 encoding")
        return False
    
    # Extract JSON from code block if present
    if '```json' in content:
        # Extract content between ```json and ```
        try:
            json_str = content.split('```json')[1].split('```')[0].strip()
        except IndexError:
            print(f"  ✗ Error: Malformed JSON code block")
            return False
    elif '```' in content:
        # Extract content between ``` and ```
        try:
            json_str = content.split('```')[1].split('```')[0].strip()
        except IndexError:
            print(f"  ✗ Error: Malformed code block")
            return False
    else:
        # Assume entire file is JSON
        json_str = content.strip()
    
    # Parse JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"  ✗ Error: Invalid JSON format - {e}")
        print(f"    Check your JSON syntax")
        return False
    
    # Validate data against filename
    is_valid, error_msg = validate_json_data(data, filename_info)
    
    if not is_valid:
        print(f"  ✗ Error: Validation failed - {error_msg}")
        return False
    
    # Add metadata
    data['upload_date'] = datetime.now().isoformat()
    data['sentiment_method'] = 'claude_analysis'
    data['report_age_days'] = 0  # Will be calculated by scanner
    data['pdf_filename'] = txt_path.name.replace('.txt', '.pdf')
    
    # Generate output filename: Ticker_YYYY-MM-DD.json
    ticker = data['ticker']
    report_date = data['report_date']
    output_filename = f"{ticker}_{report_date}.json"
    output_path = ANALYST_REPORTS_DIR / output_filename
    
    # Check if already exists
    if output_path.exists():
        print(f"  ⚠️  Warning: {output_filename} already exists")
        overwrite = input("    Overwrite? (y/n): ").strip().lower()
        if overwrite != 'y':
            print(f"  ⏭️  Skipped")
            return False
    
    # Save JSON with UTF-8 encoding
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Saved: {output_filename}")
    print(f"  Sentiment: {data['sentiment_label'].upper()} ({data['sentiment_score']:+.2f})")
    
    # Show upgrade/downgrade if present
    if 'previous_recommendation' in data and data['previous_recommendation']:
        print(f"  Rating change: {data['previous_recommendation']} → {data['recommendation']}")
    
    return True


def main():
    """Import all .txt files from analyst_reports_pdf folder"""
    print("=" * 60)
    print("CLAUDE ANALYST REPORT IMPORTER")
    print("=" * 60)
    print()
    
    # Look for .txt files in PDF directory
    txt_files = list(ANALYST_PDF_DIR.glob("*.txt"))
    
    if not txt_files:
        print(f"No .txt files found in: {ANALYST_PDF_DIR}")
        print("\nPlease add Claude-generated .txt files to this folder.")
        print("Each file should contain JSON output from Claude's analysis.")
        print("\nFilename format: YYYYMMDD_Company_Ticker.txt")
        print("Example: 20251008_Wilmar_F34.txt")
        return
    
    print(f"Found {len(txt_files)} .txt file(s) to import\n")
    
    processed = 0
    failed = 0
    failed_files = []
    
    for txt_file in txt_files:
        try:
            if import_txt_file(txt_file):
                processed += 1
            else:
                failed += 1
                failed_files.append(txt_file.name)
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            failed += 1
            failed_files.append(txt_file.name)
        print()
    
    print("=" * 60)
    print(f"SUMMARY: {processed} imported, {failed} failed")
    print("=" * 60)
    
    if failed > 0:
        print("\n⚠️  FAILED FILES:")
        for filename in failed_files:
            print(f"  • {filename}")
        print("\nCommon issues:")
        print("  1. Filename not in format: YYYYMMDD_Company_Ticker.txt")
        print("  2. JSON ticker/date doesn't match filename")
        print("  3. File not saved with UTF-8 encoding")
        print("  4. Invalid JSON syntax")
    
    if processed > 0:
        print(f"\n✓ JSON files saved to: {ANALYST_REPORTS_DIR}")
        print("✓ Using CLAUDE ANALYSIS sentiment scores")
        print("\nYou can now view these in the scanner!")
        
        print("\n📝 Optional: Move processed .txt files to avoid re-importing:")
        print(f"  mkdir -p {ANALYST_PDF_DIR}/processed")
        print(f"  mv {ANALYST_PDF_DIR}/*.txt {ANALYST_PDF_DIR}/processed/")


if __name__ == "__main__":
    main()