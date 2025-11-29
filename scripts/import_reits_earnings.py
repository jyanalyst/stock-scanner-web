"""
Import REIT earnings report data from Claude-generated .txt files
Dedicated script for new REITs data format with enhanced fields
Filename format: YYYYMMDD_CompanyAbbrev_Quarter_FiscalYear_ReportTime_Ticker.txt
Example: 20251023_FCPT_H2_FY2025_AM_J69U.txt
Each .txt file should contain a JSON object with REIT earnings report data
Successfully processed files are automatically moved to processed/ subfolder

NEW: Automatically sets company_type to 'reit' for all imported data
NEW: Supports enhanced REIT data structure with additional fields
"""
import json
from pathlib import Path
from datetime import datetime
import sys
import re
import shutil

sys.path.append(str(Path(__file__).parent.parent))
from utils.paths import EARNINGS_REPORTS_DIR, EARNINGS_PDF_DIR


def parse_quarter_to_report_type(quarter_str: str) -> str:
    """
    Convert quarter format to standardized report_type

    Examples:
        1Q -> Q1
        2Q -> Q2
        H1 -> H1
        H2 -> H2
        FY -> FY

    Returns:
        Standardized report_type: Q1, Q2, Q3, Q4, H1, H2, or FY
    """
    quarter_upper = quarter_str.upper()

    # Quarter patterns: 1Q, 2Q, 3Q, 4Q -> Q1, Q2, Q3, Q4
    if re.match(r'^[1-4]Q$', quarter_upper):
        quarter_num = quarter_upper[0]
        return f"Q{quarter_num}"

    # Half-year patterns: H1, H2 -> H1, H2 (already correct)
    if re.match(r'^H[1-2]$', quarter_upper):
        return quarter_upper

    # Full year pattern: FY -> FY (already correct)
    if quarter_upper == 'FY':
        return 'FY'

    # Default to the original string if no pattern matches
    return quarter_str


def parse_filename(filename: str) -> dict:
    """
    Parse filename to extract date, company, quarter, fiscal year, report time, and ticker

    Expected format: YYYYMMDD_CompanyAbbrev_Quarter_FiscalYear_ReportTime_Ticker.txt
    Example: 20251023_FCPT_H2_FY2025_AM_J69U.txt

    Returns:
        dict with parsed components or None if invalid
    """
    # Remove .txt extension
    basename = filename.replace('.txt', '')

    # Pattern: YYYYMMDD_CompanyAbbrev_Quarter_FiscalYear_ReportTime_Ticker
    # Date: 8 digits
    # Company: any characters (non-greedy)
    # Quarter: 1Q, 2Q, 3Q, 4Q, H1, H2, or FY
    # FiscalYear: FY followed by 4 digits (e.g., FY2025)
    # ReportTime: AM or PM
    # Ticker: uppercase letters/numbers
    pattern = r'^(\d{8})_(.+?)_([1-4]Q|H[1-2]|FY)_(FY\d{4})_(AM|PM)_([A-Z0-9]+)$'

    match = re.match(pattern, basename)

    if not match:
        return None

    date_str, company, quarter, fiscal_year, report_time, ticker = match.groups()

    # Validate date
    try:
        date_obj = datetime.strptime(date_str, '%Y%m%d')
        date_formatted = date_obj.strftime('%Y-%m-%d')
    except ValueError:
        return None

    # Convert quarter to report_type
    report_type = parse_quarter_to_report_type(quarter)

    return {
        'date': date_formatted,
        'company': company,
        'quarter': quarter,
        'fiscal_year': fiscal_year,
        'report_time': report_time,
        'ticker': ticker,
        'report_type': report_type
    }


def validate_reits_json_data(data: dict, filename_info: dict) -> tuple:
    """
    Validate REIT JSON data against filename and REIT-specific requirements

    Returns:
        (is_valid, error_message)
    """
    # Check required base fields
    required_fields = ['ticker', 'report_date', 'report_type', 'report_time', 'fiscal_year']
    missing_fields = [f for f in required_fields if f not in data]

    if missing_fields:
        return False, f"Missing required fields: {missing_fields}"

    # Validate ticker matches filename
    if data['ticker'] != filename_info['ticker']:
        return False, f"Ticker mismatch: JSON has '{data['ticker']}' but filename has '{filename_info['ticker']}'"

    # Validate date matches filename
    if data['report_date'] != filename_info['date']:
        return False, f"Date mismatch: JSON has '{data['report_date']}' but filename has '{filename_info['date']}'"

    # Validate report_type matches derived from filename
    if data['report_type'] != filename_info['report_type']:
        return False, f"Report type mismatch: JSON has '{data['report_type']}' but filename quarter '{filename_info['quarter']}' maps to '{filename_info['report_type']}'"

    # Validate report_time matches filename
    if data['report_time'] != filename_info['report_time']:
        return False, f"Report time mismatch: JSON has '{data['report_time']}' but filename has '{filename_info['report_time']}'"

    # Validate report_time is strictly AM or PM
    if data['report_time'] not in ['AM', 'PM']:
        return False, f"Invalid report_time: must be 'AM' or 'PM', got '{data['report_time']}'"

    # Validate fiscal_year matches filename
    if data['fiscal_year'] != filename_info['fiscal_year']:
        return False, f"Fiscal year mismatch: JSON has '{data['fiscal_year']}' but filename has '{filename_info['fiscal_year']}'"

    # REIT-specific validation - check for essential REIT fields
    reit_required_fields = ['dpu', 'gearing_ratio']
    missing_reit = [f for f in reit_required_fields if f not in data]
    if missing_reit:
        return False, f"Missing REIT-specific fields: {missing_reit}"

    # Check for REIT name and sector (new fields)
    if 'reit_name' not in data:
        return False, "Missing REIT name field"

    if 'reit_sector' not in data:
        return False, "Missing REIT sector field"

    return True, None


def import_reits_txt_file(txt_path: Path):
    """Import a single REIT .txt file with JSON data"""
    print(f"Processing: {txt_path.name}")

    # Parse filename
    filename_info = parse_filename(txt_path.name)

    if not filename_info:
        print(f"  ✗ Error: Invalid filename format")
        print(f"    Expected: YYYYMMDD_CompanyAbbrev_Quarter_FiscalYear_ReportTime_Ticker.txt")
        print(f"    Example: 20251023_FCPT_H2_FY2025_AM_J69U.txt")
        print(f"    Quarter: 1Q, 2Q, 3Q, 4Q, H1, H2, or FY")
        print(f"    ReportTime: AM or PM")
        return False

    print(f"  Parsed filename: {filename_info['date']} | {filename_info['company']} | {filename_info['ticker']}")
    print(f"  Quarter: {filename_info['quarter']} → Report Type: {filename_info['report_type']}")
    print(f"  Fiscal Year: {filename_info['fiscal_year']} | Report Time: {filename_info['report_time']}")

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
        try:
            json_str = content.split('```json')[1].split('```')[0].strip()
        except IndexError:
            print(f"  ✗ Error: Malformed JSON code block")
            return False
    elif '```' in content:
        try:
            json_str = content.split('```')[1].split('```')[0].strip()
        except IndexError:
            print(f"  ✗ Error: Malformed code block")
            return False
    else:
        json_str = content.strip()

    # Parse JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"  ✗ Error: Invalid JSON format - {e}")
        print(f"    Check your JSON syntax")
        return False

    # Validate REIT data against filename
    is_valid, error_msg = validate_reits_json_data(data, filename_info)

    if not is_valid:
        print(f"  ✗ Error: Validation failed - {error_msg}")
        return False

    # Automatically add company_type for REITs
    data['company_type'] = 'reit'
    print(f"  ✓ Auto-set company_type: REIT")

    # Add metadata
    data['upload_date'] = datetime.now().isoformat()
    data['analysis_method'] = 'claude_analysis'
    data['report_age_days'] = 0  # Will be calculated by scanner
    data['pdf_filename'] = txt_path.name.replace('.txt', '.pdf')

    # Generate output filename: Ticker_YYYY-MM-DD_ReportType.json
    ticker = data['ticker']
    report_date = data['report_date']
    report_type = data['report_type']
    output_filename = f"{ticker}_{report_date}_{report_type}.json"
    output_path = EARNINGS_REPORTS_DIR / output_filename

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
    print(f"  Company Type: REIT | Report Type: {report_type} | Guidance: {data.get('guidance_tone', 'unknown').upper()}")
    print(f"  Fiscal Year: {data['fiscal_year']} | Report Time: {data['report_time']}")

    # Show key REIT metrics
    reit_name = data.get('reit_name', 'Unknown REIT')
    reit_sector = data.get('reit_sector', 'Unknown Sector')
    dpu = data.get('dpu')
    dpu_change = data.get('dpu_yoy_change')
    gearing = data.get('gearing_ratio')
    nav = data.get('nav_per_unit')

    print(f"  REIT: {reit_name} ({reit_sector})")
    print(f"  REIT Metrics:")

    if dpu is not None:
        print(f"    • DPU: {dpu:.2f} cents", end="")
        if dpu_change is not None:
            print(f" (YoY: {dpu_change:+.1f}%)")
        else:
            print()

    if gearing is not None:
        print(f"    • Gearing: {gearing:.1f}%")

    if nav is not None:
        print(f"    • NAV/Unit: ${nav:.2f}")

    revenue_change = data.get('revenue_yoy_change')
    if revenue_change is not None:
        print(f"    • Revenue YoY: {revenue_change:+.1f}%")

    # Show additional REIT-specific info
    portfolio_occupancy = data.get('portfolio_occupancy')
    if portfolio_occupancy is not None:
        print(f"    • Portfolio Occupancy: {portfolio_occupancy:.1f}%")

    number_of_properties = data.get('number_of_properties')
    if number_of_properties is not None:
        print(f"    • Properties: {number_of_properties}")

    return True


def move_to_processed(txt_path: Path, processed_dir: Path) -> bool:
    """Move processed .txt file to processed subfolder"""
    try:
        destination = processed_dir / txt_path.name
        shutil.move(str(txt_path), str(destination))
        return True
    except Exception as e:
        print(f"  ⚠️  Warning: Could not move to processed folder - {e}")
        return False


def main():
    """Import all REIT .txt files from earnings_reports_pdf folder"""
    print("=" * 60)
    print("CLAUDE REIT EARNINGS REPORT IMPORTER")
    print("Dedicated script for new REITs data format")
    print("=" * 60)
    print()

    # Create processed subfolder
    processed_dir = EARNINGS_PDF_DIR / "processed"
    processed_dir.mkdir(exist_ok=True)

    # Look for .txt files
    txt_files = list(EARNINGS_PDF_DIR.glob("*.txt"))

    if not txt_files:
        print(f"No .txt files found in: {EARNINGS_PDF_DIR}")
        print("\nPlease add Claude-generated REIT .txt files to this folder.")
        print("\nFilename format: YYYYMMDD_CompanyAbbrev_Quarter_FiscalYear_ReportTime_Ticker.txt")
        print("Example: 20251023_FCPT_H2_FY2025_AM_J69U.txt")
        print("  • Quarter: 1Q, 2Q, 3Q, 4Q, H1, H2, or FY")
        print("  • ReportTime: AM (before market) or PM (after market)")
        print("\nREIT data should include fields like:")
        print("  • reit_name, reit_sector, dpu, gearing_ratio, nav_per_unit")
        print("  • key_highlights, key_concerns, recent_acquisitions, etc.")
        return

    print(f"Found {len(txt_files)} .txt file(s) to import\n")

    processed = 0
    failed = 0
    moved = 0
    failed_files = []

    for txt_file in txt_files:
        try:
            if import_reits_txt_file(txt_file):
                processed += 1

                if move_to_processed(txt_file, processed_dir):
                    print(f"  📁 Moved to: processed/{txt_file.name}")
                    moved += 1
            else:
                failed += 1
                failed_files.append(txt_file.name)
                print(f"  📌 Kept in main folder for retry")
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            failed += 1
            failed_files.append(txt_file.name)
            print(f"  📌 Kept in main folder for retry")
        print()

    print("=" * 60)
    print(f"SUMMARY: {processed} imported, {failed} failed")
    print("=" * 60)

    if moved > 0:
        print(f"\n✓ Automatically moved {moved} file(s) to: {processed_dir}")

    if failed > 0:
        print(f"\n⚠️  {failed} FAILED FILE(S):")
        for filename in failed_files:
            print(f"  • {filename}")
        print("\nCommon issues:")
        print("  1. Filename not in format: YYYYMMDD_CompanyAbbrev_Quarter_FiscalYear_ReportTime_Ticker.txt")
        print("  2. JSON ticker/date doesn't match filename")
        print("  3. Missing required fields: report_time or fiscal_year")
        print("  4. Invalid report_time (must be 'AM' or 'PM')")
        print("  5. Missing REIT-specific fields (dpu, gearing_ratio, reit_name, reit_sector)")
        print("  6. File not saved with UTF-8 encoding")

    if processed > 0:
        print(f"\n✓ JSON files saved to: {EARNINGS_REPORTS_DIR}")
        print("✓ Using CLAUDE REIT ANALYSIS (Enhanced REIT Data)")
        print("✓ Auto-set company_type to 'reit' for all imports")
        print("✓ New format with Fiscal Year and Report Time (AM/PM)")
        print("\nYou can now view these REIT reports in the scanner!")


if __name__ == "__main__":
    main()
