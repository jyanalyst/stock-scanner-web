# File: scripts/import_earnings_txt.py
"""
Import earnings report data from Claude-generated .txt files
Filename format: YYYYMMDD_CompanyAbbrev_Period_Ticker.txt
Example: 20220420_MCT_2HFY2122_N2IU.txt
Each .txt file should contain a JSON object with earnings report data
Successfully processed files are automatically moved to processed/ subfolder

UPDATED: Now supports adaptive structure for REITs, Business Trusts, and Normal Companies
"""
import json
from pathlib import Path
from datetime import datetime
import sys
import re
import shutil

sys.path.append(str(Path(__file__).parent.parent))
from utils.paths import EARNINGS_REPORTS_DIR, EARNINGS_PDF_DIR


def parse_period_to_report_type(period_str: str) -> str:
    """
    Convert various period formats to standardized report_type
    
    Examples:
        1QFY24 -> Q1
        2HFY2122 -> H2
        Q3FY25 -> Q3
        FY2024 -> FY
        1HFY24 -> H1
    
    Returns:
        Standardized report_type: Q1, Q2, Q3, Q4, H1, H2, or FY
    """
    period_upper = period_str.upper()
    
    # Quarter patterns
    if re.match(r'[1-4]Q', period_upper) or re.match(r'Q[1-4]', period_upper):
        quarter_num = re.search(r'[1-4]', period_upper).group()
        return f"Q{quarter_num}"
    
    # Half-year patterns
    if re.match(r'[1-2]H', period_upper) or re.match(r'H[1-2]', period_upper):
        half_num = re.search(r'[1-2]', period_upper).group()
        return f"H{half_num}"
    
    # Full year patterns
    if period_upper.startswith('FY') or period_upper.endswith('FY'):
        return "FY"
    
    # Default to the original string if no pattern matches
    return period_str


def parse_filename(filename: str) -> dict:
    """
    Parse filename to extract date, company, period, and ticker
    
    Expected format: YYYYMMDD_CompanyAbbrev_Period_Ticker.txt
    Example: 20220420_MCT_2HFY2122_N2IU.txt
    
    Returns:
        dict with 'date', 'company', 'period', 'ticker', 'report_type' or None if invalid
    """
    # Remove .txt extension
    basename = filename.replace('.txt', '')
    
    # Pattern: YYYYMMDD_CompanyAbbrev_Period_Ticker
    # Date: 8 digits
    # Company: letters/numbers
    # Period: alphanumeric (various formats)
    # Ticker: uppercase letters/numbers
    pattern = r'^(\d{8})_([A-Z0-9]+)_([A-Z0-9]+)_([A-Z0-9]+)$'
    
    match = re.match(pattern, basename)
    
    if not match:
        return None
    
    date_str, company, period, ticker = match.groups()
    
    # Validate date
    try:
        date_obj = datetime.strptime(date_str, '%Y%m%d')
        date_formatted = date_obj.strftime('%Y-%m-%d')
    except ValueError:
        return None
    
    # Convert period to standardized report_type
    report_type = parse_period_to_report_type(period)
    
    return {
        'date': date_formatted,
        'company': company,
        'period': period,
        'ticker': ticker,
        'report_type': report_type,
        'date_raw': date_str
    }


def validate_json_data(data: dict, filename_info: dict) -> tuple:
    """
    Validate JSON data against filename and company type
    
    Returns:
        (is_valid, error_message)
    """
    # Check company_type field exists
    if 'company_type' not in data:
        return False, "Missing required field: company_type"
    
    company_type = data['company_type']
    
    # Validate company_type values
    valid_company_types = ['reit', 'business_trust', 'normal']
    if company_type not in valid_company_types:
        return False, f"company_type must be one of {valid_company_types}, got: {company_type}"
    
    # Universal required fields (all company types)
    universal_required = [
        'ticker', 'ticker_sgx', 'report_date', 'report_type', 'fiscal_year',
        'revenue', 'eps', 'guidance_tone'
    ]
    
    # Check universal fields
    missing_fields = [f for f in universal_required if f not in data]
    if missing_fields:
        return False, f"Missing required universal fields: {missing_fields}"
    
    # Validate ticker matches filename
    if data['ticker'] != filename_info['ticker']:
        return False, f"Ticker mismatch: JSON has '{data['ticker']}' but filename has '{filename_info['ticker']}'"
    
    # Validate date matches filename
    if data['report_date'] != filename_info['date']:
        return False, f"Date mismatch: JSON has '{data['report_date']}' but filename has '{filename_info['date']}'"
    
    # Validate report type matches filename (allow some flexibility)
    if data['report_type'] != filename_info['report_type']:
        print(f"  ⚠️  Report type differs: JSON has '{data['report_type']}', filename suggests '{filename_info['report_type']}'")
        print(f"      Proceeding with JSON value: '{data['report_type']}'")
    
    # Validate ticker_sgx format
    if not data['ticker_sgx'].endswith('.SG'):
        return False, f"ticker_sgx must end with .SG, got: {data['ticker_sgx']}"
    
    if data['ticker_sgx'] != f"{data['ticker']}.SG":
        return False, f"ticker_sgx should be '{data['ticker']}.SG', got: {data['ticker_sgx']}"
    
    # Validate guidance_tone
    valid_guidance = ['positive', 'neutral', 'negative']
    if data['guidance_tone'] not in valid_guidance:
        return False, f"guidance_tone must be one of {valid_guidance}, got: {data['guidance_tone']}"
    
    # Type-specific validation
    if company_type in ['reit', 'business_trust']:
        # REIT/Business Trust specific fields that should exist
        reit_fields = ['dpu', 'net_property_income', 'gearing_ratio']
        missing_reit = [f for f in reit_fields if f not in data]
        if missing_reit:
            return False, f"Missing REIT-specific fields: {missing_reit}"
        
        # Normal company fields should be null for REITs
        normal_fields_should_be_null = ['gross_margin', 'operating_margin', 'net_margin']
        for field in normal_fields_should_be_null:
            if field in data and data[field] is not None:
                print(f"  ⚠️  Warning: {field} should be null for REIT/Business Trust, but has value: {data[field]}")
    
    elif company_type == 'normal':
        # Normal company specific fields that should exist
        normal_fields = ['gross_margin', 'operating_margin', 'net_margin', 'net_profit']
        missing_normal = [f for f in normal_fields if f not in data]
        if missing_normal:
            return False, f"Missing normal company fields: {missing_normal}"
        
        # REIT fields should be null for normal companies
        reit_fields_should_be_null = ['dpu', 'net_property_income', 'gearing_ratio', 'nav_per_unit']
        for field in reit_fields_should_be_null:
            if field in data and data[field] is not None:
                print(f"  ⚠️  Warning: {field} should be null for normal company, but has value: {data[field]}")
    
    return True, None


def import_txt_file(txt_path: Path):
    """Import a single .txt file with JSON data"""
    print(f"Processing: {txt_path.name}")
    
    # Parse filename
    filename_info = parse_filename(txt_path.name)
    
    if not filename_info:
        print(f"  ✗ Error: Invalid filename format")
        print(f"    Expected: YYYYMMDD_CompanyAbbrev_Period_Ticker.txt")
        print(f"    Example: 20220420_MCT_2HFY2122_N2IU.txt")
        return False
    
    print(f"  Parsed filename: {filename_info['date']} | {filename_info['company']} | {filename_info['ticker']} | Period: {filename_info['period']} → {filename_info['report_type']}")
    
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
    
    # Validate data against filename and company type
    is_valid, error_msg = validate_json_data(data, filename_info)
    
    if not is_valid:
        print(f"  ✗ Error: Validation failed - {error_msg}")
        return False
    
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
    print(f"  Company Type: {data['company_type'].upper()} | Report Type: {report_type} | Guidance: {data['guidance_tone'].upper()}")
    
    # Show key metrics based on company type
    company_type = data['company_type']
    
    if company_type in ['reit', 'business_trust']:
        # REIT metrics
        dpu = data.get('dpu')
        dpu_change = data.get('dpu_yoy_change')
        gearing = data.get('gearing_ratio')
        
        print(f"  REIT Metrics:")
        if dpu is not None:
            print(f"    • DPU: {dpu:.2f} cents", end="")
            if dpu_change is not None:
                print(f" (YoY: {dpu_change:+.1f}%)")
            else:
                print()
        
        if gearing is not None:
            print(f"    • Gearing: {gearing:.1f}%")
        
        revenue_change = data.get('revenue_yoy_change')
        if revenue_change is not None:
            print(f"    • Revenue YoY: {revenue_change:+.1f}%")
    
    else:  # Normal company
        revenue_change = data.get('revenue_yoy_change')
        eps_change = data.get('eps_yoy_change')
        net_margin = data.get('net_margin')
        
        print(f"  Company Metrics:")
        if revenue_change is not None:
            print(f"    • Revenue YoY: {revenue_change:+.1f}%")
        if eps_change is not None:
            print(f"    • EPS YoY: {eps_change:+.1f}%")
        if net_margin is not None:
            print(f"    • Net Margin: {net_margin:.1f}%")
    
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
    """Import all .txt files from earnings_reports_pdf folder"""
    print("=" * 60)
    print("CLAUDE EARNINGS REPORT IMPORTER (ADAPTIVE)")
    print("Supports: REITs, Business Trusts, and Normal Companies")
    print("=" * 60)
    print()
    
    # Create processed subfolder
    processed_dir = EARNINGS_PDF_DIR / "processed"
    processed_dir.mkdir(exist_ok=True)
    
    # Look for .txt files
    txt_files = list(EARNINGS_PDF_DIR.glob("*.txt"))
    
    if not txt_files:
        print(f"No .txt files found in: {EARNINGS_PDF_DIR}")
        print("\nPlease add Claude-generated .txt files to this folder.")
        print("\nFilename format: YYYYMMDD_CompanyAbbrev_Period_Ticker.txt")
        print("Example: 20220420_MCT_2HFY2122_N2IU.txt")
        return
    
    print(f"Found {len(txt_files)} .txt file(s) to import\n")
    
    processed = 0
    failed = 0
    moved = 0
    failed_files = []
    
    # Track company types processed
    company_type_counts = {'reit': 0, 'business_trust': 0, 'normal': 0}
    
    for txt_file in txt_files:
        try:
            if import_txt_file(txt_file):
                processed += 1
                
                # Track company type (read the JSON to get company_type)
                try:
                    with open(EARNINGS_REPORTS_DIR / txt_file.name.replace('.txt', '.json'), 'r') as f:
                        data = json.load(f)
                        company_type = data.get('company_type', 'unknown')
                        if company_type in company_type_counts:
                            company_type_counts[company_type] += 1
                except:
                    pass
                
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
    
    # Show company type breakdown
    if processed > 0:
        print(f"\n📊 Company Type Breakdown:")
        if company_type_counts['reit'] > 0:
            print(f"  • REITs: {company_type_counts['reit']}")
        if company_type_counts['business_trust'] > 0:
            print(f"  • Business Trusts: {company_type_counts['business_trust']}")
        if company_type_counts['normal'] > 0:
            print(f"  • Normal Companies: {company_type_counts['normal']}")
    
    if moved > 0:
        print(f"\n✓ Automatically moved {moved} file(s) to: {processed_dir}")
    
    if failed > 0:
        print(f"\n⚠️  {failed} FAILED FILE(S):")
        for filename in failed_files:
            print(f"  • {filename}")
        print("\nCommon issues:")
        print("  1. Filename not in format: YYYYMMDD_CompanyAbbrev_Period_Ticker.txt")
        print("  2. JSON ticker/date doesn't match filename")
        print("  3. Missing company_type field or invalid value")
        print("  4. REIT missing REIT-specific fields (dpu, gearing_ratio, etc.)")
        print("  5. Normal company missing company-specific fields (margins, net_profit)")
        print("  6. File not saved with UTF-8 encoding")
    
    if processed > 0:
        print(f"\n✓ JSON files saved to: {EARNINGS_REPORTS_DIR}")
        print("✓ Using CLAUDE ADAPTIVE ANALYSIS (REITs + Normal Companies)")
        print("\nYou can now view these in the scanner!")


if __name__ == "__main__":
    main()