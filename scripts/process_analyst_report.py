# File: scripts/process_analyst_report.py
"""
Process analyst report PDFs and extract key information using RULE-BASED sentiment analysis.
Run this in Codespace after uploading PDFs.

FILENAME CONVENTION: {DATE}_{ANALYST}_{STOCK_CODE}.pdf
Example: 20251001_MapleCom_N2IU.pdf → Extracts "N2IU"
"""
import json
import re
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    import PyPDF2
except ImportError:
    print("Missing PyPDF2. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyPDF2", "--quiet"])
    import PyPDF2

from utils.paths import ANALYST_PDF_DIR, ANALYST_REPORTS_DIR


# ============================================================================
# RULE-BASED SENTIMENT ANALYSIS SYSTEM
# ============================================================================

# Positive indicators with weights (1-5, higher = stronger signal)
POSITIVE_INDICATORS = {
    # Direct recommendations (strongest signals)
    'upgrade to add': 5,
    'upgrade to buy': 5,
    'upgrade to outperform': 5,
    'initiate coverage with add': 5,
    'initiate coverage with buy': 5,
    'raise to add': 4,
    'raise to buy': 4,
    'raise to outperform': 4,
    'reiterate add': 3,
    'reiterate buy': 3,
    
    # Positive language (moderate signals)
    'we are positive': 3,
    'we are optimistic': 3,
    'we are bullish': 3,
    'we are constructive': 3,
    'positive outlook': 3,
    'bullish outlook': 3,
    'optimistic outlook': 3,
    
    # Action phrases (moderate to strong)
    'time to buy': 4,
    'look forward': 2,
    'attractive entry point': 3,
    'buying opportunity': 4,
    'accumulate': 3,
    'strong buy': 5,
    
    # Upside language (moderate signals)
    'strong upside': 3,
    'significant upside': 3,
    'substantial upside': 3,
    'attractive upside': 3,
    'meaningful upside': 3,
    'considerable upside': 3,
    
    # Business outlook (mild to moderate signals)
    'improving prospects': 2,
    'improving outlook': 2,
    'recovery ahead': 2,
    'turnaround story': 2,
    'strong fundamentals': 2,
    'robust growth': 2,
    'solid growth': 2,
    'accelerating growth': 3,
    'growth momentum': 2,
    'positive momentum': 2,
    'strong momentum': 2,
    
    # Valuation (moderate signals)
    'undervalued': 2,
    'attractive valuation': 2,
    'compelling valuation': 3,
    'trading below fair value': 2,
    'discount to fair value': 2,
    'undemanding valuation': 2,
    
    # Earnings/Performance (mild signals)
    'beat expectations': 2,
    'exceeded expectations': 2,
    'strong results': 2,
    'impressive results': 2,
    'solid results': 2,
    'robust results': 2,
    
    # Risk/Reward (moderate signals)
    'favorable risk-reward': 3,
    'attractive risk-reward': 3,
    'asymmetric upside': 3,
}

# Negative indicators with weights (-1 to -5, more negative = stronger signal)
NEGATIVE_INDICATORS = {
    # Direct recommendations (strongest signals)
    'downgrade to reduce': -5,
    'downgrade to sell': -5,
    'downgrade to underperform': -5,
    'initiate coverage with reduce': -5,
    'initiate coverage with sell': -5,
    'cut to reduce': -4,
    'cut to sell': -4,
    'cut to underperform': -4,
    'reiterate reduce': -3,
    'reiterate sell': -3,
    
    # Negative language (moderate signals)
    'we are negative': -3,
    'we are cautious': -3,
    'we are bearish': -3,
    'we are concerned': -3,
    'negative outlook': -3,
    'bearish outlook': -3,
    'cautious outlook': -3,
    
    # Action phrases (moderate to strong)
    'time to sell': -4,
    'exit position': -4,
    'reduce exposure': -3,
    'avoid': -3,
    'stay away': -4,
    
    # Downside language (moderate signals)
    'limited upside': -2,
    'downside risk': -2,
    'significant downside': -3,
    'substantial downside': -3,
    'elevated risk': -2,
    'heightened risk': -2,
    
    # Business outlook (mild to moderate signals)
    'deteriorating prospects': -2,
    'deteriorating outlook': -2,
    'challenges ahead': -2,
    'headwinds persist': -2,
    'weak fundamentals': -2,
    'slowing growth': -2,
    'decelerating growth': -2,
    'growth concerns': -2,
    'negative momentum': -2,
    
    # Valuation (moderate signals)
    'overvalued': -2,
    'expensive valuation': -2,
    'rich valuation': -2,
    'trading above fair value': -2,
    'premium to fair value': -2,
    'stretched valuation': -2,
    'demanding valuation': -2,
    
    # Earnings/Performance (mild signals)
    'missed expectations': -2,
    'disappointing results': -2,
    'weak results': -2,
    'poor results': -2,
    'below expectations': -2,
    
    # Risk/Reward (moderate signals)
    'unfavorable risk-reward': -3,
    'poor risk-reward': -3,
    'asymmetric downside': -3,
}

# Neutral indicators (weight = 0)
NEUTRAL_INDICATORS = {
    'maintain hold': 0,
    'maintain neutral': 0,
    'reiterate hold': 0,
    'reiterate neutral': 0,
    'no change': 0,
    'unchanged': 0,
    'mixed signals': 0,
    'balanced view': 0,
    'neutral stance': 0,
}


def calculate_rule_based_sentiment(text, debug=False):
    """
    Calculate sentiment using rule-based phrase matching
    
    Args:
        text: Text to analyze (executive summary)
        debug: If True, print detailed matching information
        
    Returns:
        tuple: (sentiment_score, sentiment_label, matched_phrases)
        - sentiment_score: -1.0 to +1.0
        - sentiment_label: "positive", "neutral", or "negative"
        - matched_phrases: List of matched phrases with weights
    """
    text_lower = text.lower()
    score = 0
    matched_phrases = []
    
    # Check positive indicators
    for phrase, weight in POSITIVE_INDICATORS.items():
        if phrase in text_lower:
            matched_phrases.append({
                'phrase': phrase,
                'weight': weight,
                'type': 'positive'
            })
            score += weight
    
    # Check negative indicators
    for phrase, weight in NEGATIVE_INDICATORS.items():
        if phrase in text_lower:
            matched_phrases.append({
                'phrase': phrase,
                'weight': weight,
                'type': 'negative'
            })
            score += weight  # Weight is already negative
    
    # Check neutral indicators (informational only)
    for phrase, weight in NEUTRAL_INDICATORS.items():
        if phrase in text_lower:
            matched_phrases.append({
                'phrase': phrase,
                'weight': weight,
                'type': 'neutral'
            })
    
    # Normalize score to -1.0 to +1.0 range
    # Dividing by 10 means a score of 10+ = 1.0, -10 or less = -1.0
    sentiment_score = score / 10.0
    sentiment_score = max(-1.0, min(1.0, sentiment_score))
    
    # Assign label based on thresholds
    if sentiment_score > 0.3:
        sentiment_label = "positive"
    elif sentiment_score < -0.3:
        sentiment_label = "negative"
    else:
        sentiment_label = "neutral"
    
    # Debug output
    if debug:
        print("\n" + "="*60)
        print("RULE-BASED SENTIMENT ANALYSIS")
        print("="*60)
        print(f"\nTEXT ANALYZED ({len(text)} chars):")
        print(text[:500] + "..." if len(text) > 500 else text)
        print(f"\nMATCHED PHRASES ({len(matched_phrases)}):")
        
        # Group by type
        positives = [m for m in matched_phrases if m['type'] == 'positive']
        negatives = [m for m in matched_phrases if m['type'] == 'negative']
        neutrals = [m for m in matched_phrases if m['type'] == 'neutral']
        
        if positives:
            print("\n✅ POSITIVE:")
            for match in sorted(positives, key=lambda x: x['weight'], reverse=True):
                print(f"   • '{match['phrase']}' → +{match['weight']}")
        
        if negatives:
            print("\n❌ NEGATIVE:")
            for match in sorted(negatives, key=lambda x: x['weight']):
                print(f"   • '{match['phrase']}' → {match['weight']}")
        
        if neutrals:
            print("\n➖ NEUTRAL:")
            for match in neutrals:
                print(f"   • '{match['phrase']}' → {match['weight']}")
        
        print(f"\nRAW SCORE: {score}")
        print(f"NORMALIZED SCORE: {sentiment_score:.2f}")
        print(f"LABEL: {sentiment_label.upper()}")
        print("="*60 + "\n")
    
    return sentiment_score, sentiment_label, matched_phrases


# ============================================================================
# PDF PROCESSING FUNCTIONS (unchanged)
# ============================================================================

def extract_ticker_from_filename(filename):
    """
    Extract stock ticker from filename
    
    Expected format: {DATE}_{ANALYST}_{TICKER}.pdf
    Examples:
        20251001_MapleCom_N2IU.pdf -> N2IU
        20251003_CGS_A17U.pdf -> A17U
        20251006_N2IU.pdf -> N2IU (works without analyst)
        N2IU.pdf -> N2IU (fallback: just ticker.pdf)
    
    Returns:
        tuple: (base_ticker, sgx_ticker) e.g., ("N2IU", "N2IU.SG")
        None if filename doesn't match any pattern
    """
    # Pattern 1: Extract last segment before .pdf (after underscore)
    # Matches: 20251001_MapleCom_N2IU.pdf -> N2IU
    pattern_with_underscore = r'_([A-Z0-9]+)\.pdf$'
    
    match = re.search(pattern_with_underscore, filename)
    if match:
        base_ticker = match.group(1)
        sgx_ticker = f"{base_ticker}.SG"
        return base_ticker, sgx_ticker
    
    # Pattern 2: Fallback for simple format (just TICKER.pdf)
    # Matches: N2IU.pdf -> N2IU
    pattern_simple = r'^([A-Z0-9]+)\.pdf$'
    
    match = re.match(pattern_simple, filename)
    if match:
        base_ticker = match.group(1)
        sgx_ticker = f"{base_ticker}.SG"
        return base_ticker, sgx_ticker
    
    # No match found
    return None


def validate_ticker_in_watchlist(sgx_ticker):
    """
    Validate if ticker exists in scanner's watchlist
    
    Returns:
        bool: True if ticker found in watchlist
    """
    try:
        from utils.watchlist import get_active_watchlist
        watchlist = get_active_watchlist()
        return sgx_ticker in watchlist
    except Exception as e:
        print(f"  ⚠️  Could not validate ticker against watchlist: {e}")
        return False


def extract_text_from_pdf(pdf_path):
    """Extract all text from PDF"""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""


def extract_metadata(text):
    """Extract key metadata using regex patterns"""
    metadata = {}
    
    # Extract recommendation (ADD, HOLD, REDUCE, BUY, SELL)
    rec_patterns = [
        r'\b(ADD|HOLD|REDUCE|BUY|SELL|OUTPERFORM|UNDERPERFORM|NEUTRAL)\b',
        r'[Rr]ating[:\s]+(ADD|HOLD|REDUCE|BUY|SELL)',
        r'[Rr]ecommendation[:\s]+(ADD|HOLD|REDUCE|BUY|SELL)',
    ]
    for pattern in rec_patterns:
        match = re.search(pattern, text[:3000])
        if match:
            rec = match.group(1).upper()
            # Normalize recommendations
            if rec in ['OUTPERFORM', 'BUY']:
                rec = 'ADD'
            elif rec in ['UNDERPERFORM', 'SELL']:
                rec = 'REDUCE'
            elif rec == 'NEUTRAL':
                rec = 'HOLD'
            metadata['recommendation'] = rec
            break
    
    # Extract price target
    target_patterns = [
        r'[Tt]arget\s+[Pp]rice[:\s]+S?\$\s*([\d.]+)',
        r'[Tt]\.?[Pp]\.?[:\s]+S?\$\s*([\d.]+)',
        r'[Pp]rice\s+[Tt]arget[:\s]+S?\$\s*([\d.]+)',
    ]
    for pattern in target_patterns:
        match = re.search(pattern, text[:3000])
        if match:
            metadata['price_target'] = float(match.group(1))
            break
    
    # Extract current price
    price_patterns = [
        r'[Cc]urrent\s+[Pp]rice[:\s]+S?\$\s*([\d.]+)',
        r'[Pp]rice[:\s]+S?\$\s*([\d.]+)',
    ]
    for pattern in price_patterns:
        match = re.search(pattern, text[:3000])
        if match:
            metadata['price_at_report'] = float(match.group(1))
            break
    
    # Extract analyst firm
    firm_patterns = [
        r'(CGS International|DBS|UOB|OCBC|Maybank|RHB|Phillip Securities|CIMB)',
        r'([A-Z][a-z]+\s+(?:Securities|Research|Bank))',
    ]
    for pattern in firm_patterns:
        match = re.search(pattern, text[:1000])
        if match:
            metadata['analyst_firm'] = match.group(1)
            break
    
    # Extract date (multiple formats)
    date_patterns = [
        r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}',
        r'\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}',
        r'\d{4}-\d{2}-\d{2}',
    ]
    for pattern in date_patterns:
        match = re.search(pattern, text[:1000])
        if match:
            date_str = match.group(0)
            try:
                # Try multiple date formats
                for fmt in ['%B %d, %Y', '%d %B %Y', '%Y-%m-%d', '%B %d %Y']:
                    try:
                        parsed_date = datetime.strptime(date_str, fmt)
                        metadata['report_date'] = parsed_date.strftime('%Y-%m-%d')
                        break
                    except ValueError:
                        continue
            except:
                pass
            if 'report_date' in metadata:
                break
    
    # If no date found, use today
    if 'report_date' not in metadata:
        metadata['report_date'] = datetime.now().strftime('%Y-%m-%d')
    
    return metadata


def extract_section(text, headers):
    """Extract text section based on headers"""
    for header in headers:
        pattern = rf'{header}\s*(.{{500,3000}}?)(?:\n\n|\n[A-Z]{{2,}})'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # Fallback: return early text
    return text[500:2500]


def extract_executive_summary(text):
    """Extract executive summary or investment thesis"""
    headers = [
        'Investment [Tt]hesis',
        'Executive [Ss]ummary',
        'Initiate coverage',
        'Key highlights',
        'Investment [Cc]ase',
    ]
    return extract_section(text, headers)


def extract_catalysts_and_risks(text):
    """Extract key catalysts and risks"""
    catalysts = []
    risks = []
    
    # Extract catalysts
    catalyst_patterns = [
        r'Key catalysts.*?:(.*?)(?:Key risks|Downside risks|Risk|Valuation)',
        r'Catalysts.*?:(.*?)(?:Risk|Valuation)',
        r'Positive factors.*?:(.*?)(?:Risk|Negative)',
    ]
    
    for pattern in catalyst_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            section = match.group(1)
            # Extract bullet points
            bullets = re.findall(r'[•\-\d]\)?\s*(.+?)(?:\n|$)', section)
            catalysts = [b.strip() for b in bullets if 20 < len(b) < 200][:5]
            break
    
    # Extract risks
    risk_patterns = [
        r'(?:Key risks|Downside risks|Risk factors).*?:(.*?)(?:\n\n|Valuation|Financials|Key)',
        r'Risks.*?:(.*?)(?:\n\n|Valuation)',
        r'Negative factors.*?:(.*?)(?:\n\n)',
    ]
    
    for pattern in risk_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            section = match.group(1)
            bullets = re.findall(r'[•\-\d]\)?\s*(.+?)(?:\n|$)', section)
            risks = [b.strip() for b in bullets if 20 < len(b) < 200][:5]
            break
    
    return catalysts, risks


def process_report(pdf_path, debug=False):
    """Main processing function"""
    print(f"Processing: {pdf_path.name}")
    
    # Extract ticker from filename
    ticker_result = extract_ticker_from_filename(pdf_path.name)
    
    if not ticker_result:
        raise ValueError(
            f"Could not extract stock code from filename: {pdf_path.name}\n"
            f"Expected format: {{DATE}}_{{ANALYST}}_{{STOCK_CODE}}.pdf\n"
            f"Examples:\n"
            f"  ✓ 20251001_MapleCom_N2IU.pdf\n"
            f"  ✓ 20251003_CGS_A17U.pdf\n"
            f"  ✓ 20251006_N2IU.pdf\n"
            f"  ✓ N2IU.pdf\n"
            f"  ✗ 20251001_MapleCom.pdf (missing stock code)\n"
            f"  ✗ analyst_report.pdf (missing stock code)"
        )
    
    base_ticker, sgx_ticker = ticker_result
    print(f"  Extracted ticker: {base_ticker} → {sgx_ticker}")
    
    # Validate against watchlist
    if validate_ticker_in_watchlist(sgx_ticker):
        print(f"  ✓ Ticker found in scanner watchlist")
    else:
        print(f"  ⚠️  WARNING: {sgx_ticker} not found in scanner watchlist")
        print(f"     This report may not match any stocks in your scanner")
    
    # Extract text
    text = extract_text_from_pdf(pdf_path)
    if not text:
        raise ValueError("Could not extract text from PDF")
    
    print(f"  Extracted {len(text)} characters")
    
    # Extract metadata
    metadata = extract_metadata(text)
    print(f"  Recommendation: {metadata.get('recommendation', 'N/A')}")
    
    # Extract executive summary
    exec_summary = extract_executive_summary(text)
    print(f"  Executive summary: {len(exec_summary)} characters")
    
    # Run RULE-BASED sentiment analysis
    print("  Running rule-based sentiment analysis...")
    sentiment_score, sentiment_label, matched_phrases = calculate_rule_based_sentiment(
        exec_summary, debug=debug
    )
    print(f"  Sentiment: {sentiment_label.upper()} ({sentiment_score:+.2f})")
    print(f"  Matched {len(matched_phrases)} phrases")
    
    # Extract catalysts and risks
    catalysts, risks = extract_catalysts_and_risks(text)
    print(f"  Found {len(catalysts)} catalysts, {len(risks)} risks")
    
    # Calculate upside if we have both prices
    upside_pct = None
    if 'price_target' in metadata and 'price_at_report' in metadata:
        upside_pct = ((metadata['price_target'] / metadata['price_at_report']) - 1) * 100
    
    # Build output JSON
    output = {
        # Ticker information (both formats)
        "ticker": base_ticker,          # Display ticker (e.g., "N2IU")
        "ticker_sgx": sgx_ticker,       # SGX ticker for scanner matching (e.g., "N2IU.SG")
        
        # Metadata from PDF
        **metadata,
        
        # Processing metadata
        "upload_date": datetime.now().isoformat(),
        "sentiment_score": round(sentiment_score, 2),
        "sentiment_label": sentiment_label,
        "sentiment_method": "rule_based",  # NEW: Track which method was used
        "matched_phrases": matched_phrases,  # NEW: Store which phrases matched
        "executive_summary": exec_summary[:1000],  # Truncate for storage
        "key_catalysts": catalysts,
        "key_risks": risks,
        "upside_pct": round(upside_pct, 1) if upside_pct else None,
        "report_age_days": 0,
        "pdf_filename": pdf_path.name
    }
    
    # Generate output filename using base ticker
    report_date = metadata.get('report_date', datetime.now().strftime('%Y-%m-%d'))
    output_filename = f"{base_ticker}_{report_date}.json"
    
    return output, output_filename


def main():
    """Process all PDFs in analyst_reports_pdf folder"""
    print("=" * 60)
    print("ANALYST REPORT PROCESSOR (RULE-BASED SENTIMENT)")
    print("=" * 60)
    print()
    
    # Check for PDFs
    pdf_files = list(ANALYST_PDF_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in: {ANALYST_PDF_DIR}")
        print("\nPlease upload analyst report PDFs to this folder.")
        print("\nFILENAME CONVENTION: {DATE}_{ANALYST}_{STOCK_CODE}.pdf")
        print("Examples:")
        print("  ✓ 20251001_MapleCom_N2IU.pdf")
        print("  ✓ 20251003_CGS_A17U.pdf")
        print("  ✓ 20251006_N2IU.pdf (analyst name optional)")
        print("  ✓ N2IU.pdf (minimal format)")
        print("  ✗ 20251001_MapleCom.pdf (missing stock code)")
        print("  ✗ analyst_report.pdf (missing stock code)")
        return
    
    print(f"Found {len(pdf_files)} PDF(s) to process\n")
    
    # Ask if user wants debug output
    print("Enable debug output? (shows matched phrases for each report)")
    debug_choice = input("Enter 'y' for yes, any other key for no: ").strip().lower()
    debug = (debug_choice == 'y')
    print()
    
    processed = 0
    failed = 0
    failed_details = []
    
    for pdf_file in pdf_files:
        try:
            output_data, output_filename = process_report(pdf_file, debug=debug)
            
            # Save JSON
            output_path = ANALYST_REPORTS_DIR / output_filename
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"  ✓ Saved: {output_filename}\n")
            processed += 1
            
        except Exception as e:
            error_msg = str(e).split('\n')[0]  # Get first line of error
            print(f"  ✗ Error: {error_msg}\n")
            failed += 1
            failed_details.append({
                'filename': pdf_file.name,
                'error': error_msg
            })
    
    print("=" * 60)
    print(f"SUMMARY: {processed} processed, {failed} failed")
    print("=" * 60)
    
    if failed > 0:
        print("\n⚠️  FAILED FILES:")
        for detail in failed_details:
            print(f"  • {detail['filename']}")
            print(f"    Reason: {detail['error']}")
        print("\nPlease rename these files to include stock code at the end:")
        print("  Format: {DATE}_{ANALYST}_{STOCK_CODE}.pdf")
        print("  Example: 20251001_MapleCom_N2IU.pdf")
    
    if processed > 0:
        print(f"\n✓ JSON files saved to: {ANALYST_REPORTS_DIR}")
        print("✓ Using RULE-BASED sentiment analysis (no ML required)")
        print("You can now view these in the scanner!")


if __name__ == "__main__":
    main()