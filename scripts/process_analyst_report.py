"""
Process analyst report PDFs and extract key information using FinBERT sentiment analysis.
Run this in Codespace after uploading PDFs.
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
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
except ImportError:
    print("Missing dependencies. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "PyPDF2", "transformers", "torch", "--quiet"])
    import PyPDF2
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

from utils.paths import ANALYST_PDF_DIR, ANALYST_REPORTS_DIR


# Initialize FinBERT (will download on first run)
print("Loading FinBERT model...")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
print("✓ FinBERT loaded\n")


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
    
    # Extract ticker (pattern: "SANLI SP" or similar)
    ticker_patterns = [
        r'([A-Z]{2,6})\s+SP\b',  # Singapore stocks
        r'([A-Z]{2,6})\.SI',      # Yahoo format
        r'Bloomberg[:\s]+([A-Z]{2,6})\s+SP',
    ]
    for pattern in ticker_patterns:
        match = re.search(pattern, text[:3000])
        if match:
            ticker_code = match.group(1)
            metadata['ticker'] = f"{ticker_code} SP"
            break
    
    # Extract company name
    name_pattern = r'((?:[A-Z][a-z]+\s+){1,4}(?:Ltd|Limited|Corp|Corporation|Inc|Group|Holdings|Pte|Co)\.?)'
    name_match = re.search(name_pattern, text[:2000])
    if name_match:
        metadata['company'] = name_match.group(1).strip()
    
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


def get_finbert_sentiment(text, max_length=512):
    """Run FinBERT sentiment analysis on text"""
    # Truncate to reasonable length for analysis
    text_sample = text[:3000]
    
    # Tokenize
    inputs = tokenizer(text_sample, return_tensors="pt", truncation=True, 
                      max_length=max_length, padding=True)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # FinBERT outputs: [negative, neutral, positive]
    scores = predictions[0].tolist()
    
    # Convert to single score: -1 (negative) to +1 (positive)
    sentiment_score = scores[2] - scores[0]  # positive - negative
    
    # Determine label
    max_idx = predictions[0].argmax().item()
    labels = ['negative', 'neutral', 'positive']
    sentiment_label = labels[max_idx]
    
    return sentiment_score, sentiment_label


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


def process_report(pdf_path):
    """Main processing function"""
    print(f"Processing: {pdf_path.name}")
    
    # Extract text
    text = extract_text_from_pdf(pdf_path)
    if not text:
        raise ValueError("Could not extract text from PDF")
    
    print(f"  Extracted {len(text)} characters")
    
    # Extract metadata
    metadata = extract_metadata(text)
    print(f"  Ticker: {metadata.get('ticker', 'N/A')}")
    print(f"  Recommendation: {metadata.get('recommendation', 'N/A')}")
    
    # Extract executive summary
    exec_summary = extract_executive_summary(text)
    print(f"  Executive summary: {len(exec_summary)} characters")
    
    # Run sentiment analysis
    print("  Running FinBERT sentiment analysis...")
    sentiment_score, sentiment_label = get_finbert_sentiment(exec_summary)
    print(f"  Sentiment: {sentiment_label} ({sentiment_score:.2f})")
    
    # Extract catalysts and risks
    catalysts, risks = extract_catalysts_and_risks(text)
    print(f"  Found {len(catalysts)} catalysts, {len(risks)} risks")
    
    # Calculate upside if we have both prices
    upside_pct = None
    if 'price_target' in metadata and 'price_at_report' in metadata:
        upside_pct = ((metadata['price_target'] / metadata['price_at_report']) - 1) * 100
    
    # Build output JSON
    output = {
        **metadata,
        "upload_date": datetime.now().isoformat(),
        "sentiment_score": round(sentiment_score, 2),
        "sentiment_label": sentiment_label,
        "executive_summary": exec_summary[:1000],  # Truncate for storage
        "key_catalysts": catalysts,
        "key_risks": risks,
        "upside_pct": round(upside_pct, 1) if upside_pct else None,
        "report_age_days": 0,
        "pdf_filename": pdf_path.name
    }
    
    # Generate output filename
    ticker_clean = metadata.get('ticker', 'UNKNOWN').replace(' ', '_').replace('.', '_')
    report_date = metadata.get('report_date', datetime.now().strftime('%Y-%m-%d'))
    output_filename = f"{ticker_clean}_{report_date}.json"
    
    return output, output_filename


def main():
    """Process all PDFs in analyst_reports_pdf folder"""
    print("=" * 60)
    print("ANALYST REPORT PROCESSOR")
    print("=" * 60)
    print()
    
    # Check for PDFs
    pdf_files = list(ANALYST_PDF_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in: {ANALYST_PDF_DIR}")
        print("\nPlease upload analyst report PDFs to this folder.")
        return
    
    print(f"Found {len(pdf_files)} PDF(s) to process\n")
    
    processed = 0
    failed = 0
    
    for pdf_file in pdf_files:
        try:
            output_data, output_filename = process_report(pdf_file)
            
            # Save JSON
            output_path = ANALYST_REPORTS_DIR / output_filename
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"  ✓ Saved: {output_filename}\n")
            processed += 1
            
        except Exception as e:
            print(f"  ✗ Error processing {pdf_file.name}: {e}\n")
            failed += 1
    
    print("=" * 60)
    print(f"SUMMARY: {processed} processed, {failed} failed")
    print("=" * 60)
    
    if processed > 0:
        print(f"\nJSON files saved to: {ANALYST_REPORTS_DIR}")
        print("You can now view these in the scanner!")


if __name__ == "__main__":
    main()