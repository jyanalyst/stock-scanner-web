# File: scripts/bullet_focused_processor.py
"""
BULLET-FOCUSED Analyst Report Processor
Extracts ONLY the opening bullet points which contain analyst opinions
"""
import json
import re
from datetime import datetime
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

try:
    import PyPDF2
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "PyPDF2", "transformers", "torch", "--quiet"])
    import PyPDF2
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

from utils.paths import ANALYST_PDF_DIR, ANALYST_REPORTS_DIR

print("Loading FinBERT-Tone model...")
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
print("✓ Model loaded\n")


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


def extract_ticker_from_filename(filename):
    """Extract stock ticker from filename"""
    pattern_with_underscore = r'_([A-Z0-9]+)\.pdf$'
    match = re.search(pattern_with_underscore, filename)
    if match:
        base_ticker = match.group(1)
        return base_ticker, f"{base_ticker}.SG"
    
    pattern_simple = r'^([A-Z0-9]+)\.pdf$'
    match = re.match(pattern_simple, filename)
    if match:
        base_ticker = match.group(1)
        return base_ticker, f"{base_ticker}.SG"
    
    return None


def extract_opening_bullets(text):
    """
    Extract ONLY the opening bullet points from first page
    These contain the actual analyst opinion in CGS reports
    """
    # Find the first bullet point
    first_bullet = text.find('■')
    
    if first_bullet == -1:
        return None
    
    # Extract from first bullet to the next major section
    # Look for section breaks (usually double newline or new heading)
    section_end_markers = [
        '\n\n\n',  # Multiple blank lines
        'Figure 1:',
        'Table 1:',
        'Financial Summary',
        'SOURCES:',
        'BY THE NUMBERS'
    ]
    
    # Start from first bullet
    start = first_bullet
    
    # Find the earliest section end marker
    end = len(text)
    for marker in section_end_markers:
        marker_pos = text.find(marker, start)
        if marker_pos > start and marker_pos < end:
            end = marker_pos
    
    # Extract the bullet section
    bullet_section = text[start:end].strip()
    
    # If too long, limit to first 1500 chars (should be plenty for bullets)
    if len(bullet_section) > 1500:
        bullet_section = bullet_section[:1500]
    
    return bullet_section


def get_finbert_sentiment(text):
    """Run FinBERT-Tone sentiment analysis"""
    if not text or len(text) < 20:
        return None
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                      max_length=512, padding=True)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    scores = predictions[0].tolist()
    sentiment_score = scores[2] - scores[0]
    
    max_idx = predictions[0].argmax().item()
    labels = ['negative', 'neutral', 'positive']
    sentiment_label = labels[max_idx]
    
    return {
        'score': sentiment_score,
        'label': sentiment_label,
        'negative': scores[0],
        'neutral': scores[1],
        'positive': scores[2]
    }


def extract_metadata(text):
    """Extract metadata"""
    metadata = {}
    
    # Recommendation
    rec_patterns = [
        r'\b(ADD|HOLD|REDUCE|BUY|SELL|OUTPERFORM|UNDERPERFORM|NEUTRAL)\b',
    ]
    for pattern in rec_patterns:
        match = re.search(pattern, text[:3000])
        if match:
            rec = match.group(1).upper()
            if rec in ['OUTPERFORM', 'BUY']:
                rec = 'ADD'
            elif rec in ['UNDERPERFORM', 'SELL']:
                rec = 'REDUCE'
            elif rec == 'NEUTRAL':
                rec = 'HOLD'
            metadata['recommendation'] = rec
            break
    
    # Date
    date_patterns = [
        r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}',
    ]
    for pattern in date_patterns:
        match = re.search(pattern, text[:1000])
        if match:
            date_str = match.group(0)
            try:
                for fmt in ['%B %d, %Y', '%B %d %Y']:
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
    
    if 'report_date' not in metadata:
        metadata['report_date'] = datetime.now().strftime('%Y-%m-%d')
    
    return metadata


def process_report(pdf_path):
    """Process report focusing ONLY on bullet points"""
    print(f"\nProcessing: {pdf_path.name}")
    print("-" * 60)
    
    # Extract ticker
    ticker_result = extract_ticker_from_filename(pdf_path.name)
    if not ticker_result:
        raise ValueError(f"Could not extract ticker from: {pdf_path.name}")
    
    base_ticker, sgx_ticker = ticker_result
    print(f"  Ticker: {base_ticker} → {sgx_ticker}")
    
    # Extract text
    text = extract_text_from_pdf(pdf_path)
    if not text:
        raise ValueError("Could not extract text")
    
    print(f"  Total text: {len(text):,} characters")
    
    # Extract metadata
    metadata = extract_metadata(text)
    print(f"  Recommendation: {metadata.get('recommendation', 'Not found')}")
    
    # FOCUS: Extract ONLY opening bullets
    bullet_text = extract_opening_bullets(text)
    
    if not bullet_text:
        print("  ⚠️  No bullet points found - using first 1000 chars")
        bullet_text = text[500:1500]
    
    print(f"  Bullet text: {len(bullet_text)} characters")
    print(f"\n  📝 Analyzing:")
    print("  " + "-" * 58)
    print("  " + bullet_text[:300].replace('\n', '\n  '))
    print("  " + "-" * 58)
    
    # Sentiment analysis on bullets ONLY
    print("\n  Running sentiment analysis...")
    sentiment = get_finbert_sentiment(bullet_text)
    
    if sentiment:
        print(f"  Sentiment: {sentiment['label']} ({sentiment['score']:.3f})")
        print(f"    Positive: {sentiment['positive']:.3f}")
        print(f"    Neutral:  {sentiment['neutral']:.3f}")
        print(f"    Negative: {sentiment['negative']:.3f}")
    else:
        sentiment = {'score': 0.0, 'label': 'neutral', 
                    'negative': 0.0, 'neutral': 1.0, 'positive': 0.0}
    
    # Build output
    output = {
        "ticker": base_ticker,
        "ticker_sgx": sgx_ticker,
        **metadata,
        "upload_date": datetime.now().isoformat(),
        "sentiment_score": round(sentiment['score'], 2),
        "sentiment_label": sentiment['label'],
        "executive_summary": bullet_text[:1000],
        "key_catalysts": [],
        "key_risks": [],
        "upside_pct": None,
        "report_age_days": 0,
        "pdf_filename": pdf_path.name
    }
    
    report_date = metadata.get('report_date', datetime.now().strftime('%Y-%m-%d'))
    output_filename = f"{base_ticker}_{report_date}.json"
    
    return output, output_filename


def main():
    """Process all PDFs focusing on bullets only"""
    print("=" * 60)
    print("BULLET-FOCUSED ANALYST REPORT PROCESSOR")
    print("=" * 60)
    
    pdf_files = list(ANALYST_PDF_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print(f"\nNo PDFs found in: {ANALYST_PDF_DIR}")
        return
    
    print(f"\nFound {len(pdf_files)} PDF(s)\n")
    
    processed = 0
    failed = 0
    
    for pdf_file in pdf_files:
        try:
            output_data, output_filename = process_report(pdf_file)
            
            # Save JSON
            output_path = ANALYST_REPORTS_DIR / output_filename
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"  ✓ Saved: {output_filename}")
            processed += 1
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"SUMMARY: {processed} processed, {failed} failed")
    print("=" * 60)
    
    if processed > 0:
        print(f"\n✓ JSON files saved to: {ANALYST_REPORTS_DIR}")
        print("Refresh scanner to see updated sentiment scores!")


if __name__ == "__main__":
    main()