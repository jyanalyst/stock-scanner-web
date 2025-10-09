# File: scripts/improved_analyst_processor.py
"""
Improved Analyst Report Processor with Better Text Extraction
Specifically designed for CGS International and similar format reports
"""
import json
import re
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import PyPDF2
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
except ImportError:
    print("Installing dependencies...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "PyPDF2", "transformers", "torch", "--quiet"])
    import PyPDF2
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

from utils.paths import ANALYST_PDF_DIR, ANALYST_REPORTS_DIR

# Initialize FinBERT-Tone
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
    # Pattern: Extract last segment before .pdf
    pattern_with_underscore = r'_([A-Z0-9]+)\.pdf$'
    match = re.search(pattern_with_underscore, filename)
    if match:
        base_ticker = match.group(1)
        return base_ticker, f"{base_ticker}.SG"
    
    # Fallback: simple format
    pattern_simple = r'^([A-Z0-9]+)\.pdf$'
    match = re.match(pattern_simple, filename)
    if match:
        base_ticker = match.group(1)
        return base_ticker, f"{base_ticker}.SG"
    
    return None


def find_bullet_point_sections(text):
    """
    Find sections with bullet points (■ or •) - these often contain key opinions
    CGS International reports use ■ bullets at the start for key points
    """
    # Split into lines
    lines = text.split('\n')
    
    bullet_sections = []
    current_section = []
    in_bullet_section = False
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Check if line starts with bullet point
        if line.startswith('■') or line.startswith('•'):
            if not in_bullet_section:
                in_bullet_section = True
                current_section = [line]
            else:
                current_section.append(line)
        elif in_bullet_section:
            # Check if we should continue the section
            if len(line) > 20 and not line.isupper():  # Likely continuation
                current_section.append(line)
            else:
                # End of section
                if current_section:
                    bullet_sections.append('\n'.join(current_section))
                current_section = []
                in_bullet_section = False
    
    # Add last section if exists
    if current_section:
        bullet_sections.append('\n'.join(current_section))
    
    return bullet_sections


def extract_first_page_content(text):
    """
    Extract the first 2000-3000 characters which often contain the summary
    But skip the header/boilerplate (first 500 chars)
    """
    # Find where the actual content starts (after header/boilerplate)
    # Look for company name or first bullet point
    content_start = 0
    
    # Try to find where real content begins
    patterns_to_skip = [
        'IMPORTANT DISCLOSURES',
        'Company Note',
        'LSEG ESG',
    ]
    
    for pattern in patterns_to_skip:
        idx = text.find(pattern)
        if idx > content_start:
            content_start = idx
    
    # Start after the header section (add 200 chars buffer)
    content_start = min(content_start + 200, 800)
    
    # Extract next 2500 characters
    return text[content_start:content_start + 2500]


def extract_opinion_paragraphs(text):
    """
    Find paragraphs containing opinion keywords
    These are likely to have analyst views
    """
    opinion_keywords = [
        'expect', 'believe', 'view', 'outlook', 'recommend',
        'maintain', 'upgrade', 'downgrade', 'reiterate',
        'attractive', 'positive', 'negative', 'bullish', 'bearish',
        'strong', 'weak', 'favorable', 'unfavorable',
        'add', 'hold', 'reduce', 'buy', 'sell'
    ]
    
    # Split into paragraphs (double newline)
    paragraphs = text.split('\n\n')
    
    opinion_paragraphs = []
    for para in paragraphs:
        if len(para) < 100:  # Skip very short paragraphs
            continue
        
        # Count opinion keywords
        keyword_count = sum(1 for keyword in opinion_keywords 
                          if keyword in para.lower())
        
        if keyword_count >= 2:  # At least 2 opinion keywords
            opinion_paragraphs.append(para)
    
    return opinion_paragraphs


def extract_recommendation_section(text):
    """
    Extract text around recommendation keywords (ADD, HOLD, REDUCE, etc.)
    """
    recommendation_keywords = ['ADD', 'HOLD', 'REDUCE', 'BUY', 'SELL', 
                              'OUTPERFORM', 'UNDERPERFORM', 'NEUTRAL']
    
    sections = []
    
    for keyword in recommendation_keywords:
        # Find the keyword
        pattern = rf'\b{keyword}\b'
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        
        for match in matches:
            start = match.start()
            # Extract 500 chars before and after
            section_start = max(0, start - 500)
            section_end = min(len(text), start + 1000)
            section = text[section_start:section_end]
            
            if len(section) > 200:  # Meaningful section
                sections.append(section)
    
    return sections


def get_finbert_sentiment(text, max_length=512):
    """Run FinBERT-Tone sentiment analysis"""
    if not text or len(text) < 50:
        return None
    
    # Truncate
    text_sample = text[:3000]
    
    # Tokenize
    inputs = tokenizer(text_sample, return_tensors="pt", truncation=True, 
                      max_length=max_length, padding=True)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # FinBERT-Tone: [negative, neutral, positive]
    scores = predictions[0].tolist()
    
    # Convert to single score
    sentiment_score = scores[2] - scores[0]
    
    # Determine label
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


def extract_comprehensive_opinion(text):
    """
    IMPROVED: Extract analyst opinion using multiple methods and combine
    Returns the best representation of analyst opinion
    """
    print("  Extracting opinion using multiple methods...")
    
    # Method 1: Bullet point sections (CGS format)
    bullet_sections = find_bullet_point_sections(text)
    print(f"  Found {len(bullet_sections)} bullet point sections")
    
    # Method 2: First page content (after header)
    first_page = extract_first_page_content(text)
    print(f"  Extracted first page content: {len(first_page)} chars")
    
    # Method 3: Opinion paragraphs
    opinion_paras = extract_opinion_paragraphs(text)
    print(f"  Found {len(opinion_paras)} opinion paragraphs")
    
    # Method 4: Recommendation sections
    rec_sections = extract_recommendation_section(text)
    print(f"  Found {len(rec_sections)} recommendation sections")
    
    # Combine all methods - prioritize bullet points, then opinions, then first page
    combined_text = ""
    
    # Priority 1: Bullet sections (often the executive summary)
    if bullet_sections:
        combined_text += '\n\n'.join(bullet_sections[:3])  # Top 3 bullet sections
        print(f"  Added bullet sections: {len(combined_text)} chars")
    
    # Priority 2: Opinion paragraphs
    if opinion_paras and len(combined_text) < 1500:
        combined_text += '\n\n' + '\n\n'.join(opinion_paras[:2])
        print(f"  Added opinion paragraphs: {len(combined_text)} chars")
    
    # Priority 3: Recommendation sections
    if rec_sections and len(combined_text) < 1500:
        combined_text += '\n\n' + rec_sections[0]
        print(f"  Added recommendation section: {len(combined_text)} chars")
    
    # Fallback: Use first page if nothing else worked
    if len(combined_text) < 500:
        combined_text = first_page
        print(f"  Used first page fallback: {len(combined_text)} chars")
    
    # Limit to reasonable size
    if len(combined_text) > 3000:
        combined_text = combined_text[:3000]
    
    return combined_text


def extract_metadata(text):
    """Extract recommendation, prices, firm, date"""
    metadata = {}
    
    # Recommendation
    rec_patterns = [
        r'\b(ADD|HOLD|REDUCE|BUY|SELL|OUTPERFORM|UNDERPERFORM|NEUTRAL)\b',
        r'[Rr]ating[:\s]+(ADD|HOLD|REDUCE|BUY|SELL)',
        r'[Rr]ecommendation[:\s]+(ADD|HOLD|REDUCE|BUY|SELL)',
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
    
    # Price target
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
    
    # Current price
    price_patterns = [
        r'[Cc]urrent\s+[Pp]rice[:\s]+S?\$\s*([\d.]+)',
        r'[Pp]rice[:\s]+S?\$\s*([\d.]+)',
    ]
    for pattern in price_patterns:
        match = re.search(pattern, text[:3000])
        if match:
            metadata['price_at_report'] = float(match.group(1))
            break
    
    # Analyst firm
    firm_patterns = [
        r'(CGS International|CGS-CIMB|DBS|UOB|OCBC|Maybank|RHB|Phillip Securities|CIMB)',
    ]
    for pattern in firm_patterns:
        match = re.search(pattern, text[:1000])
        if match:
            metadata['analyst_firm'] = match.group(1)
            break
    
    # Date
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
    
    if 'report_date' not in metadata:
        metadata['report_date'] = datetime.now().strftime('%Y-%m-%d')
    
    return metadata


def extract_catalysts_and_risks(text):
    """Extract catalysts and risks"""
    catalysts = []
    risks = []
    
    # Catalysts
    catalyst_patterns = [
        r'Key catalysts.*?:(.*?)(?:Key risks|Downside risks|Risk|Valuation)',
        r'Catalysts.*?:(.*?)(?:Risk|Valuation)',
        r'Positive factors.*?:(.*?)(?:Risk|Negative)',
        r'Key positives.*?:(.*?)(?:Key risks|Risk)',
    ]
    
    for pattern in catalyst_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            section = match.group(1)
            bullets = re.findall(r'[•■\-\d]\)?\s*(.+?)(?:\n|$)', section)
            catalysts = [b.strip() for b in bullets if 20 < len(b) < 200][:5]
            break
    
    # Risks
    risk_patterns = [
        r'(?:Key risks|Downside risks|Risk factors).*?:(.*?)(?:\n\n|Valuation|Financials|Key)',
        r'Risks.*?:(.*?)(?:\n\n|Valuation)',
        r'Negative factors.*?:(.*?)(?:\n\n)',
    ]
    
    for pattern in risk_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            section = match.group(1)
            bullets = re.findall(r'[•■\-\d]\)?\s*(.+?)(?:\n|$)', section)
            risks = [b.strip() for b in bullets if 20 < len(b) < 200][:5]
            break
    
    return catalysts, risks


def process_report(pdf_path):
    """Process a single report with improved extraction"""
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
    
    # IMPROVED: Extract comprehensive opinion
    opinion_text = extract_comprehensive_opinion(text)
    print(f"  Opinion text: {len(opinion_text)} characters")
    
    # Sentiment analysis
    print("  Running sentiment analysis...")
    sentiment = get_finbert_sentiment(opinion_text)
    
    if sentiment:
        print(f"  Sentiment: {sentiment['label']} ({sentiment['score']:.3f})")
        print(f"    Positive: {sentiment['positive']:.3f}")
        print(f"    Neutral:  {sentiment['neutral']:.3f}")
        print(f"    Negative: {sentiment['negative']:.3f}")
    else:
        print("  ⚠️  Could not analyze sentiment")
        sentiment = {'score': 0.0, 'label': 'neutral', 
                    'negative': 0.0, 'neutral': 1.0, 'positive': 0.0}
    
    # Extract catalysts and risks
    catalysts, risks = extract_catalysts_and_risks(text)
    print(f"  Catalysts: {len(catalysts)}, Risks: {len(risks)}")
    
    # Calculate upside
    upside_pct = None
    if 'price_target' in metadata and 'price_at_report' in metadata:
        upside_pct = ((metadata['price_target'] / metadata['price_at_report']) - 1) * 100
    
    # Build output
    output = {
        "ticker": base_ticker,
        "ticker_sgx": sgx_ticker,
        **metadata,
        "upload_date": datetime.now().isoformat(),
        "sentiment_score": round(sentiment['score'], 2),
        "sentiment_label": sentiment['label'],
        "executive_summary": opinion_text[:1000],
        "key_catalysts": catalysts,
        "key_risks": risks,
        "upside_pct": round(upside_pct, 1) if upside_pct else None,
        "report_age_days": 0,
        "pdf_filename": pdf_path.name
    }
    
    # Output filename
    report_date = metadata.get('report_date', datetime.now().strftime('%Y-%m-%d'))
    output_filename = f"{base_ticker}_{report_date}.json"
    
    return output, output_filename


def main():
    """Process all PDFs with improved extraction"""
    print("=" * 60)
    print("IMPROVED ANALYST REPORT PROCESSOR")
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