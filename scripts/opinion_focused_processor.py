# File: scripts/opinion_focused_processor.py
"""
Opinion-Focused Analyst Report Processor
FINAL FIX: Extracts actual analyst opinion paragraphs, not tables/bullets
Searches for "We upgrade/believe/expect" phrases to find real opinions
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


def extract_title(text):
    """Extract report title (analyst's headline)"""
    # Look for title patterns - usually short line before first bullet
    title_patterns = [
        # After company name, before bullets
        r'(?:Ltd|Limited|Corp|Corporation|Holdings|Group|International|REIT|Marine|Oil)\s+(.{15,100}?)\s*■',
        # Standalone line before bullets
        r'\n(.{20,100})\s*\n+\s*■',
    ]
    
    for pattern in title_patterns:
        match = re.search(pattern, text[:2000], re.DOTALL)
        if match:
            title = match.group(1).strip()
            # Clean up
            title = re.sub(r'\s+', ' ', title)
            title = re.sub(r'Insert\s+', '', title)
            
            # Filter out non-title text
            if any(word in title.lower() for word in ['table', 'figure', 'source', 'page']):
                continue
            
            if 15 < len(title) < 120:
                return title
    
    return None


def extract_opinion_paragraphs(text):
    """
    CRITICAL FIX: Find paragraphs with actual analyst opinions
    Look for "We upgrade/downgrade/believe/expect/maintain" phrases
    These indicate the analyst's actual view, not just facts
    """
    # Opinion trigger phrases (what analysts write when giving opinions)
    opinion_triggers = [
        r'We upgrade \w+ to (ADD|HOLD|REDUCE)',
        r'We downgrade \w+ to (ADD|HOLD|REDUCE)',
        r'We maintain our (ADD|HOLD|REDUCE)',
        r'We reiterate our (ADD|HOLD|REDUCE)',
        r'We believe (?:the |that )',
        r'We expect (?:the |that )',
        r'We think (?:the |that )',
        r'We are (positive|negative|optimistic|cautious)',
        r'We view (?:the |this )',
        r'In our view,',
        r'We initiate coverage .* with (?:an |a )?(ADD|HOLD|REDUCE)',
    ]
    
    opinion_paragraphs = []
    
    # Search for each trigger phrase
    for trigger in opinion_triggers:
        matches = re.finditer(trigger, text, re.IGNORECASE)
        
        for match in matches:
            # Get position of trigger
            trigger_pos = match.start()
            
            # Extract paragraph containing this trigger
            # Go back to find paragraph start (previous double newline or start of text)
            para_start = text.rfind('\n\n', max(0, trigger_pos - 500), trigger_pos)
            if para_start == -1:
                para_start = max(0, trigger_pos - 500)
            
            # Go forward to find paragraph end (next double newline or section marker)
            para_end = trigger_pos + 1000
            
            # Look for paragraph end markers
            end_markers = ['\n\n', 'Figure ', 'Table ', 'SOURCES:', '\n■']
            for marker in end_markers:
                marker_pos = text.find(marker, trigger_pos + 50, trigger_pos + 1000)
                if marker_pos != -1 and marker_pos < para_end:
                    para_end = marker_pos
            
            # Extract the paragraph
            paragraph = text[para_start:para_end].strip()
            
            # Clean up
            paragraph = re.sub(r'\n+', ' ', paragraph)  # Remove newlines
            paragraph = re.sub(r'\s+', ' ', paragraph)  # Normalize spaces
            
            # Only keep if it's substantial and not a table
            if 100 < len(paragraph) < 1500:
                # Check it's not a table (tables have lots of numbers)
                numbers = re.findall(r'\d+\.?\d*', paragraph)
                if len(numbers) < 20:  # Not a table
                    opinion_paragraphs.append({
                        'text': paragraph,
                        'trigger': match.group(0),
                        'position': trigger_pos
                    })
    
    # Sort by position (chronological order)
    opinion_paragraphs.sort(key=lambda x: x['position'])
    
    return opinion_paragraphs


def extract_section_headers(text):
    """
    Extract section headers that contain opinions
    Headers like "Look forward, time to buy" or "Worst could be over"
    """
    opinion_headers = []
    
    # Look for short lines (20-80 chars) that sound like opinions
    # These are usually section headers
    lines = text[:5000].split('\n')
    
    opinion_keywords = [
        'upgrade', 'downgrade', 'buy', 'sell', 'positive', 'negative',
        'improve', 'decline', 'strong', 'weak', 'attractive', 'expensive',
        'opportunity', 'risk', 'outlook', 'view', 'expect', 'believe',
        'recovery', 'growth', 'momentum', 'time to', 'wait', 'ready'
    ]
    
    for line in lines:
        line = line.strip()
        
        # Check if line is right length and contains opinion keywords
        if 20 < len(line) < 100:
            lower_line = line.lower()
            if any(keyword in lower_line for keyword in opinion_keywords):
                # Make sure it's not a table header or bullet
                if not line.startswith('■') and not re.match(r'^[\d\s.,%]+$', line):
                    opinion_headers.append(line)
    
    return opinion_headers


def combine_opinion_text(title, opinion_paragraphs, opinion_headers):
    """
    Combine opinion-rich text
    Prioritize: Title > Opinion paragraphs > Section headers
    """
    combined = []
    total_chars = 0
    max_chars = 2000
    
    # Add title
    if title:
        combined.append(f"TITLE: {title}\n\n")
        total_chars += len(title) + 10
    
    # Add section headers (these are often the best opinion signals)
    if opinion_headers and total_chars < max_chars:
        for header in opinion_headers[:3]:  # Top 3 headers
            if total_chars + len(header) < max_chars:
                combined.append(f"SECTION: {header}\n\n")
                total_chars += len(header) + 15
    
    # Add opinion paragraphs (the actual analyst views)
    if opinion_paragraphs:
        for para_info in opinion_paragraphs:
            if total_chars >= max_chars:
                break
            
            para_text = para_info['text']
            remaining = max_chars - total_chars
            
            if len(para_text) <= remaining:
                combined.append(f"{para_text}\n\n")
                total_chars += len(para_text) + 2
            else:
                # Add what we can fit
                combined.append(para_text[:remaining])
                break
    
    final_text = ''.join(combined)
    
    return final_text


def get_finbert_sentiment(text):
    """Run FinBERT-Tone sentiment analysis"""
    if not text or len(text) < 20:
        return None
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                      max_length=512, padding=True)
    
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
    """Extract metadata from report"""
    metadata = {}
    
    # Recommendation
    rec_patterns = [
        r'Upgrade to (ADD|HOLD|REDUCE)',
        r'Downgrade to (ADD|HOLD|REDUCE)',
        r'Reiterate (ADD|HOLD|REDUCE)',
        r'Initiate coverage with (ADD|HOLD|REDUCE)',
        r'Maintain (ADD|HOLD|REDUCE)',
        r'\b(ADD|HOLD|REDUCE)\b.*\(previously',
    ]
    
    for pattern in rec_patterns:
        match = re.search(pattern, text[:3000], re.IGNORECASE)
        if match:
            rec = match.group(1).upper()
            metadata['recommendation'] = rec
            break
    
    # Price target - FIX: Handle decimal point issues
    target_patterns = [
        r'[Tt]arget price[:\s]+S?\$\s*([\d.]+)',
        r'[Tt]\.?[Pp]\.?[:\s]+S?\$\s*([\d.]+)',
        r'TP of S?\$\s*([\d.]+)',
    ]
    for pattern in target_patterns:
        match = re.search(pattern, text[:3000])
        if match:
            try:
                price_str = match.group(1).rstrip('.')  # Remove trailing dot
                metadata['price_target'] = float(price_str)
                break
            except ValueError:
                continue
    
    # Current price
    price_patterns = [
        r'Current price[:\s]+S?\$\s*([\d.]+)',
    ]
    for pattern in price_patterns:
        match = re.search(pattern, text[:3000])
        if match:
            try:
                price_str = match.group(1).rstrip('.')
                metadata['price_at_report'] = float(price_str)
                break
            except ValueError:
                continue
    
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
    """Process report using opinion-focused extraction"""
    print(f"\n{'='*60}")
    print(f"Processing: {pdf_path.name}")
    print('='*60)
    
    # Extract ticker
    ticker_result = extract_ticker_from_filename(pdf_path.name)
    if not ticker_result:
        raise ValueError(f"Could not extract ticker from: {pdf_path.name}")
    
    base_ticker, sgx_ticker = ticker_result
    print(f"Ticker: {base_ticker} → {sgx_ticker}")
    
    # Extract text
    text = extract_text_from_pdf(pdf_path)
    if not text:
        raise ValueError("Could not extract text")
    
    print(f"Total text: {len(text):,} characters\n")
    
    # Extract metadata
    metadata = extract_metadata(text)
    print(f"Recommendation: {metadata.get('recommendation', 'Not found')}")
    
    # OPINION-FOCUSED EXTRACTION
    print("\n" + "-"*60)
    print("EXTRACTING OPINION TEXT:")
    print("-"*60)
    
    # 1. Extract title
    title = extract_title(text)
    if title:
        print(f"✓ Title: \"{title}\"")
    else:
        print("✗ Title: Not found")
    
    # 2. Extract opinion paragraphs (THE KEY FIX!)
    opinion_paragraphs = extract_opinion_paragraphs(text)
    if opinion_paragraphs:
        print(f"✓ Opinion Paragraphs: {len(opinion_paragraphs)} found")
        for i, para_info in enumerate(opinion_paragraphs[:3], 1):
            print(f"  {i}. Trigger: \"{para_info['trigger']}\"")
            print(f"     Preview: {para_info['text'][:80]}...")
    else:
        print("✗ Opinion Paragraphs: Not found")
    
    # 3. Extract opinion headers
    opinion_headers = extract_section_headers(text)
    if opinion_headers:
        print(f"✓ Opinion Headers: {len(opinion_headers)} found")
        for header in opinion_headers[:3]:
            print(f"  - \"{header}\"")
    else:
        print("✗ Opinion Headers: Not found")
    
    # 4. Combine opinion text
    combined_text = combine_opinion_text(title, opinion_paragraphs, opinion_headers)
    
    print("\n" + "-"*60)
    print(f"COMBINED OPINION TEXT: {len(combined_text)} characters")
    print("-"*60)
    print("Preview (first 500 chars):")
    print(combined_text[:500])
    if len(combined_text) > 500:
        print("...")
    print("-"*60)
    
    # Sentiment analysis
    print("\nRunning sentiment analysis...")
    sentiment = get_finbert_sentiment(combined_text)
    
    if sentiment:
        print(f"\n{'='*60}")
        print(f"SENTIMENT: {sentiment['label'].upper()} ({sentiment['score']:+.3f})")
        print('='*60)
        print(f"  Positive: {sentiment['positive']:.3f}")
        print(f"  Neutral:  {sentiment['neutral']:.3f}")
        print(f"  Negative: {sentiment['negative']:.3f}")
    else:
        print("\n⚠️  Could not analyze sentiment")
        sentiment = {'score': 0.0, 'label': 'neutral', 
                    'negative': 0.0, 'neutral': 1.0, 'positive': 0.0}
    
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
        "executive_summary": combined_text[:1000],
        "key_catalysts": [],
        "key_risks": [],
        "upside_pct": round(upside_pct, 1) if upside_pct else None,
        "report_age_days": 0,
        "pdf_filename": pdf_path.name,
        "extraction_method": "opinion_focused_v4"
    }
    
    report_date = metadata.get('report_date', datetime.now().strftime('%Y-%m-%d'))
    output_filename = f"{base_ticker}_{report_date}.json"
    
    return output, output_filename


def is_already_processed(pdf_path, json_dir):
    """Check if PDF has already been processed"""
    ticker_result = extract_ticker_from_filename(pdf_path.name)
    if not ticker_result:
        return False
    
    base_ticker, _ = ticker_result
    pattern = f"{base_ticker}_*.json"
    existing_jsons = list(json_dir.glob(pattern))
    
    if not existing_jsons:
        return False
    
    pdf_mtime = pdf_path.stat().st_mtime
    
    for json_file in existing_jsons:
        json_mtime = json_file.stat().st_mtime
        if json_mtime < pdf_mtime:
            return False
    
    return True


def main():
    """Process PDFs using opinion-focused extraction"""
    print("\n" + "="*60)
    print("OPINION-FOCUSED PROCESSOR (Final Fix)")
    print("="*60)
    print("Extracts: Title + Opinion Paragraphs + Opinion Headers")
    print("="*60 + "\n")
    
    force_reprocess = '--force' in sys.argv
    
    pdf_files = list(ANALYST_PDF_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDFs found in: {ANALYST_PDF_DIR}")
        return
    
    # Filter PDFs
    if force_reprocess:
        print("🔧 FORCE MODE: Reprocessing all PDFs\n")
        pdfs_to_process = pdf_files
        skipped_pdfs = []
    else:
        new_pdfs = []
        skipped_pdfs = []
        
        for pdf_file in pdf_files:
            if is_already_processed(pdf_file, ANALYST_REPORTS_DIR):
                skipped_pdfs.append(pdf_file.name)
            else:
                new_pdfs.append(pdf_file)
        
        pdfs_to_process = new_pdfs
    
    print(f"Found {len(pdf_files)} total PDF(s)")
    print(f"  {len(pdfs_to_process)} to process")
    print(f"  {len(skipped_pdfs)} already processed")
    
    if skipped_pdfs and not force_reprocess:
        print(f"\nSkipping {len(skipped_pdfs)} already-processed files")
    
    if not pdfs_to_process:
        print("\n✅ All PDFs already processed!")
        print("Use --force flag to reprocess all")
        return
    
    print(f"\n{'='*60}")
    print(f"PROCESSING {len(pdfs_to_process)} PDF(s)")
    print('='*60)
    
    processed = 0
    failed = 0
    
    for pdf_file in pdfs_to_process:
        try:
            output_data, output_filename = process_report(pdf_file)
            
            output_path = ANALYST_REPORTS_DIR / output_filename
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"\n✓ Saved: {output_filename}")
            processed += 1
            
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Processed: {processed}")
    print(f"  Failed: {failed}")
    print(f"  Skipped: {len(skipped_pdfs)}")
    print("="*60)
    
    if processed > 0:
        print(f"\n✓ JSON files saved to: {ANALYST_REPORTS_DIR}")
        print("\n📊 Next steps:")
        print("  1. Open scanner in browser")
        print("  2. Click '🔄 Refresh Reports' button")
        print("  3. Check sentiment scores!")
        print("\n💡 Expected improvements:")
        print("  - Wilmar: Should now be POSITIVE")
        print("  - Reports with 'We upgrade': Should be POSITIVE")
        print("  - Reports with 'We downgrade': Should be NEGATIVE")


if __name__ == "__main__":
    main()