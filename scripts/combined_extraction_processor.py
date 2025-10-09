# File: scripts/combined_extraction_processor.py
"""
Combined Extraction Analyst Report Processor
Uses Approach 3: Title + Bullets + First Section + Recommendation Context
Maximizes opinion density within FinBERT's 512 token limit
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
    """
    Extract report title (the analyst's headline opinion)
    Title is usually between company name and first bullet point
    """
    # Look for common title patterns
    title_patterns = [
        # Pattern 1: Line before first bullet
        r'Insert\s+Insert\s+(.+?)\s*■',
        # Pattern 2: After company name, before bullets
        r'(?:Singapore|Malaysia|Thailand|Indonesia)\s+(.+?)\s*■',
        # Pattern 3: Standalone short line (20-80 chars) before bullets
        r'\n(.{20,80})\s*\n\s*■',
    ]
    
    for pattern in title_patterns:
        match = re.search(pattern, text[:2000], re.DOTALL)
        if match:
            title = match.group(1).strip()
            # Clean up title
            title = re.sub(r'\s+', ' ', title)  # Normalize whitespace
            if 20 < len(title) < 150:  # Reasonable title length
                return title
    
    return None


def extract_bullets(text):
    """Extract bullet points (■ markers)"""
    # Find first bullet
    first_bullet = text.find('■')
    if first_bullet == -1:
        return None
    
    # Section end markers
    section_end_markers = [
        '\n\n\n',
        'Figure 1:',
        'Table 1:',
        'Financial Summary',
        'SOURCES:',
        'BY THE NUMBERS',
        'Key changes',
    ]
    
    # Find end of bullet section
    start = first_bullet
    end = len(text)
    
    for marker in section_end_markers:
        marker_pos = text.find(marker, start)
        if marker_pos > start and marker_pos < end:
            end = marker_pos
    
    # Extract bullets
    bullet_section = text[start:end].strip()
    
    # Limit to reasonable length
    if len(bullet_section) > 800:
        bullet_section = bullet_section[:800]
    
    return bullet_section


def extract_first_section(text):
    """
    Extract first section after bullets (usually contains main opinion)
    Includes section heading + first paragraph
    """
    # Find where bullets end
    first_bullet = text.find('■')
    if first_bullet == -1:
        search_start = 500
    else:
        # Find end of bullets (double newline or section marker)
        search_start = first_bullet + 200
    
    # Look for first section heading (short line in title case)
    section_heading_pattern = r'\n([A-Z][^.\n]{10,80})\s*\n'
    match = re.search(section_heading_pattern, text[search_start:search_start+2000])
    
    if not match:
        return None
    
    heading_start = search_start + match.start()
    heading = match.group(1).strip()
    
    # Extract paragraph after heading (until next heading or table)
    para_start = heading_start + len(heading)
    para_text = text[para_start:para_start+1500]
    
    # Stop at next section
    stop_markers = [
        '\nFigure',
        '\nTable',
        '\n\n\n',
        'SOURCES:',
    ]
    
    end_pos = len(para_text)
    for marker in stop_markers:
        marker_pos = para_text.find(marker)
        if 100 < marker_pos < end_pos:  # At least 100 chars of content
            end_pos = marker_pos
    
    para_text = para_text[:end_pos].strip()
    
    # Combine heading and paragraph
    section_text = f"{heading}\n\n{para_text}"
    
    return section_text


def extract_recommendation_context(text):
    """
    Find recommendation keywords and extract surrounding context
    Keywords: Upgrade, Downgrade, Reiterate, Initiate, Maintain
    """
    recommendation_keywords = [
        'Upgrade to ADD',
        'Upgrade to Add',
        'Downgrade to REDUCE',
        'Downgrade to Reduce',
        'Downgrade to HOLD',
        'Downgrade to Hold',
        'Reiterate ADD',
        'Reiterate Add',
        'Reiterate HOLD',
        'Reiterate Hold',
        'Reiterate REDUCE',
        'Reiterate Reduce',
        'Initiate coverage with ADD',
        'Initiate coverage with Add',
        'Maintain ADD',
        'Maintain Add',
    ]
    
    for keyword in recommendation_keywords:
        pos = text.find(keyword)
        if pos != -1:
            # Extract 400 chars before and 600 chars after
            start = max(0, pos - 400)
            end = min(len(text), pos + 600)
            context = text[start:end].strip()
            
            return context
    
    return None


def combine_opinion_text(title, bullets, first_section, rec_context):
    """
    Combine extracted sections intelligently
    Prioritize: Title > Recommendation > First Section > Bullets
    Keep under 2000 chars for FinBERT
    """
    combined = []
    total_chars = 0
    max_chars = 2000
    
    # Priority 1: Title (always include if available)
    if title:
        combined.append(f"TITLE: {title}")
        total_chars += len(title) + 10
    
    # Priority 2: Recommendation context (critical opinion signal)
    if rec_context and total_chars < max_chars:
        remaining = max_chars - total_chars
        rec_text = rec_context[:remaining]
        combined.append(f"\nRECOMMENDATION: {rec_text}")
        total_chars += len(rec_text) + 20
    
    # Priority 3: First section (usually contains main argument)
    if first_section and total_chars < max_chars:
        remaining = max_chars - total_chars
        section_text = first_section[:remaining]
        combined.append(f"\n\n{section_text}")
        total_chars += len(section_text)
    
    # Priority 4: Bullets (fill remaining space)
    if bullets and total_chars < max_chars:
        remaining = max_chars - total_chars
        bullet_text = bullets[:remaining]
        combined.append(f"\n\n{bullet_text}")
        total_chars += len(bullet_text)
    
    final_text = ''.join(combined)
    
    return final_text


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
    
    # Price target
    target_patterns = [
        r'[Tt]arget price[:\s]+S?\$\s*([\d.]+)',
        r'[Tt]\.?[Pp]\.?[:\s]+S?\$\s*([\d.]+)',
        r'TP of S?\$\s*([\d.]+)',
    ]
    for pattern in target_patterns:
        match = re.search(pattern, text[:3000])
        if match:
            metadata['price_target'] = float(match.group(1))
            break
    
    # Current price
    price_patterns = [
        r'Current price[:\s]+S?\$\s*([\d.]+)',
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
    """Process report using combined extraction approach"""
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
    
    # COMBINED EXTRACTION - Approach 3
    print("\n" + "-"*60)
    print("EXTRACTING OPINION-DENSE SECTIONS:")
    print("-"*60)
    
    # 1. Extract title
    title = extract_title(text)
    if title:
        print(f"✓ Title: \"{title}\"")
    else:
        print("✗ Title: Not found")
    
    # 2. Extract bullets
    bullets = extract_bullets(text)
    if bullets:
        print(f"✓ Bullets: {len(bullets)} chars")
    else:
        print("✗ Bullets: Not found")
    
    # 3. Extract first section
    first_section = extract_first_section(text)
    if first_section:
        print(f"✓ First Section: {len(first_section)} chars")
        # Show section heading
        section_lines = first_section.split('\n')
        if section_lines:
            print(f"  Heading: \"{section_lines[0][:60]}...\"")
    else:
        print("✗ First Section: Not found")
    
    # 4. Extract recommendation context
    rec_context = extract_recommendation_context(text)
    if rec_context:
        print(f"✓ Recommendation Context: {len(rec_context)} chars")
    else:
        print("✗ Recommendation Context: Not found")
    
    # 5. Combine intelligently
    combined_text = combine_opinion_text(title, bullets, first_section, rec_context)
    
    print("\n" + "-"*60)
    print(f"COMBINED TEXT: {len(combined_text)} characters")
    print("-"*60)
    print("Preview (first 400 chars):")
    print(combined_text[:400])
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
        "extraction_method": "combined_approach_v3"
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
    
    # Look for existing JSON
    pattern = f"{base_ticker}_*.json"
    existing_jsons = list(json_dir.glob(pattern))
    
    if not existing_jsons:
        return False
    
    # Check if PDF is newer than JSON
    pdf_mtime = pdf_path.stat().st_mtime
    
    for json_file in existing_jsons:
        json_mtime = json_file.stat().st_mtime
        if json_mtime < pdf_mtime:
            return False
    
    return True


def main():
    """Process PDFs using combined extraction approach"""
    print("\n" + "="*60)
    print("COMBINED EXTRACTION PROCESSOR (Approach 3)")
    print("="*60)
    print("Extracts: Title + Bullets + First Section + Recommendation")
    print("="*60 + "\n")
    
    # Check for force flag
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
        if len(skipped_pdfs) <= 5:
            for name in skipped_pdfs:
                print(f"  ✓ {name}")
        else:
            for name in skipped_pdfs[:3]:
                print(f"  ✓ {name}")
            print(f"  ... and {len(skipped_pdfs) - 3} more")
    
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
            
            # Save JSON
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
        print("  3. Check sentiment scores in results!")


if __name__ == "__main__":
    main()