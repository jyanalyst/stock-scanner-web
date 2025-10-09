# File: scripts/tight_opinion_processor.py
"""
Tight Opinion Extraction Processor
FINAL VERSION: Extracts ONLY the immediate opinion sentences
No headers, no disclaimers, no boilerplate - just pure analyst opinions
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
    """Extract report title"""
    title_patterns = [
        r'(?:Ltd|Limited|Corp|Corporation|Holdings|Group|International|REIT|Marine|Oil)\s+(.{15,100}?)\s*■',
        r'\n(.{20,100})\s*\n+\s*■',
    ]
    
    for pattern in title_patterns:
        match = re.search(pattern, text[:2000], re.DOTALL)
        if match:
            title = match.group(1).strip()
            title = re.sub(r'\s+', ' ', title)
            title = re.sub(r'Insert\s+', '', title)
            
            if any(word in title.lower() for word in ['table', 'figure', 'source', 'page', 'important', 'disclosure']):
                continue
            
            if 15 < len(title) < 120:
                return title
    
    return None


def is_boilerplate(text):
    """Check if text is boilerplate/disclaimer/header"""
    boilerplate_keywords = [
        'IMPORTANT DISCLOSURES',
        'CERTIFICATIONS',
        'Company Note',
        'SOURCES:',
        'Bloomberg',
        'Figure',
        'Table',
        '│',
        'LSEG ESG',
        'Combined Score',
        'EFA Platform',
        'Powered by',
        'Insert',
    ]
    
    return any(keyword in text for keyword in boilerplate_keywords)


def extract_tight_opinion_sentences(text):
    """
    CRITICAL FIX: Extract ONLY the immediate sentences containing opinions
    Maximum 200 chars per opinion (one tight paragraph)
    No headers, no disclaimers, no boilerplate
    """
    # Opinion trigger phrases
    opinion_triggers = [
        # Strong opinion indicators
        (r'We upgrade \w+ to (ADD|Add)', 150),
        (r'We downgrade \w+ to (REDUCE|Reduce|HOLD|Hold)', 150),
        (r'We maintain (?:our )?(ADD|Add|HOLD|Hold|REDUCE|Reduce)', 150),
        (r'We reiterate (?:our )?(ADD|Add|HOLD|Hold|REDUCE|Reduce)', 150),
        (r'We initiate coverage .* with (?:an |a )?(ADD|Add|HOLD|Hold|REDUCE|Reduce)', 150),
        
        # Opinion phrases
        (r'We believe (?:the |that |investors )', 150),
        (r'We expect (?:the |that )', 150),
        (r'We think (?:the |that )', 120),
        (r'We are (positive|negative|optimistic|cautious)', 120),
        (r'We view (?:the |this )', 120),
        (r'In our view,', 120),
        (r'we see (?:the |a |an )', 100),
        
        # Recommendation phrases
        (r'Look forward', 80),
        (r'time to (buy|sell)', 80),
        (r'attractive (entry point|valuation)', 100),
        (r'remain (positive|negative|optimistic|cautious)', 100),
    ]
    
    opinion_sentences = []
    
    for trigger_pattern, max_length in opinion_triggers:
        matches = re.finditer(trigger_pattern, text, re.IGNORECASE)
        
        for match in matches:
            trigger_pos = match.start()
            trigger_text = match.group(0)
            
            # Extract tight context around trigger
            # Go back to sentence start (. or newline)
            sent_start = trigger_pos
            for i in range(trigger_pos - 1, max(0, trigger_pos - 200), -1):
                if text[i] in '.!\n':
                    sent_start = i + 1
                    break
            
            # Go forward to sentence end
            sent_end = trigger_pos + max_length
            for i in range(trigger_pos, min(len(text), trigger_pos + max_length)):
                if text[i] in '.!' and i > trigger_pos + 30:  # At least 30 chars after trigger
                    sent_end = i + 1
                    break
            
            # Extract the sentence
            sentence = text[sent_start:sent_end].strip()
            
            # Clean up
            sentence = re.sub(r'\n+', ' ', sentence)
            sentence = re.sub(r'\s+', ' ', sentence)
            
            # Quality checks
            if len(sentence) < 30:  # Too short
                continue
            
            if len(sentence) > 300:  # Too long, trim
                sentence = sentence[:300]
            
            # Skip if boilerplate
            if is_boilerplate(sentence):
                continue
            
            # Skip if too many numbers (likely a table)
            numbers = re.findall(r'\d+', sentence)
            if len(numbers) > 10:
                continue
            
            opinion_sentences.append({
                'text': sentence,
                'trigger': trigger_text,
                'position': trigger_pos
            })
    
    # Remove duplicates (same position area)
    unique_opinions = []
    seen_positions = set()
    
    for opinion in sorted(opinion_sentences, key=lambda x: x['position']):
        pos = opinion['position']
        # Check if we've already captured this area (within 100 chars)
        if not any(abs(pos - seen_pos) < 100 for seen_pos in seen_positions):
            unique_opinions.append(opinion)
            seen_positions.add(pos)
    
    return unique_opinions


def combine_tight_opinions(title, opinion_sentences):
    """
    Combine ONLY tight opinion sentences
    No boilerplate, no headers, pure opinions only
    """
    combined = []
    total_chars = 0
    max_chars = 2000
    
    # Add title (if clean)
    if title and not is_boilerplate(title):
        combined.append(f"{title}\n\n")
        total_chars += len(title) + 2
    
    # Add opinion sentences
    for opinion_info in opinion_sentences:
        if total_chars >= max_chars:
            break
        
        sentence = opinion_info['text']
        
        # Skip if would exceed limit
        if total_chars + len(sentence) + 2 > max_chars:
            # Add what we can fit
            remaining = max_chars - total_chars
            if remaining > 50:  # Only if meaningful amount left
                combined.append(sentence[:remaining])
            break
        
        combined.append(f"{sentence}\n\n")
        total_chars += len(sentence) + 2
    
    final_text = ''.join(combined).strip()
    
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
        r'TP of S?\$\s*([\d.]+)',
    ]
    for pattern in target_patterns:
        match = re.search(pattern, text[:3000])
        if match:
            try:
                price_str = match.group(1).rstrip('.')
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
    """Process report using tight opinion extraction"""
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
    
    # TIGHT OPINION EXTRACTION
    print("\n" + "-"*60)
    print("EXTRACTING TIGHT OPINION SENTENCES:")
    print("-"*60)
    
    # Extract title
    title = extract_title(text)
    if title:
        print(f"✓ Title: \"{title}\"")
    else:
        print("✗ Title: Not found")
    
    # Extract tight opinion sentences
    opinion_sentences = extract_tight_opinion_sentences(text)
    
    if opinion_sentences:
        print(f"✓ Opinion Sentences: {len(opinion_sentences)} found")
        for i, opinion in enumerate(opinion_sentences[:5], 1):
            print(f"\n  {i}. [{opinion['trigger']}]")
            print(f"     \"{opinion['text'][:100]}...\"")
    else:
        print("✗ Opinion Sentences: Not found")
        print("⚠️  WARNING: No opinion text found! Sentiment will be unreliable.")
    
    # Combine tight opinions
    combined_text = combine_tight_opinions(title, opinion_sentences)
    
    print("\n" + "-"*60)
    print(f"COMBINED TEXT: {len(combined_text)} characters")
    print("-"*60)
    print("FULL TEXT TO ANALYZE:")
    print(combined_text)
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
        print('='*60)
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
        "extraction_method": "tight_opinion_v5"
    }
    
    report_date = metadata.get('report_date', datetime.now().strftime('%Y-%m-%d'))
    output_filename = f"{base_ticker}_{report_date}.json"
    
    return output, output_filename


def main():
    """Process PDFs using tight opinion extraction"""
    print("\n" + "="*60)
    print("TIGHT OPINION PROCESSOR (Final Version)")
    print("="*60)
    print("Extracts ONLY opinion sentences - no boilerplate!")
    print("="*60 + "\n")
    
    force_reprocess = '--force' in sys.argv
    
    pdf_files = list(ANALYST_PDF_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDFs found in: {ANALYST_PDF_DIR}")
        return
    
    if force_reprocess:
        print("🔧 FORCE MODE: Reprocessing all PDFs\n")
        pdfs_to_process = pdf_files
    else:
        print("💡 TIP: Use --force to reprocess all PDFs\n")
        pdfs_to_process = pdf_files[:3]  # Process first 3 for testing
        print(f"📊 Processing first {len(pdfs_to_process)} PDFs for testing...\n")
    
    print(f"Found {len(pdf_files)} total PDF(s)")
    print(f"  {len(pdfs_to_process)} to process\n")
    
    print('='*60)
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
    print("="*60)
    
    if processed > 0:
        print(f"\n✓ JSON files saved to: {ANALYST_REPORTS_DIR}")
        print("\n🎯 Expected Results:")
        print("  - Wilmar (Upgrade to ADD): Should be POSITIVE")
        print("  - Reports with upgrade/optimistic: POSITIVE")
        print("  - Reports with downgrade/cautious: NEGATIVE")
        print("\n💡 If you're satisfied, run with --force to process all PDFs")


if __name__ == "__main__":
    main()