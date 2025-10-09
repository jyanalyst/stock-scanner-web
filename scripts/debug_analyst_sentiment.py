# File: scripts/debug_analyst_sentiment.py
"""
Debug script to compare sentiment analysis between reports
Analyzes why some reports get positive sentiment while others get neutral
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


def get_finbert_sentiment(text, max_length=512):
    """Run FinBERT-Tone sentiment analysis"""
    # Truncate to reasonable length
    text_sample = text[:3000]
    
    # Tokenize
    inputs = tokenizer(text_sample, return_tensors="pt", truncation=True, 
                      max_length=max_length, padding=True)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # FinBERT-Tone outputs: [negative, neutral, positive]
    scores = predictions[0].tolist()
    
    # Convert to single score: -1 (negative) to +1 (positive)
    sentiment_score = scores[2] - scores[0]  # positive - negative
    
    # Determine label
    max_idx = predictions[0].argmax().item()
    labels = ['negative', 'neutral', 'positive']
    sentiment_label = labels[max_idx]
    
    # Return detailed scores
    return {
        'score': sentiment_score,
        'label': sentiment_label,
        'negative': scores[0],
        'neutral': scores[1],
        'positive': scores[2]
    }


def extract_executive_summary_original(text):
    """Original extraction method from your script"""
    headers = [
        'Investment [Tt]hesis',
        'Executive [Ss]ummary',
        'Initiate coverage',
        'Key highlights',
        'Investment [Cc]ase',
    ]
    
    for header in headers:
        pattern = rf'{header}\s*(.{{500,3000}}?)(?:\n\n|\n[A-Z]{{2,}})'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip(), f"Matched header: {header}"
    
    # Fallback
    return text[500:2500], "Using fallback: text[500:2500]"


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


def extract_recommendation(text):
    """Extract recommendation from text"""
    rec_patterns = [
        r'\b(ADD|HOLD|REDUCE|BUY|SELL|OUTPERFORM|UNDERPERFORM|NEUTRAL)\b',
        r'[Rr]ating[:\s]+(ADD|HOLD|REDUCE|BUY|SELL)',
        r'[Rr]ecommendation[:\s]+(ADD|HOLD|REDUCE|BUY|SELL)',
    ]
    
    for pattern in rec_patterns:
        match = re.search(pattern, text[:3000])
        if match:
            return match.group(1).upper()
    
    return None


def find_opinion_sections(text):
    """Find sections likely containing analyst opinions"""
    opinion_keywords = [
        'recommend', 'rating', 'target', 'outlook', 'view', 
        'bullish', 'bearish', 'positive', 'negative',
        'upgrade', 'downgrade', 'maintain', 'initiate',
        'attractive', 'favorable', 'strong', 'robust'
    ]
    
    # Split into paragraphs
    paragraphs = text.split('\n\n')
    opinion_sections = []
    
    for i, para in enumerate(paragraphs):
        if len(para) > 100:  # Substantial paragraph
            keyword_count = sum(1 for keyword in opinion_keywords if keyword in para.lower())
            if keyword_count >= 2:  # At least 2 opinion keywords
                opinion_sections.append({
                    'paragraph_num': i,
                    'length': len(para),
                    'keyword_count': keyword_count,
                    'preview': para[:200] + '...' if len(para) > 200 else para
                })
    
    return opinion_sections


def analyze_report(pdf_filename):
    """Comprehensive analysis of a single report"""
    print("=" * 80)
    print(f"ANALYZING: {pdf_filename}")
    print("=" * 80)
    
    pdf_path = ANALYST_PDF_DIR / pdf_filename
    
    if not pdf_path.exists():
        print(f"❌ File not found: {pdf_path}")
        return None
    
    # Step 1: Extract full text
    print("\n📄 STEP 1: EXTRACTING FULL TEXT")
    print("-" * 80)
    full_text = extract_text_from_pdf(pdf_path)
    print(f"✓ Extracted {len(full_text):,} characters from PDF")
    print(f"✓ First 500 characters:")
    print(full_text[:500])
    print()
    
    # Step 2: Extract executive summary (original method)
    print("\n📝 STEP 2: EXTRACTING EXECUTIVE SUMMARY (Original Method)")
    print("-" * 80)
    exec_summary, extraction_method = extract_executive_summary_original(full_text)
    print(f"✓ {extraction_method}")
    print(f"✓ Extracted {len(exec_summary):,} characters")
    print(f"\n✓ Executive Summary Content:")
    print("-" * 80)
    print(exec_summary)
    print("-" * 80)
    print()
    
    # Step 3: Run sentiment analysis on executive summary
    print("\n🎯 STEP 3: SENTIMENT ANALYSIS ON EXECUTIVE SUMMARY")
    print("-" * 80)
    sentiment = get_finbert_sentiment(exec_summary)
    print(f"Overall Sentiment: {sentiment['label'].upper()} ({sentiment['score']:.3f})")
    print(f"  Negative: {sentiment['negative']:.3f}")
    print(f"  Neutral:  {sentiment['neutral']:.3f}")
    print(f"  Positive: {sentiment['positive']:.3f}")
    print()
    
    # Step 4: Find opinion sections
    print("\n🔍 STEP 4: FINDING OPINION-RICH SECTIONS")
    print("-" * 80)
    opinion_sections = find_opinion_sections(full_text)
    print(f"Found {len(opinion_sections)} opinion-rich paragraphs:")
    for section in opinion_sections[:3]:  # Show top 3
        print(f"\n  Paragraph #{section['paragraph_num']} ({section['length']} chars, {section['keyword_count']} keywords):")
        print(f"  {section['preview']}")
    print()
    
    # Step 5: Analyze opinion sections
    if opinion_sections:
        print("\n🎯 STEP 5: SENTIMENT ANALYSIS ON TOP OPINION SECTION")
        print("-" * 80)
        top_section = opinion_sections[0]
        paragraphs = full_text.split('\n\n')
        top_text = paragraphs[top_section['paragraph_num']]
        
        print(f"Analyzing paragraph #{top_section['paragraph_num']}:")
        print("-" * 80)
        print(top_text[:1000])
        print("-" * 80)
        
        opinion_sentiment = get_finbert_sentiment(top_text)
        print(f"\nOpinion Section Sentiment: {opinion_sentiment['label'].upper()} ({opinion_sentiment['score']:.3f})")
        print(f"  Negative: {opinion_sentiment['negative']:.3f}")
        print(f"  Neutral:  {opinion_sentiment['neutral']:.3f}")
        print(f"  Positive: {opinion_sentiment['positive']:.3f}")
        print()
    
    # Step 6: Extract metadata
    print("\n📊 STEP 6: EXTRACTING METADATA")
    print("-" * 80)
    recommendation = extract_recommendation(full_text)
    catalysts, risks = extract_catalysts_and_risks(full_text)
    
    print(f"Recommendation: {recommendation if recommendation else 'Not found'}")
    print(f"Catalysts found: {len(catalysts)}")
    if catalysts:
        for i, catalyst in enumerate(catalysts, 1):
            print(f"  {i}. {catalyst[:100]}...")
    
    print(f"\nRisks found: {len(risks)}")
    if risks:
        for i, risk in enumerate(risks, 1):
            print(f"  {i}. {risk[:100]}...")
    print()
    
    # Summary comparison
    print("\n📈 COMPARISON SUMMARY")
    print("=" * 80)
    print(f"Executive Summary Sentiment: {sentiment['label'].upper()} ({sentiment['score']:.3f})")
    if opinion_sections:
        print(f"Opinion Section Sentiment:   {opinion_sentiment['label'].upper()} ({opinion_sentiment['score']:.3f})")
    print(f"Recommendation:              {recommendation if recommendation else 'N/A'}")
    print(f"Catalysts/Risks:             {len(catalysts)}/{len(risks)}")
    print("=" * 80)
    print()
    
    return {
        'filename': pdf_filename,
        'total_chars': len(full_text),
        'exec_summary_chars': len(exec_summary),
        'extraction_method': extraction_method,
        'sentiment': sentiment,
        'opinion_sentiment': opinion_sentiment if opinion_sections else None,
        'recommendation': recommendation,
        'catalysts_count': len(catalysts),
        'risks_count': len(risks),
        'opinion_sections_count': len(opinion_sections)
    }


def compare_reports(positive_file, neutral_file):
    """Compare two reports side by side"""
    print("\n" + "=" * 80)
    print("COMPARATIVE ANALYSIS")
    print("=" * 80)
    print()
    
    print("Analyzing POSITIVE report...")
    positive_result = analyze_report(positive_file)
    
    print("\n" + "=" * 80)
    print()
    
    print("Analyzing NEUTRAL report...")
    neutral_result = analyze_report(neutral_file)
    
    # Final comparison
    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 80)
    print()
    
    comparison_data = [
        ("Metric", "POSITIVE Report", "NEUTRAL Report"),
        ("-" * 30, "-" * 25, "-" * 25),
        ("Filename", positive_result['filename'], neutral_result['filename']),
        ("Total Characters", f"{positive_result['total_chars']:,}", f"{neutral_result['total_chars']:,}"),
        ("Exec Summary Chars", f"{positive_result['exec_summary_chars']:,}", f"{neutral_result['exec_summary_chars']:,}"),
        ("Extraction Method", positive_result['extraction_method'][:40], neutral_result['extraction_method'][:40]),
        ("Sentiment Score", f"{positive_result['sentiment']['score']:.3f}", f"{neutral_result['sentiment']['score']:.3f}"),
        ("Sentiment Label", positive_result['sentiment']['label'], neutral_result['sentiment']['label']),
        ("Recommendation", positive_result['recommendation'] or 'N/A', neutral_result['recommendation'] or 'N/A'),
        ("Catalysts Found", str(positive_result['catalysts_count']), str(neutral_result['catalysts_count'])),
        ("Risks Found", str(positive_result['risks_count']), str(neutral_result['risks_count'])),
        ("Opinion Sections", str(positive_result['opinion_sections_count']), str(neutral_result['opinion_sections_count'])),
    ]
    
    for row in comparison_data:
        print(f"{row[0]:<30} {row[1]:<25} {row[2]:<25}")
    
    print()
    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    # Analyze differences
    if positive_result['extraction_method'] != neutral_result['extraction_method']:
        print("⚠️  DIFFERENT EXTRACTION METHODS USED!")
        print(f"   Positive: {positive_result['extraction_method']}")
        print(f"   Neutral:  {neutral_result['extraction_method']}")
    
    if positive_result['catalysts_count'] > neutral_result['catalysts_count']:
        print(f"📊 Positive report has more catalysts ({positive_result['catalysts_count']} vs {neutral_result['catalysts_count']})")
    
    if positive_result['opinion_sections_count'] > neutral_result['opinion_sections_count']:
        print(f"💬 Positive report has more opinion sections ({positive_result['opinion_sections_count']} vs {neutral_result['opinion_sections_count']})")
    
    print()


def main():
    """Main debug script"""
    print("=" * 80)
    print("ANALYST REPORT SENTIMENT DEBUGGER")
    print("=" * 80)
    print()
    
    # The reports to compare based on your output
    positive_report = "20250516_CSE_544.pdf"  # Got 0.69 positive
    neutral_report = "20250820_AEM_AWX.pdf"   # Got 0.00 neutral
    
    print(f"Comparing:")
    print(f"  POSITIVE: {positive_report} (scored 0.69)")
    print(f"  NEUTRAL:  {neutral_report} (scored 0.00)")
    print()
    
    compare_reports(positive_report, neutral_report)
    
    print("\n✓ Debug analysis complete!")
    print("\nNext steps:")
    print("1. Review the extraction methods used for each report")
    print("2. Compare the actual text being analyzed")
    print("3. Check if opinion sections are being captured correctly")
    print("4. Verify if catalysts/risks extraction is working")


if __name__ == "__main__":
    main()