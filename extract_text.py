import spacy
import os

# Load the spaCy English language model
nlp = spacy.load("en_core_web_sm")

def extract_text_from_txt(filepath):
    """Read a plain text contract file"""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def clean_and_split(raw_text):
    """Clean text and split into individual sentences"""
    # Remove extra whitespace and blank lines
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    cleaned = " ".join(lines)

    # Use spaCy to split into sentences
    doc = nlp(cleaned)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]
    return sentences

def display_results(sentences, filepath):
    """Print results in a readable format"""
    print(f"\n{'='*60}")
    print(f"CONTRACT TEXT EXTRACTION RESULTS")
    print(f"{'='*60}")
    print(f"File: {os.path.basename(filepath)}")
    print(f"Total sentences extracted: {len(sentences)}")
    print(f"{'='*60}\n")

    for i, sentence in enumerate(sentences, 1):
        print(f"[{i:03d}] {sentence}")
        print()

# ── TEST WITH A SAMPLE CONTRACT ────────────────────────────
sample_contract = """
STERILIZATION SERVICES AGREEMENT

This Agreement is entered into as of January 1, 2025, between Vantive 
Corporation and Steris International Inc.

1. TERM AND RENEWAL
This Agreement shall commence on the Effective Date and shall 
automatically renew for successive one-year periods unless either 
party provides written notice of termination no less than ninety 
days prior to the end of the then-current term.

2. PAYMENT TERMS
Invoices shall be due and payable within thirty days of receipt. 
Late payments shall accrue interest at a rate of 1.5% per month.

3. LIABILITY
In no event shall either party be liable for any indirect, 
incidental, or consequential damages. The aggregate liability 
of Steris shall not exceed the fees paid in the prior three months.

4. PRICE ADJUSTMENTS
Steris reserves the right to adjust pricing annually based on 
changes in the Consumer Price Index with no cap on increases.

5. TERMINATION
Either party may terminate this Agreement upon one hundred and 
eighty days written notice. Early termination by the customer
shall incur a termination fee equal to six months of average 
monthly fees.

6. FORCE MAJEURE
Neither party shall be liable for delays caused by circumstances 
beyond their reasonable control, including but not limited to 
acts of God, government actions, pandemics, labor disputes, 
or supply chain disruptions.

7. GOVERNING LAW
This Agreement shall be governed by the laws of the State of 
illinois and any disputes shall be resolved through binding 
arbitration in Chicago, Illinois.
"""

# Save sample contract to a text file
sample_path = "sample_contract.txt"
with open(sample_path, "w", encoding="utf-8") as f:
    f.write(sample_contract)
print(f"Sample contract saved as: {sample_path}")

# Extract and display
raw_text = extract_text_from_txt(sample_path)
sentences = clean_and_split(raw_text)
display_results(sentences, sample_path)

print("="*60)
print("Week 1 Complete! Text extraction is working.")
print("="*60)