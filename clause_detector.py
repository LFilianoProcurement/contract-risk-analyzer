import csv
import spacy
import os

# Load spaCy language model
nlp = spacy.load("en_core_web_sm")

# ── LOAD CLAUSE LIBRARY ────────────────────────────────────
def load_clause_library(filepath="clause_library_v2.csv"):
    """Load the clause library from CSV"""
    library = {}
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = row["Risk_Category"]
            if category not in library:
                library[category] = []
            library[category].append({
                "risk_level": row["Risk_Level"],
                "trigger_phrase": row["Trigger_Phrase"].lower(),
                "example": row["Example_Clause"]
            })
    print(f"Loaded {sum(len(v) for v in library.values())} clauses "
          f"across {len(library)} categories")
    return library

# ── LOAD CONTRACT TEXT ─────────────────────────────────────
def load_contract(filepath):
    """Load and clean contract text"""
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    return " ".join(lines)

# ── SPLIT INTO SENTENCES ───────────────────────────────────
def split_sentences(text):
    """Split contract text into sentences using spaCy"""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents 
            if len(sent.text.strip()) > 20]

# ── DETECT RISKY CLAUSES ───────────────────────────────────
def detect_clauses(sentences, library):
    """Check each sentence against the clause library"""
    findings = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        
        for category, clauses in library.items():
            for clause in clauses:
                trigger = clause["trigger_phrase"]
                
                # Check if trigger phrase appears in sentence
                if trigger in sentence_lower:
                    findings.append({
                        "sentence": sentence,
                        "category": category,
                        "risk_level": clause["risk_level"],
                        "trigger_found": trigger,
                        "example_clause": clause["example"]
                    })
                    break  # One match per category per sentence

    return findings

# ── CALCULATE RISK SCORE ───────────────────────────────────
def calculate_risk_score(findings):
    """Calculate overall contract risk score 0-100"""
    weights = {"Risky": 10, "Acceptable": 3, "Preferred": 0}
    
    # Max possible score based on 18 categories all being Risky
    max_score = 18 * 10
    
    total = sum(weights.get(f["risk_level"], 0) for f in findings)
    score = min(100, round((total / max_score) * 100))
    
    # Risk level label
    if score <= 25:
        label = "LOW RISK"
        action = "Proceed with standard review and signature"
    elif score <= 50:
        label = "MODERATE RISK"
        action = "Address flagged clauses before signing"
    elif score <= 75:
        label = "HIGH RISK"
        action = "Significant negotiation required before signing"
    else:
        label = "CRITICAL RISK"
        action = "Do not sign — escalate to legal immediately"

    return score, label, action

# ── CHECK FOR MISSING CRITICAL CLAUSES ────────────────────
def check_missing_clauses(findings, library):
    """Flag critical categories that are completely absent"""
    critical_categories = [
        "Business Continuity",
        "Capacity Commitment",
        "Termination Rights",
        "Liability Cap",
        "Sterilization Certificate",
        "Validation Protocol"
    ]
    
    found_categories = set(f["category"] for f in findings)
    missing = []
    
    for cat in critical_categories:
        if cat not in found_categories:
            missing.append(cat)
    
    return missing

# ── DISPLAY RESULTS ────────────────────────────────────────
def display_results(findings, score, label, action, missing):
    """Print full detection results"""
    
    print("\n" + "="*65)
    print("  CONTRACT RISK ANALYSIS REPORT")
    print("="*65)
    
    # Risk score
    print(f"\n  OVERALL RISK SCORE: {score}/100  —  {label}")
    print(f"  RECOMMENDED ACTION: {action}")
    
    # Summary counts
    risky = [f for f in findings if f["risk_level"] == "Risky"]
    acceptable = [f for f in findings if f["risk_level"] == "Acceptable"]
    preferred = [f for f in findings if f["risk_level"] == "Preferred"]
    
    print(f"\n  FINDINGS SUMMARY:")
    print(f"    Risky clauses found:      {len(risky)}")
    print(f"    Acceptable clauses found: {len(acceptable)}")
    print(f"    Preferred clauses found:  {len(preferred)}")
    print(f"    Missing critical clauses: {len(missing)}")

    # Missing clauses
    if missing:
        print(f"\n{'='*65}")
        print("  MISSING CRITICAL CLAUSES")
        print(f"{'='*65}")
        for cat in missing:
            print(f"\n  WARNING: No {cat} clause detected in this contract.")
            print(f"  This clause is required for sterilization agreements.")

    # Risky findings detail
    if risky:
        print(f"\n{'='*65}")
        print("  RISKY CLAUSES — REQUIRE NEGOTIATION")
        print(f"{'='*65}")
        for i, f in enumerate(risky, 1):
            print(f"\n  [{i}] CATEGORY: {f['category']}")
            print(f"      TRIGGER:  '{f['trigger_found']}'")
            print(f"      FOUND:    {f['sentence']}")

    # Acceptable findings
    if acceptable:
        print(f"\n{'='*65}")
        print("  ACCEPTABLE CLAUSES — MONITOR BUT OK TO PROCEED")
        print(f"{'='*65}")
        for i, f in enumerate(acceptable, 1):
            print(f"\n  [{i}] CATEGORY: {f['category']}")
            print(f"      FOUND:    {f['sentence']}")

    print(f"\n{'='*65}")
    print("  END OF REPORT")
    print(f"{'='*65}\n")

# ── SAVE RESULTS TO CSV ────────────────────────────────────
def save_results(findings, missing, score, label, output_file="detection_results.csv"):
    """Save all findings to CSV for further review"""
    with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Type", "Category", "Risk_Level", 
                         "Trigger_Found", "Contract_Sentence"])
        
        for finding in findings:
            writer.writerow([
                "FOUND",
                finding["category"],
                finding["risk_level"],
                finding["trigger_found"],
                finding["sentence"]
            ])
        
        for cat in missing:
            writer.writerow([
                "MISSING",
                cat,
                "Risky",
                "N/A - Clause not found in contract",
                "N/A"
            ])
    
    print(f"Results saved to {output_file}")

# ══════════════════════════════════════════════════════════
# MAIN — RUN THE DETECTOR
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    
    # Load clause library
    library = load_clause_library("clause_library_v2.csv")
    
    # Load the sample contract we created in Week 1
    contract_text = load_contract("sample_contract.txt")
    
    # Split into sentences
    sentences = split_sentences(contract_text)
    print(f"Contract sentences to analyze: {len(sentences)}")
    
    # Detect clauses
    findings = detect_clauses(sentences, library)
    
    # Calculate risk score
    score, label, action = calculate_risk_score(findings)
    
    # Check for missing critical clauses
    missing = check_missing_clauses(findings, library)
    
    # Display results
    display_results(findings, score, label, action, missing)
    
    # Save results
    save_results(findings, missing, score, label)
    
    print("Week 3 Complete!")