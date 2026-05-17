import csv
import spacy
import os

# Load spaCy language model
nlp = spacy.load("en_core_web_sm")

# ── RISK WEIGHTS PER CATEGORY ──────────────────────────────
# Higher weight = more critical to flag
CATEGORY_WEIGHTS = {
    # Standard procurement — high commercial risk
    "Auto-Renewal":         9,
    "Liability Cap":        10,
    "Indemnification":      9,
    "Payment Terms":        7,
    "Price Escalation":     9,
    "Termination Rights":   10,
    "Force Majeure":        6,
    "Intellectual Property":7,
    "Exclusivity":          8,
    "Governing Law":        5,
    # Sterilization-specific — critical for regulated manufacturing
    "Dose Mapping":         9,
    "Bioburden Management": 8,
    "Sterilization Certificate": 8,
    "Capacity Commitment":  10,
    "Business Continuity":  10,
    "Regulatory Change":    8,
    "Validation Protocol":  10,
    "ETO Residual Limits":  9,
}

RISK_LEVEL_MULTIPLIER = {
    "Risky":      1.0,
    "Acceptable": 0.3,
    "Preferred":  0.0,
}

# ── MINIMUM SENTENCE LENGTH PER CATEGORY ──────────────────
# Prevents short sentences from triggering false positives
MIN_SENTENCE_LENGTH = {
    "Force Majeure":    60,
    "Governing Law":    50,
    "Payment Terms":    40,
    "Auto-Renewal":     50,
    "default":          25,
}

# ── EXCLUSION PHRASES ──────────────────────────────────────
# If these phrases appear in a sentence, skip the match
EXCLUSION_PHRASES = {
    "Force Majeure":  ["invoice", "payment", "net 30", "net 60", "net 90"],
    "Payment Terms":  ["terminate", "renew", "liability", "arbitration"],
    "Auto-Renewal":   ["invoice", "payment", "net 30"],
    "Governing Law":  ["invoice", "payment"],
}

# ── MISSING CLAUSE PENALTIES ───────────────────────────────
MISSING_CLAUSE_PENALTY = {
    "Business Continuity":       15,
    "Capacity Commitment":       12,
    "Termination Rights":        12,
    "Liability Cap":             12,
    "Sterilization Certificate": 10,
    "Validation Protocol":       12,
    "Dose Mapping":              8,
    "ETO Residual Limits":       8,
}

# ── LOAD CLAUSE LIBRARY ────────────────────────────────────
def load_clause_library(filepath="clause_library_v2.csv"):
    library = {}
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = row["Risk_Category"]
            if category not in library:
                library[category] = []
            library[category].append({
                "risk_level":     row["Risk_Level"],
                "trigger_phrase": row["Trigger_Phrase"].lower(),
                "example":        row["Example_Clause"]
            })
    return library

# ── LOAD CONTRACT ──────────────────────────────────────────
def load_contract(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    return " ".join(lines)

# ── SPLIT SENTENCES ────────────────────────────────────────
def split_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents
            if len(sent.text.strip()) > 20]

# ── IMPROVED CLAUSE DETECTION ──────────────────────────────
def detect_clauses(sentences, library):
    findings = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        
        for category, clauses in library.items():
            
            # Check minimum sentence length for this category
            min_len = MIN_SENTENCE_LENGTH.get(
                category, MIN_SENTENCE_LENGTH["default"])
            if len(sentence) < min_len:
                continue
            
            # Check exclusion phrases
            exclusions = EXCLUSION_PHRASES.get(category, [])
            if any(excl in sentence_lower for excl in exclusions):
                continue
            
            for clause in clauses:
                trigger = clause["trigger_phrase"]
                
                if trigger in sentence_lower:
                    findings.append({
                        "sentence":      sentence,
                        "category":      category,
                        "risk_level":    clause["risk_level"],
                        "trigger_found": trigger,
                        "example_clause": clause["example"]
                    })
                    break

    return findings

# ── IMPROVED RISK SCORING ──────────────────────────────────
def calculate_risk_score(findings, missing_clauses):
    total_score = 0
    
    # Score each finding
    for f in findings:
        weight = CATEGORY_WEIGHTS.get(f["category"], 5)
        multiplier = RISK_LEVEL_MULTIPLIER.get(f["risk_level"], 0)
        total_score += weight * multiplier
    
    # Add penalties for missing critical clauses
    missing_penalty = sum(
        MISSING_CLAUSE_PENALTY.get(cat, 5) 
        for cat in missing_clauses
    )
    total_score += missing_penalty
    
    # Normalize to 0-100
    max_possible = sum(CATEGORY_WEIGHTS.values()) + \
                   sum(MISSING_CLAUSE_PENALTY.values())
    score = min(100, round((total_score / max_possible) * 100))
    
    # Breakdown by category
    breakdown = {}
    for f in findings:
        cat = f["category"]
        weight = CATEGORY_WEIGHTS.get(cat, 5)
        multiplier = RISK_LEVEL_MULTIPLIER.get(f["risk_level"], 0)
        breakdown[cat] = breakdown.get(cat, 0) + (weight * multiplier)
    
    for cat in missing_clauses:
        breakdown[f"MISSING: {cat}"] = MISSING_CLAUSE_PENALTY.get(cat, 5)
    
    # Risk label
    if score <= 25:
        label = "LOW RISK"
        color = "GREEN"
        action = "Proceed with standard review and signature"
    elif score <= 50:
        label = "MODERATE RISK"
        color = "YELLOW"
        action = "Address flagged clauses before signing"
    elif score <= 75:
        label = "HIGH RISK"
        color = "ORANGE"
        action = "Significant negotiation required before signing"
    else:
        label = "CRITICAL RISK"
        color = "RED"
        action = "Do not sign — escalate to legal immediately"
    
    return score, label, action, breakdown

# ── CHECK MISSING CLAUSES ──────────────────────────────────
def check_missing_clauses(findings, library):
    critical = list(MISSING_CLAUSE_PENALTY.keys())
    found_categories = set(f["category"] for f in findings)
    return [cat for cat in critical if cat not in found_categories]

# ── DISPLAY RESULTS ────────────────────────────────────────
def display_results(findings, score, label, action, 
                    breakdown, missing):
    
    print("\n" + "="*65)
    print("  CONTRACT RISK ANALYSIS REPORT  —  Version 2")
    print("="*65)
    
    # Risk score
    bar_filled = int(score / 5)
    bar = "█" * bar_filled + "░" * (20 - bar_filled)
    print(f"\n  RISK SCORE:  {score}/100  [{bar}]")
    print(f"  RISK LEVEL:  {label}")
    print(f"  ACTION:      {action}")
    
    # Summary
    risky =      [f for f in findings if f["risk_level"] == "Risky"]
    acceptable = [f for f in findings if f["risk_level"] == "Acceptable"]
    preferred =  [f for f in findings if f["risk_level"] == "Preferred"]
    
    print(f"\n  FINDINGS:")
    print(f"    Risky clauses:          {len(risky)}")
    print(f"    Acceptable clauses:     {len(acceptable)}")
    print(f"    Preferred clauses:      {len(preferred)}")
    print(f"    Missing critical:       {len(missing)}")
    
    # Score breakdown
    print(f"\n{'='*65}")
    print("  RISK SCORE BREAKDOWN BY CATEGORY")
    print(f"{'='*65}")
    sorted_breakdown = sorted(
        breakdown.items(), key=lambda x: x[1], reverse=True)
    for cat, points in sorted_breakdown:
        bar_w = int(points / 2)
        print(f"  {cat:<35} {points:>5.1f} pts  {'█' * min(bar_w, 20)}")
    
    # Missing clauses
    if missing:
        print(f"\n{'='*65}")
        print("  MISSING CRITICAL CLAUSES — ADD BEFORE SIGNING")
        print(f"{'='*65}")
        for cat in missing:
            penalty = MISSING_CLAUSE_PENALTY.get(cat, 5)
            print(f"\n  WARNING: No {cat} clause found")
            print(f"  Risk penalty: {penalty} points added to score")
    
    # Risky clauses
    if risky:
        print(f"\n{'='*65}")
        print("  RISKY CLAUSES — NEGOTIATE BEFORE SIGNING")
        print(f"{'='*65}")
        for i, f in enumerate(risky, 1):
            weight = CATEGORY_WEIGHTS.get(f["category"], 5)
            print(f"\n  [{i}] {f['category']}  "
                  f"(Weight: {weight}/10)")
            print(f"      Trigger: '{f['trigger_found']}'")
            print(f"      Found:   {f['sentence'][:120]}...")
    
    # Acceptable clauses
    if acceptable:
        print(f"\n{'='*65}")
        print("  ACCEPTABLE CLAUSES — MONITOR")
        print(f"{'='*65}")
        for i, f in enumerate(acceptable, 1):
            print(f"\n  [{i}] {f['category']}")
            print(f"      Found: {f['sentence'][:120]}...")
    
    print(f"\n{'='*65}")
    print("  END OF REPORT")
    print(f"{'='*65}\n")

# ── SAVE RESULTS ───────────────────────────────────────────
def save_results(findings, missing, score, label,
                 output_file="risk_analysis_results.csv"):
    with open(output_file, "w", newline="", 
              encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Type", "Category", "Risk_Level",
            "Weight", "Trigger_Found", "Contract_Sentence"
        ])
        for finding in findings:
            writer.writerow([
                "FOUND",
                finding["category"],
                finding["risk_level"],
                CATEGORY_WEIGHTS.get(finding["category"], 5),
                finding["trigger_found"],
                finding["sentence"]
            ])
        for cat in missing:
            writer.writerow([
                "MISSING",
                cat,
                "Critical",
                MISSING_CLAUSE_PENALTY.get(cat, 5),
                "N/A",
                "Clause not found in contract"
            ])
    print(f"Results saved to {output_file}")

# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    
    print("Loading clause library...")
    library = load_clause_library("clause_library_v2.csv")
    print(f"Loaded {sum(len(v) for v in library.values())} "
          f"clauses across {len(library)} categories")
    
    print("\nLoading contract...")
    contract_text = load_contract("sample_contract.txt")
    sentences = split_sentences(contract_text)
    print(f"Contract sentences to analyze: {len(sentences)}")
    
    print("\nAnalyzing contract...")
    findings = detect_clauses(sentences, library)
    missing = check_missing_clauses(findings, library)
    score, label, action, breakdown = calculate_risk_score(
        findings, missing)
    
    display_results(findings, score, label, action, 
                    breakdown, missing)
    save_results(findings, missing, score, label)
    
    print("Week 4 Complete!")