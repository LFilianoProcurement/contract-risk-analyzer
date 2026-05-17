import os
import csv
import spacy
from dotenv import load_dotenv
import anthropic

# Load API key from .env file
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

# Load spaCy
nlp = spacy.load("en_core_web_sm")

# ── CATEGORY WEIGHTS (same as risk_scorer.py) ─────────────
CATEGORY_WEIGHTS = {
    "Auto-Renewal":              9,
    "Liability Cap":             10,
    "Indemnification":           9,
    "Payment Terms":             7,
    "Price Escalation":          9,
    "Termination Rights":        10,
    "Force Majeure":             6,
    "Intellectual Property":     7,
    "Exclusivity":               8,
    "Governing Law":             5,
    "Dose Mapping":              9,
    "Bioburden Management":      8,
    "Sterilization Certificate": 8,
    "Capacity Commitment":       10,
    "Business Continuity":       10,
    "Regulatory Change":         8,
    "Validation Protocol":       10,
    "ETO Residual Limits":       9,
}

RISK_LEVEL_MULTIPLIER = {
    "Risky":      1.0,
    "Acceptable": 0.3,
    "Preferred":  0.0,
}

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

MIN_SENTENCE_LENGTH = {
    "Force Majeure":  60,
    "Governing Law":  50,
    "Payment Terms":  40,
    "Auto-Renewal":   50,
    "default":        25,
}

EXCLUSION_PHRASES = {
    "Force Majeure": ["invoice", "payment", "net 30", "net 60", "net 90"],
    "Payment Terms": ["terminate", "renew", "liability", "arbitration"],
    "Auto-Renewal":  ["invoice", "payment", "net 30"],
    "Governing Law": ["invoice", "payment"],
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

# ── DETECT CLAUSES ─────────────────────────────────────────
def detect_clauses(sentences, library):
    findings = []
    for sentence in sentences:
        sentence_lower = sentence.lower()
        for category, clauses in library.items():
            min_len = MIN_SENTENCE_LENGTH.get(
                category, MIN_SENTENCE_LENGTH["default"])
            if len(sentence) < min_len:
                continue
            exclusions = EXCLUSION_PHRASES.get(category, [])
            if any(excl in sentence_lower for excl in exclusions):
                continue
            for clause in clauses:
                trigger = clause["trigger_phrase"]
                if trigger in sentence_lower:
                    findings.append({
                        "sentence":       sentence,
                        "category":       category,
                        "risk_level":     clause["risk_level"],
                        "trigger_found":  trigger,
                        "example_clause": clause["example"]
                    })
                    break
    return findings

# ── CALCULATE RISK SCORE ───────────────────────────────────
def calculate_risk_score(findings, missing_clauses):
    total_score = 0
    for f in findings:
        weight = CATEGORY_WEIGHTS.get(f["category"], 5)
        multiplier = RISK_LEVEL_MULTIPLIER.get(f["risk_level"], 0)
        total_score += weight * multiplier
    missing_penalty = sum(
        MISSING_CLAUSE_PENALTY.get(cat, 5)
        for cat in missing_clauses)
    total_score += missing_penalty
    max_possible = sum(CATEGORY_WEIGHTS.values()) + \
                   sum(MISSING_CLAUSE_PENALTY.values())
    score = min(100, round((total_score / max_possible) * 100))
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

# ── CHECK MISSING CLAUSES ──────────────────────────────────
def check_missing_clauses(findings, library):
    critical = list(MISSING_CLAUSE_PENALTY.keys())
    found_categories = set(f["category"] for f in findings)
    return [cat for cat in critical if cat not in found_categories]

# ── GET AI LANGUAGE SUGGESTION ─────────────────────────────
def get_suggestion(finding, contract_context="sterilization services"):
    """Call Claude API to suggest improved contract language"""
    
    if not api_key:
        return "API key not configured — suggestion unavailable"
    
    client = anthropic.Anthropic(api_key=api_key)
    
    prompt = f"""You are an expert procurement attorney specializing in 
{contract_context} agreements for regulated manufacturing environments 
including medical devices and pharmaceuticals.

A contract clause has been flagged as RISKY in the category: 
{finding['category']}

The risky clause reads:
"{finding['sentence']}"

The specific risk trigger identified was: "{finding['trigger_found']}"

Please provide:
1. A brief explanation (2-3 sentences) of exactly why this clause 
   is risky for the buying organization
2. A suggested improved clause that protects the buyer's interests

Format your response exactly like this:
RISK EXPLANATION: [your explanation here]
SUGGESTED LANGUAGE: [your improved clause here]

Keep the suggested language professional, legally precise, and 
appropriate for a regulated manufacturing environment."""

    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return message.content[0].text

# ── GET MISSING CLAUSE SUGGESTION ─────────────────────────
def get_missing_clause_suggestion(category, 
                                   contract_context="sterilization services"):
    """Get suggested language for a missing critical clause"""
    
    if not api_key:
        return "API key not configured — suggestion unavailable"
    
    client = anthropic.Anthropic(api_key=api_key)
    
    prompt = f"""You are an expert procurement attorney specializing in 
{contract_context} agreements for regulated manufacturing environments.

A critical clause is MISSING from this contract: {category}

Please provide recommended contract language for this clause that 
protects the buying organization in a regulated manufacturing environment.

Format your response exactly like this:
WHY IT MATTERS: [2 sentence explanation of why this clause is critical]
RECOMMENDED LANGUAGE: [the actual clause language to add to the contract]

Keep the language professional, legally precise, and specific to 
{contract_context} in regulated manufacturing."""

    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return message.content[0].text

# ── DISPLAY FULL RESULTS WITH SUGGESTIONS ─────────────────
def display_full_results(findings, score, label, action, missing):
    
    print("\n" + "="*65)
    print("  CONTRACT RISK ANALYZER — WITH AI SUGGESTIONS")
    print("="*65)
    
    bar_filled = int(score / 5)
    bar = "█" * bar_filled + "░" * (20 - bar_filled)
    print(f"\n  RISK SCORE:  {score}/100  [{bar}]")
    print(f"  RISK LEVEL:  {label}")
    print(f"  ACTION:      {action}")
    
    risky = [f for f in findings if f["risk_level"] == "Risky"]
    
    print(f"\n  Risky clauses:      {len(risky)}")
    print(f"  Missing critical:   {len(missing)}")
    
    # Risky clauses with AI suggestions
    if risky:
        print(f"\n{'='*65}")
        print("  RISKY CLAUSES WITH AI-SUGGESTED IMPROVEMENTS")
        print(f"{'='*65}")
        
        for i, f in enumerate(risky, 1):
            print(f"\n  [{i}] CATEGORY: {f['category']}")
            print(f"  {'─'*60}")
            print(f"  ORIGINAL CLAUSE:")
            print(f"  {f['sentence']}")
            print(f"\n  Getting AI suggestion...")
            suggestion = get_suggestion(f)
            print(f"\n  {suggestion}")
            print(f"  {'─'*60}")
    
    # Missing clauses with AI suggestions
    if missing:
        print(f"\n{'='*65}")
        print("  MISSING CLAUSES WITH RECOMMENDED LANGUAGE")
        print(f"{'='*65}")
        
        for cat in missing:
            print(f"\n  MISSING: {cat}")
            print(f"  {'─'*60}")
            print(f"  Getting AI recommendation...")
            suggestion = get_missing_clause_suggestion(cat)
            print(f"\n  {suggestion}")
            print(f"  {'─'*60}")
    
    print(f"\n{'='*65}")
    print("  END OF REPORT")
    print(f"{'='*65}\n")

# ── SAVE FULL RESULTS WITH SUGGESTIONS ────────────────────
def save_full_results(findings, missing, score, label,
                      output_file="full_analysis_with_suggestions.csv"):
    risky = [f for f in findings if f["risk_level"] == "Risky"]
    
    with open(output_file, "w", newline="", 
              encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Type", "Category", "Risk_Level",
            "Original_Clause", "AI_Suggestion"
        ])
        
        for finding in risky:
            print(f"Getting suggestion for {finding['category']}...")
            suggestion = get_suggestion(finding)
            writer.writerow([
                "RISKY CLAUSE",
                finding["category"],
                finding["risk_level"],
                finding["sentence"],
                suggestion
            ])
        
        for cat in missing:
            print(f"Getting missing clause language for {cat}...")
            suggestion = get_missing_clause_suggestion(cat)
            writer.writerow([
                "MISSING CLAUSE",
                cat,
                "Critical",
                "Clause not found in contract",
                suggestion
            ])
    
    print(f"\nFull results with AI suggestions saved to {output_file}")

# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    
    print("Loading clause library...")
    library = load_clause_library("clause_library_v2.csv")
    
    print("Loading contract...")
    contract_text = load_contract("sample_contract.txt")
    sentences = split_sentences(contract_text)
    print(f"Analyzing {len(sentences)} sentences...\n")
    
    # Detect and score
    findings = detect_clauses(sentences, library)
    missing = check_missing_clauses(findings, library)
    score, label, action = calculate_risk_score(findings, missing)
    
    # Display with AI suggestions
    display_full_results(findings, score, label, action, missing)
    
    # Save full results
    print("Saving full results with AI suggestions...")
    save_full_results(findings, missing, score, label)
    
    print("Week 5 Complete!")