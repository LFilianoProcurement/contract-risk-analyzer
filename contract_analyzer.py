import os
import csv
import spacy
from dotenv import load_dotenv
import anthropic

# Load API key
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

# Load spaCy
nlp = spacy.load("en_core_web_sm")

# ── CONTRACT TYPE CONFIGURATION ────────────────────────────
CONTRACT_TYPES = {
    "1": {
        "name": "Sterilization Services Agreement",
        "library": "clause_library_sterilization.csv",
        "context": "sterilization services for regulated manufacturing including medical devices and pharmaceuticals",
        "critical_missing": [
            "Business Continuity", "Capacity Commitment",
            "Termination Rights", "Liability Cap",
            "Sterilization Certificate", "Validation Protocol",
            "Dose Mapping", "ETO Residual Limits"
        ]
    },
    "2": {
        "name": "MRO / Indirect Spend Agreement",
        "library": "clause_library_mro.csv",
        "context": "MRO and indirect spend procurement for manufacturing operations",
        "critical_missing": [
            "Termination Rights", "Liability Cap",
            "Delivery & Lead Time", "Warranty",
            "Inventory & Availability", "Pricing & Catalog"
        ]
    },
    "3": {
        "name": "Supplier MSA (Master Service Agreement)",
        "library": "clause_library_msa.csv",
        "context": "master service agreements for supplier partnerships in regulated manufacturing",
        "critical_missing": [
            "Termination Rights", "Liability Cap",
            "Confidentiality", "Scope of Services",
            "Service Levels", "Data Protection"
        ]
    },
    "4": {
        "name": "General Procurement Agreement",
        "library": "clause_library_general.csv",
        "context": "general procurement agreements for regulated manufacturing",
        "critical_missing": [
            "Termination Rights", "Liability Cap",
            "Indemnification", "Price Escalation"
        ]
    }
}

# ── CATEGORY WEIGHTS ───────────────────────────────────────
CATEGORY_WEIGHTS = {
    # Standard
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
    # Sterilization
    "Dose Mapping":              9,
    "Bioburden Management":      8,
    "Sterilization Certificate": 8,
    "Capacity Commitment":       10,
    "Business Continuity":       10,
    "Regulatory Change":         8,
    "Validation Protocol":       10,
    "ETO Residual Limits":       9,
    # MRO
    "Delivery & Lead Time":      9,
    "Warranty":                  8,
    "Inventory & Availability":  8,
    "Pricing & Catalog":         7,
    # MSA
    "Confidentiality":           9,
    "Scope of Services":         8,
    "Service Levels":            9,
    "Data Protection":           9,
}

RISK_LEVEL_MULTIPLIER = {
    "Risky":      1.0,
    "Acceptable": 0.3,
    "Preferred":  0.0,
}

MISSING_CLAUSE_PENALTY = {
    # Sterilization
    "Business Continuity":       15,
    "Capacity Commitment":       12,
    "Validation Protocol":       12,
    "Sterilization Certificate": 10,
    "Dose Mapping":              8,
    "ETO Residual Limits":       8,
    # Standard
    "Termination Rights":        12,
    "Liability Cap":             12,
    "Indemnification":           8,
    "Price Escalation":          7,
    # MRO
    "Delivery & Lead Time":      10,
    "Warranty":                  8,
    "Inventory & Availability":  8,
    "Pricing & Catalog":         7,
    # MSA
    "Confidentiality":           10,
    "Scope of Services":         9,
    "Service Levels":            10,
    "Data Protection":           10,
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

# ── SELECT CONTRACT TYPE ───────────────────────────────────
def select_contract_type():
    print("\n" + "="*65)
    print("  CONTRACT RISK ANALYZER")
    print("  Powered by AI  |  Louis Filiano")
    print("="*65)
    print("\n  Select the contract type to analyze:\n")
    for key, val in CONTRACT_TYPES.items():
        print(f"  [{key}]  {val['name']}")
    print()
    
    while True:
        choice = input("  Enter your choice (1-4): ").strip()
        if choice in CONTRACT_TYPES:
            selected = CONTRACT_TYPES[choice]
            print(f"\n  Selected: {selected['name']}")
            print(f"  Library:  {selected['library']}")
            return selected
        print("  Invalid choice. Please enter 1, 2, 3, or 4.")

# ── LOAD CLAUSE LIBRARY ────────────────────────────────────
def load_clause_library(filepath):
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

# ── CHECK MISSING CLAUSES ──────────────────────────────────
def check_missing_clauses(findings, contract_config):
    found_categories = set(f["category"] for f in findings)
    return [cat for cat in contract_config["critical_missing"]
            if cat not in found_categories]

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

# ── GET AI SUGGESTION ──────────────────────────────────────
def get_suggestion(finding, context):
    if not api_key:
        return "API key not configured"
    client = anthropic.Anthropic(api_key=api_key)
    prompt = f"""You are an expert procurement attorney specializing in 
{context} agreements for regulated manufacturing environments.

A contract clause has been flagged as RISKY in the category: 
{finding['category']}

The risky clause reads:
"{finding['sentence']}"

The specific risk trigger identified was: "{finding['trigger_found']}"

Please provide:
1. A brief explanation (2 sentences) of why this clause is risky
2. Suggested improved clause language that protects the buyer

Format exactly like this:
RISK EXPLANATION: [your explanation]
SUGGESTED LANGUAGE: [your improved clause]"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

# ── GET MISSING CLAUSE SUGGESTION ─────────────────────────
def get_missing_suggestion(category, context):
    if not api_key:
        return "API key not configured"
    client = anthropic.Anthropic(api_key=api_key)
    prompt = f"""You are an expert procurement attorney specializing in 
{context} agreements for regulated manufacturing environments.

A critical clause is MISSING from this contract: {category}

Provide recommended contract language that protects the buying 
organization. Format exactly like this:
WHY IT MATTERS: [2 sentence explanation]
RECOMMENDED LANGUAGE: [the actual clause language to add]"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

# ── DISPLAY RESULTS ────────────────────────────────────────
def display_results(findings, score, label, action,
                    missing, contract_config, use_ai=True):

    risky =      [f for f in findings if f["risk_level"] == "Risky"]
    acceptable = [f for f in findings if f["risk_level"] == "Acceptable"]
    context = contract_config["context"]

    print("\n" + "="*65)
    print(f"  CONTRACT RISK ANALYSIS REPORT")
    print(f"  Contract Type: {contract_config['name']}")
    print("="*65)

    bar_filled = int(score / 5)
    bar = "█" * bar_filled + "░" * (20 - bar_filled)
    print(f"\n  RISK SCORE:  {score}/100  [{bar}]")
    print(f"  RISK LEVEL:  {label}")
    print(f"  ACTION:      {action}")
    print(f"\n  Risky clauses:      {len(risky)}")
    print(f"  Acceptable clauses: {len(acceptable)}")
    print(f"  Missing critical:   {len(missing)}")

    # Risky clauses with AI suggestions
    if risky:
        print(f"\n{'='*65}")
        print("  RISKY CLAUSES — WITH AI SUGGESTED IMPROVEMENTS")
        print(f"{'='*65}")
        for i, f in enumerate(risky, 1):
            print(f"\n  [{i}] CATEGORY: {f['category']}")
            print(f"  {'─'*60}")
            print(f"  ORIGINAL:")
            print(f"  {f['sentence']}")
            if use_ai:
                print(f"\n  Getting AI suggestion...")
                suggestion = get_suggestion(f, context)
                print(f"\n  {suggestion}")
            print(f"  {'─'*60}")

    # Missing clauses
    if missing:
        print(f"\n{'='*65}")
        print("  MISSING CRITICAL CLAUSES")
        print(f"{'='*65}")
        for cat in missing:
            print(f"\n  MISSING: {cat}")
            print(f"  {'─'*60}")
            if use_ai:
                print(f"  Getting recommended language...")
                suggestion = get_missing_suggestion(cat, context)
                print(f"\n  {suggestion}")
            print(f"  {'─'*60}")

    # Acceptable clauses
    if acceptable:
        print(f"\n{'='*65}")
        print("  ACCEPTABLE CLAUSES — MONITOR")
        print(f"{'='*65}")
        for i, f in enumerate(acceptable, 1):
            print(f"\n  [{i}] {f['category']}")
            print(f"      {f['sentence'][:100]}...")

    print(f"\n{'='*65}")
    print("  END OF REPORT")
    print(f"{'='*65}\n")

# ── SAVE RESULTS ───────────────────────────────────────────
def save_results(findings, missing, score, label,
                 contract_config, use_ai=True,
                 output_file="contract_analysis_results.csv"):
    risky = [f for f in findings if f["risk_level"] == "Risky"]
    context = contract_config["context"]

    with open(output_file, "w", newline="",
              encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Type", "Category", "Risk_Level",
            "Original_Clause", "AI_Suggestion"
        ])
        for finding in risky:
            suggestion = get_suggestion(
                finding, context) if use_ai else "N/A"
            writer.writerow([
                "RISKY CLAUSE",
                finding["category"],
                finding["risk_level"],
                finding["sentence"],
                suggestion
            ])
        for cat in missing:
            suggestion = get_missing_suggestion(
                cat, context) if use_ai else "N/A"
            writer.writerow([
                "MISSING CLAUSE", cat, "Critical",
                "Clause not found in contract",
                suggestion
            ])
    print(f"Results saved to {output_file}")

# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":

    # Select contract type
    contract_config = select_contract_type()

    # Get contract file
    print("\n  Enter the contract filename to analyze.")
    print("  Press Enter to use sample_contract.txt: ")
    filename = input("  Filename: ").strip()
    if not filename:
        filename = "sample_contract.txt"

    # Ask about AI suggestions
    print("\n  Use AI suggestions? (uses API credits)")
    ai_choice = input("  Enter Y for yes, N for no: ").strip().upper()
    use_ai = ai_choice == "Y"

    print(f"\n  Loading {filename}...")
    library = load_clause_library(contract_config["library"])
    print(f"  Loaded {sum(len(v) for v in library.values())} "
          f"clauses across {len(library)} categories")

    contract_text = load_contract(filename)
    sentences = split_sentences(contract_text)
    print(f"  Analyzing {len(sentences)} sentences...")

    findings = detect_clauses(sentences, library)
    missing = check_missing_clauses(findings, contract_config)
    score, label, action = calculate_risk_score(findings, missing)

    display_results(findings, score, label, action,
                    missing, contract_config, use_ai)

    print("  Saving results...")
    save_results(findings, missing, score, label,
                 contract_config, use_ai)

    print("\n  Analysis complete!")