import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import spacy
import os
import tempfile
from io import StringIO

# ── PAGE CONFIG ────────────────────────────────────────────
st.set_page_config(
    page_title="Contract Risk Analyzer",
    page_icon="📋",
    layout="wide"
)

# ── CUSTOM STYLING ─────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .block-container { padding-top: 1.5rem; }
    h1 { color: #1F4E79; font-family: Arial; }
    h2, h3 { color: #2E75B6; font-family: Arial; }
    .section-divider {
        border-top: 2px solid #2E75B6;
        margin: 20px 0px;
    }
    .footer {
        text-align: center;
        color: #888888;
        font-size: 12px;
        margin-top: 40px;
        font-family: Arial;
    }
    .risk-high { color: #C00000; font-weight: bold; }
    .risk-moderate { color: #FF8C00; font-weight: bold; }
    .risk-low { color: #375623; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ── LOAD SPACY ─────────────────────────────────────────────
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

nlp = load_nlp()

# ── CONTRACT TYPE CONFIGURATION ────────────────────────────
CONTRACT_TYPES = {
    "Sterilization Services Agreement": {
        "library": "clause_library_sterilization.csv",
        "context": "sterilization services for regulated manufacturing",
        "critical_missing": [
            "Business Continuity", "Capacity Commitment",
            "Termination Rights", "Liability Cap",
            "Sterilization Certificate", "Validation Protocol",
            "Dose Mapping", "ETO Residual Limits"
        ]
    },
    "MRO / Indirect Spend Agreement": {
        "library": "clause_library_mro.csv",
        "context": "MRO and indirect spend procurement",
        "critical_missing": [
            "Termination Rights", "Liability Cap",
            "Delivery & Lead Time", "Warranty",
            "Inventory & Availability", "Pricing & Catalog"
        ]
    },
    "Supplier MSA (Master Service Agreement)": {
        "library": "clause_library_msa.csv",
        "context": "master service agreements for supplier partnerships",
        "critical_missing": [
            "Termination Rights", "Liability Cap",
            "Confidentiality", "Scope of Services",
            "Service Levels", "Data Protection"
        ]
    },
    "General Procurement Agreement": {
        "library": "clause_library_general.csv",
        "context": "general procurement agreements",
        "critical_missing": [
            "Termination Rights", "Liability Cap",
            "Indemnification", "Price Escalation"
        ]
    }
}

# ── WEIGHTS & SCORING ──────────────────────────────────────
CATEGORY_WEIGHTS = {
    "Auto-Renewal": 9, "Liability Cap": 10,
    "Indemnification": 9, "Payment Terms": 7,
    "Price Escalation": 9, "Termination Rights": 10,
    "Force Majeure": 6, "Intellectual Property": 7,
    "Exclusivity": 8, "Governing Law": 5,
    "Dose Mapping": 9, "Bioburden Management": 8,
    "Sterilization Certificate": 8, "Capacity Commitment": 10,
    "Business Continuity": 10, "Regulatory Change": 8,
    "Validation Protocol": 10, "ETO Residual Limits": 9,
    "Delivery & Lead Time": 9, "Warranty": 8,
    "Inventory & Availability": 8, "Pricing & Catalog": 7,
    "Confidentiality": 9, "Scope of Services": 8,
    "Service Levels": 9, "Data Protection": 9,
}

RISK_LEVEL_MULTIPLIER = {
    "Risky": 1.0, "Acceptable": 0.3, "Preferred": 0.0}

MISSING_CLAUSE_PENALTY = {
    "Business Continuity": 15, "Capacity Commitment": 12,
    "Validation Protocol": 12, "Sterilization Certificate": 10,
    "Dose Mapping": 8, "ETO Residual Limits": 8,
    "Termination Rights": 12, "Liability Cap": 12,
    "Indemnification": 8, "Price Escalation": 7,
    "Delivery & Lead Time": 10, "Warranty": 8,
    "Inventory & Availability": 8, "Pricing & Catalog": 7,
    "Confidentiality": 10, "Scope of Services": 9,
    "Service Levels": 10, "Data Protection": 10,
}

MIN_SENTENCE_LENGTH = {
    "Force Majeure": 60, "Governing Law": 50,
    "Payment Terms": 40, "Auto-Renewal": 50, "default": 25,
}

EXCLUSION_PHRASES = {
    "Force Majeure": ["invoice", "payment", "net 30",
                      "net 60", "net 90"],
    "Payment Terms": ["terminate", "renew",
                      "liability", "arbitration"],
    "Auto-Renewal":  ["invoice", "payment", "net 30"],
    "Governing Law": ["invoice", "payment"],
}

# ── CORE FUNCTIONS ─────────────────────────────────────────
@st.cache_data
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

def split_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents
            if len(sent.text.strip()) > 20]

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
                        "sentence":      sentence,
                        "category":      category,
                        "risk_level":    clause["risk_level"],
                        "trigger_found": trigger,
                    })
                    break
    return findings

def check_missing_clauses(findings, contract_config):
    found_categories = set(f["category"] for f in findings)
    return [cat for cat in contract_config["critical_missing"]
            if cat not in found_categories]

def calculate_risk_score(findings, missing_clauses):
    total_score = 0
    breakdown = {}
    for f in findings:
        weight = CATEGORY_WEIGHTS.get(f["category"], 5)
        mult = RISK_LEVEL_MULTIPLIER.get(f["risk_level"], 0)
        pts = weight * mult
        total_score += pts
        if pts > 0:
            breakdown[f["category"]] = \
                breakdown.get(f["category"], 0) + pts
    for cat in missing_clauses:
        penalty = MISSING_CLAUSE_PENALTY.get(cat, 5)
        total_score += penalty
        breakdown[f"MISSING: {cat}"] = penalty
    max_possible = sum(CATEGORY_WEIGHTS.values()) + \
                   sum(MISSING_CLAUSE_PENALTY.values())
    score = min(100, round((total_score / max_possible) * 100))
    if score <= 25:
        label = "LOW RISK"
        color = "#375623"
        action = "Proceed with standard review and signature"
    elif score <= 50:
        label = "MODERATE RISK"
        color = "#FF8C00"
        action = "Address flagged clauses before signing"
    elif score <= 75:
        label = "HIGH RISK"
        color = "#C00000"
        action = "Significant negotiation required before signing"
    else:
        label = "CRITICAL RISK"
        color = "#7B0000"
        action = "Do not sign — escalate to legal immediately"
    return score, label, color, action, breakdown

# ── HEADER ─────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;'>📋 Contract Risk Analyzer</h1>",
    unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#555; font-family:Arial;'>"
    "AI-Powered Contract Clause Detection & Risk Scoring  |  "
    "Louis Filiano</p>", unsafe_allow_html=True)
st.markdown("<div class='section-divider'></div>",
            unsafe_allow_html=True)

# ── SIDEBAR CONTROLS ───────────────────────────────────────
st.sidebar.markdown("## Contract Settings")

contract_type = st.sidebar.selectbox(
    "Contract Type",
    list(CONTRACT_TYPES.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Contract (.txt)",
    type=["txt"],
    help="Upload a plain text version of your contract"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About This Tool")
st.sidebar.markdown("""
Detects risky clauses across:
- Auto-Renewal & Evergreen terms
- Liability Caps & Indemnification
- Payment Terms & Price Escalation
- Termination Rights & Force Majeure
- Contract-specific categories
""")

# ── MAIN CONTENT ───────────────────────────────────────────
if uploaded_file is not None:

    # Read contract text
    contract_text = uploaded_file.read().decode("utf-8")
    contract_config = CONTRACT_TYPES[contract_type]

    # Load library and analyze
    with st.spinner("Loading clause library..."):
        library = load_clause_library(contract_config["library"])

    with st.spinner("Analyzing contract..."):
        sentences = split_sentences(contract_text)
        findings = detect_clauses(sentences, library)
        missing = check_missing_clauses(findings, contract_config)
        score, label, color, action, breakdown = \
            calculate_risk_score(findings, missing)

    risky =      [f for f in findings if f["risk_level"] == "Risky"]
    acceptable = [f for f in findings if f["risk_level"] == "Acceptable"]

    # ── SECTION 1: EXECUTIVE SUMMARY ──────────────────────
    st.markdown("## Executive Summary")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Contract Type", contract_type.split("(")[0].strip())
    c2.metric("Sentences Analyzed", len(sentences))
    c3.metric("Risky Clauses", len(risky))
    c4.metric("Missing Critical", len(missing))
    c5.metric("Risk Score", f"{score}/100")

    # Risk score gauge
    st.markdown("### Overall Risk Score")
    col_score, col_action = st.columns([1, 2])

    with col_score:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.barh(["Risk Score"], [score],
                color=color, height=0.4)
        ax.barh(["Risk Score"], [100 - score],
                left=[score], color="#EEEEEE", height=0.4)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Score (0-100)")
        ax.set_title(f"{score}/100 — {label}",
                     color=color, fontweight="bold")
        ax.axvline(x=25, color="green",
                   linestyle="--", alpha=0.5, linewidth=1)
        ax.axvline(x=50, color="orange",
                   linestyle="--", alpha=0.5, linewidth=1)
        ax.axvline(x=75, color="red",
                   linestyle="--", alpha=0.5, linewidth=1)
        plt.tight_layout()
        st.pyplot(fig)

    with col_action:
        st.markdown(f"**Risk Level:** "
                    f"<span style='color:{color}'>{label}</span>",
                    unsafe_allow_html=True)
        st.markdown(f"**Recommended Action:** {action}")
        st.markdown("**Score Guide:**")
        st.markdown("🟢 0-25: Low Risk — proceed with review")
        st.markdown("🟡 26-50: Moderate — address before signing")
        st.markdown("🔴 51-75: High — negotiate before signing")
        st.markdown("🚨 76-100: Critical — do not sign")

    st.markdown("<div class='section-divider'></div>",
                unsafe_allow_html=True)

    # ── SECTION 2: SCORE BREAKDOWN ─────────────────────────
    st.markdown("## Risk Score Breakdown")
    sorted_bd = sorted(breakdown.items(),
                       key=lambda x: x[1], reverse=True)
    bd_df = pd.DataFrame(sorted_bd,
                         columns=["Category", "Risk Points"])
    bd_df["Type"] = bd_df["Category"].apply(
        lambda x: "Missing Clause" if "MISSING" in x else "Found Clause")

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    colors_bd = ["#C00000" if "MISSING" in cat else "#2E75B6"
                 for cat, _ in sorted_bd]
    ax2.barh([x[0] for x in sorted_bd],
             [x[1] for x in sorted_bd],
             color=colors_bd)
    ax2.set_xlabel("Risk Points")
    ax2.set_title(
        "Risk Contribution by Category  "
        "(Red = Missing Clause  Blue = Found Clause)")
    plt.tight_layout()
    st.pyplot(fig2)

    st.markdown("<div class='section-divider'></div>",
                unsafe_allow_html=True)

    # ── SECTION 3: RISKY CLAUSES ───────────────────────────
    st.markdown("## Risky Clauses Found")

    if len(risky) > 0:
        v1, v2 = st.columns(2)
        v1.metric("Total Risky Clauses", len(risky))
        v2.metric("Categories Affected",
                  len(set(f["category"] for f in risky)))

        for i, f in enumerate(risky, 1):
            with st.expander(
                f"[{i}] {f['category']} — "
                f"Trigger: '{f['trigger_found']}'"):
                st.markdown("**Original Clause:**")
                st.warning(f['sentence'])
                st.markdown(
                    f"**Risk Category:** {f['category']}  |  "
                    f"**Weight:** "
                    f"{CATEGORY_WEIGHTS.get(f['category'], 5)}/10")
                st.info(
                    "Run locally with AI suggestions enabled "
                    "to see Claude-generated improved language "
                    "for this clause.")
    else:
        st.success("No risky clauses detected!")

    st.markdown("<div class='section-divider'></div>",
                unsafe_allow_html=True)

    # ── SECTION 4: MISSING CLAUSES ─────────────────────────
    st.markdown("## Missing Critical Clauses")

    if len(missing) > 0:
        st.error(
            f"{len(missing)} critical clauses are missing "
            f"from this contract.")
        for cat in missing:
            penalty = MISSING_CLAUSE_PENALTY.get(cat, 5)
            with st.expander(
                f"MISSING: {cat} "
                f"(+{penalty} risk points)"):
                st.markdown(
                    f"**{cat}** was not detected in this contract.")
                st.markdown(
                    f"This clause is critical for "
                    f"**{contract_type}** agreements.")
                st.info(
                    "Run locally with AI suggestions enabled "
                    "to generate recommended clause language.")
    else:
        st.success("All critical clauses detected!")

    st.markdown("<div class='section-divider'></div>",
                unsafe_allow_html=True)

    # ── SECTION 5: ACCEPTABLE CLAUSES ─────────────────────
    if len(acceptable) > 0:
        st.markdown("## Acceptable Clauses — Monitor")
        for i, f in enumerate(acceptable, 1):
            with st.expander(f"[{i}] {f['category']}"):
                st.markdown("**Clause:**")
                st.success(f['sentence'])
                st.markdown(
                    "This clause is acceptable but could "
                    "be improved in future negotiations.")

        st.markdown("<div class='section-divider'></div>",
                    unsafe_allow_html=True)

    # ── SECTION 6: DOWNLOAD ────────────────────────────────
    st.markdown("## Download Results")

    output_rows = []
    for f in risky:
        output_rows.append({
            "Type": "RISKY CLAUSE",
            "Category": f["category"],
            "Risk Level": f["risk_level"],
            "Trigger Found": f["trigger_found"],
            "Contract Sentence": f["sentence"],
            "AI Suggestion": "Run locally for AI suggestions"
        })
    for cat in missing:
        output_rows.append({
            "Type": "MISSING CLAUSE",
            "Category": cat,
            "Risk Level": "Critical",
            "Trigger Found": "N/A",
            "Contract Sentence": "Clause not found in contract",
            "AI Suggestion": "Run locally for AI suggestions"
        })
    for f in acceptable:
        output_rows.append({
            "Type": "ACCEPTABLE CLAUSE",
            "Category": f["category"],
            "Risk Level": f["risk_level"],
            "Trigger Found": f["trigger_found"],
            "Contract Sentence": f["sentence"],
            "AI Suggestion": "N/A"
        })

    if output_rows:
        output_df = pd.DataFrame(output_rows)
        csv_out = output_df.to_csv(index=False,
                                   encoding="utf-8-sig")
        st.download_button(
            label="Download Contract Analysis as CSV",
            data=csv_out,
            file_name="contract_risk_analysis.csv",
            mime="text/csv"
        )

    # FOOTER
    st.markdown(
        "<div class='footer'>Contract Risk Analyzer  |  "
        "Louis Filiano  |  "
        "Powered by Python, spaCy & Scikit-Learn</div>",
        unsafe_allow_html=True)

else:
    st.info(
        "Upload a contract text file using the sidebar "
        "to begin analysis.")
    st.markdown("### This tool detects:")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Standard Procurement Risks:**
        - Auto-renewal and evergreen clauses
        - Inadequate liability caps
        - One-sided indemnification
        - Unfavorable payment terms
        - Uncapped price escalation
        - Weak termination rights
        - Broad force majeure clauses
        - IP ownership issues
        - Exclusivity traps
        - Unfavorable governing law
        """)
    with col2:
        st.markdown("""
        **Sterilization-Specific Risks:**
        - Missing Business Continuity clause
        - No Capacity Commitment
        - Missing Validation Protocol approval rights
        - No Sterilization Certificate requirements
        - Missing Dose Mapping obligations
        - No ETO Residual Limit guarantees
        - Bioburden management gaps
        - Regulatory Change cost allocation

        **MRO & MSA Risks also detected**
        """)