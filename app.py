import os
import re
import json
import pdfplumber
import spacy
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer, util
import torch

# Load spaCy and SentenceTransformer models once at startup
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

app = Flask(__name__)

# --- PDF Parsing: Extract Clauses ---
def extract_clauses(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    # Split by Section or Clause headings
    parts = re.split(r'(Section\s*\d+\.?\d*|Clause\s*\d+\.?\d*)', text)
    clauses = []
    i = 0
    while i < len(parts):
        if re.match(r'(Section|Clause)\s*\d+\.?\d*', parts[i]):
            header = parts[i].strip()
            body = parts[i+1].strip() if i + 1 < len(parts) else ""
            clauses.append(f"{header} - {body}")
            i += 2
        else:
            if parts[i].strip():
                clauses.append(parts[i].strip())
            i += 1
    # Filter out short fragments
    return [clause for clause in clauses if len(clause) > 40]

# --- Load all policies on startup ---
policy_files = ['policy1.pdf', 'policy2.pdf']  # Adjust to your files
all_clauses = []
all_clause_refs = []

for pf in policy_files:
    if os.path.exists(pf):
        cls = extract_clauses(pf)
        refs = [f"{pf}::Clause-{i+1}" for i in range(len(cls))]
        all_clauses.extend(cls)
        all_clause_refs.extend(refs)

# Precompute embeddings for fast semantic search
clause_embeddings = embedder.encode(all_clauses, convert_to_tensor=True)

# --- Helper: Convert duration text to days (example: "3 months" -> 90 days) ---
def duration_to_days(duration_str):
    if not duration_str:
        return None
    match_months = re.search(r'(\d+)\s*month', duration_str.lower())
    if match_months:
        return int(match_months.group(1)) * 30
    match_days = re.search(r'(\d+)\s*day', duration_str.lower())
    if match_days:
        return int(match_days.group(1))
    return None

# --- Parse user query ---
def parse_query(query):
    doc = nlp(query)
    age = None
    sex = None
    location = None
    duration = None
    procedure = None

    for ent in doc.ents:
        if ent.label_ == "DATE":
            if "month" in ent.text.lower() or "day" in ent.text.lower():
                duration = ent.text
            else:
                age = ent.text
        elif ent.label_ == "GPE":
            location = ent.text

    sex_match = re.search(r'\b(male|man|m|female|woman|f)\b', query, re.I)
    if sex_match:
        val = sex_match.group(1).lower()
        sex = "male" if val in ['male', 'man', 'm'] else "female"

    proc_match = re.search(
        r'(surgery|operation|hospitalization|therapy|replacement|procedure|transplant)', query, re.I)
    if proc_match:
        procedure = proc_match.group(1).lower()

    return {
        "age": age,
        "sex": sex,
        "location": location,
        "duration": duration,
        "procedure": procedure,
        "duration_days": duration_to_days(duration)
    }

# --- Improved decision logic ---
def decision_logic(clause_text, query_features):
    text = clause_text.lower()
    proc = query_features.get("procedure", "").lower() if query_features.get("procedure") else ""

    coverage_keywords = ['covered', 'included', 'payable', 'benefit']
    exclusion_keywords = ['excluded', 'not covered', 'waiting period', 'except', 'pre-existing']

    # Check for waiting period conditions
    waiting_period_found = False
    waiting_period_days = None
    wp_match = re.search(r'waiting period of (\d+) days', text)
    if wp_match:
        waiting_period_found = True
        waiting_period_days = int(wp_match.group(1))

    # Check exclusions
    if any(word in text for word in exclusion_keywords) and proc in text:
        # If waiting period applies, check user duration
        if waiting_period_found and query_features.get("duration_days") is not None:
            if query_features["duration_days"] < waiting_period_days:
                return (
                    "rejected",
                    "N/A",
                    f"Policy has a waiting period of {waiting_period_days} days which is not completed.",
                )
        return (
            "rejected",
            "N/A",
            f"Procedure '{proc}' is excluded as per the policy clause.",
        )

    # Approve if coverage keywords present and procedure found in the text
    if proc and any(word in text for word in coverage_keywords) and proc in text:
        # Extract amount
        amt_match = re.search(r'(rs\.?\s?[\d,]+)', text)
        if amt_match:
            raw_amt = amt_match.group(0).lower().replace("rs", "Rs").replace(".", "")
            # Clean amount format
            raw_amt = raw_amt.replace(" ", "")
            try:
                # Parse number and format nicely with commas
                num_str = re.sub(r'[^\d]', '', raw_amt)
                amt_int = int(num_str)
                amount = f"Rs {amt_int:,}"
            except:
                amount = raw_amt
        else:
            amount = "Rs 50,000"  # Default fallback
        return (
            "approved",
            amount,
            f"Procedure '{proc}' is covered as per the policy clause.",
        )

    # Fallback: reject if no clear coverage found
    return (
        "rejected",
        "N/A",
        "Procedure coverage unclear or not included in the policy clause. Requires manual review.",
    )

# --- Semantic search utility ---
def semantic_search(user_query):
    query_embedding = embedder.encode(user_query, convert_to_tensor=True)
    sims = util.pytorch_cos_sim(query_embedding, clause_embeddings)[0]
    top_idx = torch.argmax(sims).item()
    return all_clauses[top_idx], all_clause_refs[top_idx]

# --- Flask routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query_api():
    data = request.json
    user_query = data.get("query", "").strip()

    if not user_query:
        return jsonify({"error": "Query is empty"}), 400

    best_clause, clause_ref = semantic_search(user_query)
    query_features = parse_query(user_query)
    decision, amount, justification = decision_logic(best_clause, query_features)

    result = {
        "decision": decision,
        "amount": amount,
        "justification": justification,
        "clause_reference": clause_ref,
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=8000)
