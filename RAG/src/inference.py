# src/inference.py
import joblib
import numpy as np
import pandas as pd
from symptom_extraction import extract_symptoms
from retrieval import retrieve_evidence
from explanation import generate_explanation

# Load models
clf = joblib.load('models/rf_model.joblib')
le = joblib.load('models/label_encoder.joblib')
symptom_columns = joblib.load('models/symptom_columns.joblib')

def main():
    user_nl = input("Describe your symptoms: ")
    extracted = extract_symptoms(user_nl)
    print("Detected symptoms:", extracted)

    # Prepare binary vector
    input_vec = np.zeros(len(symptom_columns), dtype=int)
    for s in extracted:
        if s in symptom_columns:
            idx = symptom_columns.index(s)
            input_vec[idx] = 1

    input_df = pd.DataFrame([input_vec], columns=symptom_columns)
    probas = clf.predict_proba(input_df)[0]
    top3_idx = probas.argsort()[-3:][::-1]
    top3_diseases = le.inverse_transform(top3_idx)
    top3_probs = probas[top3_idx]

    print("\nTop predictions:")
    for d, p in zip(top3_diseases, top3_probs):
        print(f"{d}: {p*100:.2f}% likelihood")

    evidence_query = user_nl + " " + ", ".join(top3_diseases[:1])
    evidence = retrieve_evidence(evidence_query)
    print("\nRelevant medical info:")
    for doc in evidence:
        print(doc)

    # Use Gemini API for user-friendly explanation
    explanation = generate_explanation(extracted, top3_diseases, evidence)
    print("\nGemini Explanation:\n", explanation)

if __name__ == "__main__":
    main()
