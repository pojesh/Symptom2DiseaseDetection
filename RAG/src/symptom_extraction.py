# src/symptom_extraction.py
import spacy
nlp = spacy.load("en_ner_bc5cdr_md")  # ensures DISEASE/chemical/condition is extracted

def extract_symptoms(text: str):
    doc = nlp(text)
    return [ent.text.lower() for ent in doc.ents if ent.label_ == "DISEASE"]
