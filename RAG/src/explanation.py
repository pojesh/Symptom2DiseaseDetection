# src/explanation.py
import google.generativeai as palm

palm.configure(api_key='api key')

def generate_explanation(symptoms, diseases, medical_evidence):
    prompt = (
        f"Given the symptoms: {', '.join(symptoms)}.\n"
        f"Top predicted diseases: {', '.join(diseases)}.\n"
        "Relevant clinical evidence:\n" +
        "\n---\n".join(medical_evidence) +
        "\n\nProvide a concise user-facing medical explanation, referencing retrieved evidence."
    )
    response = palm.chat(
        model="models/gemini-2.5-flash",  # or the specific model endpoint
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.text
