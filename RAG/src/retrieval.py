# src/retrieval.py
import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.Client()
collection = client.get_collection(name="medquad_collection")
model = SentenceTransformer('emilyalsentzer/Bio_ClinicalBERT')

def retrieve_evidence(query: str, k=3):
    embedding = model.encode(query)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=k
    )
    return results['documents']
