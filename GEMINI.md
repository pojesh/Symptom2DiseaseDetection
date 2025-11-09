# Project Overview

This project is a symptom-to-disease prediction tool. It takes a user's description of their symptoms in natural language, predicts the top 3 likely diseases, and provides a user-friendly explanation for the predictions with supporting evidence from a medical knowledge base.

The project is structured as a Python application with two main components:

1.  **Disease Prediction:** A machine learning model (Random Forest) trained on a dataset of symptoms and diseases.
2.  **Retrieval-Augmented Generation (RAG):** A RAG pipeline that uses a vector database (ChromaDB) and a sentence transformer model to retrieve relevant medical information and a large language model (Gemini) to generate a user-friendly explanation.

## Key Technologies

*   **Programming Language:** Python
*   **Machine Learning:** scikit-learn
*   **Natural Language Processing:** spaCy, scispaCy, sentence-transformers
*   **Vector Database:** ChromaDB
*   **Generative AI:** Google Generative AI (Gemini)

# Building and Running

## Prerequisites

*   Python 3.10+
*   Install the required Python packages:

    ```bash
    pip install -r RAG/requirements.txt
    ```

## Running the Application

The main entry point of the application is `RAG/src/inference.py`. To run the application, navigate to the project root directory and execute the following command:

```bash
python RAG/src/inference.py
```

The application will then prompt you to describe your symptoms.

# Development Conventions

*   The project follows a standard Python project structure.
*   The core logic is located in the `RAG/src` directory.
*   Machine learning models are stored in the `RAG/models` directory.
*   Data used for training and retrieval is stored in the `RAG/data` directory.
*   The ChromaDB vector database is stored in the `RAG/data/chroma_db_store` directory.
