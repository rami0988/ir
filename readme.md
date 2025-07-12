Information Retrieval System 2025
Project Description
This project implements a custom search engine capable of retrieving documents from two datasets (ANTIQUE and ArgsMe) using various representation methods (TF-IDF, Embedding, Hybrid, BM25) and Retrieval-Augmented Generation (RAG).
Setup Instructions

Clone the repository.
Install dependencies: pip install -r requirements.txt
Ensure the data/ directory contains the processed datasets (docs_antique.csv, etc.).
Run the web application: python src/web_app.py
Access the application at http://localhost:5000.

Directory Structure

data/: Contains dataset files.
src/: Contains Python scripts for preprocessing, indexing, representation, query processing, ranking, and RAG.
static/: Contains CSS files.
templates/: Contains HTML templates.

Features

Data preprocessing (Normalization, Stemming, Lemmatization)
Document representation (TF-IDF, Embedding, Hybrid, BM25)
Inverted Index for efficient retrieval
Query processing and ranking using Cosine Similarity
Web interface using Flask
Retrieval-Augmented Generation (RAG)

