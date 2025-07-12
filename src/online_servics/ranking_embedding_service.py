import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import requests

def rank_documents_embedding(query_vector, dataset_name, use_vector_store=False):
    try:
        ranked_docs = []

        # Load document mapping from database
        db_path = f"data/{dataset_name}/index.db"
        if not os.path.exists(db_path):
            return {"status": "error", "message": "Database not found"}
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT doc_id, text FROM documents")
        doc_mapping = {i: (doc_id, text) for i, (doc_id, text) in enumerate(cursor.fetchall())}
        
        # Prepare query vector
        query_vector = np.array(query_vector).reshape(1, -1)

        # Load all embeddings
        embedding_file = f"data/{dataset_name}/embeddings_matrix.joblib"
        if not os.path.exists(embedding_file):
            return {"status": "error", "message": "Embeddings not found"}
        
        embeddings = joblib.load(embedding_file)
        if query_vector.shape[1] != embeddings.shape[1]:
            return {"status": "error", "message": "Dimension mismatch between query vector and embeddings"}

        if use_vector_store:
            # Call Vector Store service to retrieve top documents
            vector_store_url = "http://localhost:8005/vector_search"
            vector_store_data = {
                "query_vector": query_vector.tolist()[0],  # Convert to list for JSON serialization
                "dataset_name": dataset_name
            }
            try:
                vector_store_response = requests.post(vector_store_url, json=vector_store_data)
                vector_store_response.raise_for_status()
                vector_store_data = vector_store_response.json()
                if vector_store_data.get("status") != "success":
                    raise ValueError("Vector Store failed: " + vector_store_data.get("message", "Unknown error"))
                
                vector_store_docs = vector_store_data.get("ranked_docs", [])
                if not vector_store_docs:
                    raise ValueError("No valid documents returned from Vector Store")
                
                # Extract embeddings for the returned documents
                selected_indices = []
                for doc in vector_store_docs:
                    doc_id = doc["doc_id"]
                    for idx, (id_, _) in doc_mapping.items():
                        if id_ == doc_id:
                            selected_indices.append(idx)
                            break
                
                if not selected_indices:
                    raise ValueError("No matching documents found in doc_mapping")
                
                # Select embeddings for the retrieved documents
                selected_embeddings = embeddings[selected_indices]
                
                # Compute cosine similarity with selected embeddings
                similarities = cosine_similarity(query_vector, selected_embeddings)
                doc_indices = np.argsort(similarities[0])[::-1][:10]  # Get top 10 documents
                
                # Build ranked documents list
                for idx in doc_indices:
                    original_idx = selected_indices[idx]
                    if original_idx in doc_mapping:
                        doc_id, text = doc_mapping[original_idx]
                        ranked_docs.append({
                            "doc_id": doc_id,
                            "score": float(similarities[0][idx]),
                            "text": text
                        })
                
            except (requests.RequestException, ValueError) as e:
                print(f"Vector Store error: {str(e)}")
                return {"status": "error", "message": f"Vector Store failed: {str(e)}"}
        
        else:
            # Compute cosine similarity with all embeddings
            similarities = cosine_similarity(query_vector, embeddings)
            doc_indices = np.argsort(similarities[0])[::-1][:10]  # Get top 10 documents
            
            # Build ranked documents list
            for idx in doc_indices:
                if idx in doc_mapping:
                    doc_id, text = doc_mapping[idx]
                    ranked_docs.append({
                        "doc_id": doc_id,
                        "score": float(similarities[0][idx]),
                        "text": text
                    })

        if not ranked_docs:
            return {"status": "error", "message": "No valid documents found"}
        
        return {"status": "success", "ranked_docs": ranked_docs}
    
    except Exception as e:
        print(f"Ranking error: {str(e)}")
        if 'conn' in locals():
            conn.close()
        return {"status": "error", "message": str(e)}

class EmbeddingRankingHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/rank_embedding':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode('utf-8'))
            except json.JSONDecodeError as e:
                self._send_error(400, f"Invalid JSON: {str(e)}")
                return

            query_vector = data.get('query_vector')
            dataset_name = data.get('dataset_name')
            use_vector_store = data.get('use_vector_store', False)

            if not query_vector or not dataset_name:
                self._send_error(400, "query_vector and dataset_name are required")
                return
            
            # Handle hybrid query vector
            vector_to_use = query_vector
            if isinstance(query_vector, dict) and 'embedding' in query_vector:
                vector_to_use = query_vector['embedding']
                print(f"Using embedding part of hybrid query vector: {vector_to_use[:10]}...")

            # Call ranking function
            response = rank_documents_embedding(vector_to_use, dataset_name, use_vector_store)
            self._send_response(response)
        
        else:
            self.send_response(404)
            self.end_headers()

    def _send_response(self, response):
        self.send_response(200)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode('utf-8'))

    def _send_error(self, status_code, message):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(json.dumps({"status": "error", "message": message}).encode('utf-8'))

if __name__ == "__main__":
    server_address = ('', 8007)
    httpd = HTTPServer(server_address, EmbeddingRankingHandler)
    print("Embedding Ranking Service running on port 8007...")
    httpd.serve_forever()