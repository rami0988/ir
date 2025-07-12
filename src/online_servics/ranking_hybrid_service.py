import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import requests
from scipy.sparse import csr_matrix

def rank_documents_hybrid(query_vector, dataset_name, use_vector_store=False):
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
        conn.close()

        # TF-IDF ranking
        tfidf_file = f"data/{dataset_name}/tfidf_matrix.joblib"
        if not os.path.exists(tfidf_file):
            return {"status": "error", "message": "TF-IDF matrix not found"}
        
        tfidf_matrix = joblib.load(tfidf_file)
        query_vector_tfidf = np.array(query_vector['tfidf']).reshape(1, -1)
        if query_vector_tfidf.shape[1] != tfidf_matrix.shape[1]:
            return {"status": "error", "message": "Dimension mismatch between TF-IDF query vector and TF-IDF matrix"}
        
        # query_vector_sparse = csr_matrix(query_vector_tfidf)
        similarities_tfidf = cosine_similarity(query_vector_tfidf, tfidf_matrix)
        top_k = 1000
        doc_indices_tfidf = np.argsort(similarities_tfidf[0])[::-1][:top_k]

        # Embedding ranking
        query_vector_embedding = np.array(query_vector['embedding']).reshape(1, -1)

        if use_vector_store:
            # Call Vector Store service
            vector_store_url = "http://localhost:8005/vector_search"
            vector_store_data = {
                "query_vector": query_vector['embedding'],
                "dataset_name": dataset_name
            }
            try:
                vector_store_response = requests.post(vector_store_url, json=vector_store_data)
                vector_store_response.raise_for_status()
                vector_store_data = vector_store_response.json()
                if vector_store_data.get("status") != "success":
                    raise ValueError("Vector Store search failed: " + vector_store_data.get("message", "Unknown error"))
                
                vector_store_docs = vector_store_data.get("ranked_docs", [])
                if not vector_store_docs:
                    raise ValueError("No valid documents returned from Vector Store")
                
                # Filter vector store results to include only those in top_k TF-IDF documents
                tfidf_doc_ids = {doc_mapping[idx][0] for idx in doc_indices_tfidf}  # Get doc_ids from TF-IDF top_k
                selected_embeddings = []
                selected_indices = []
                for doc in vector_store_docs:
                    doc_id = doc["doc_id"]
                    if doc_id in tfidf_doc_ids:  # Only include documents that are in TF-IDF top_k
                        for idx, (id_, _) in doc_mapping.items():
                            if id_ == doc_id:
                                selected_indices.append(idx)
                                break
                
                # Load all embeddings and select those corresponding to filtered results
                embedding_file = f"data/{dataset_name}/embeddings_matrix.joblib"
                if not os.path.exists(embedding_file):
                    return {"status": "error", "message": "Embeddings not found"}
                embeddings = joblib.load(embedding_file)
                if not selected_indices:
                    raise ValueError("No valid documents found in Vector Store that match TF-IDF top_k")
                selected_embeddings = embeddings[selected_indices]
                
                # Compute cosine similarity with filtered embeddings
                similarities_embedding = cosine_similarity(query_vector_embedding, selected_embeddings)
                doc_indices_embedding = np.argsort(similarities_embedding[0])[::-1][:10]
                
                # Combine results
                for idx in doc_indices_embedding:
                    original_idx = selected_indices[idx]
                    if original_idx < tfidf_matrix.shape[0] and original_idx in doc_mapping:
                        doc_id, text = doc_mapping[original_idx]
                        tfidf_score = similarities_tfidf[0][original_idx] if original_idx in doc_indices_tfidf else 0.0
                        ranked_docs.append({
                            "doc_id": doc_id,
                            "score": float(similarities_embedding[0][idx]),
                            "text": text,
                            "tfidf_score": float(tfidf_score)
                        })
                
            except (requests.RequestException, ValueError) as e:
                print(f"Vector Store error: {str(e)}")
                return {"status": "error", "message": f"Vector Store processing failed: {str(e)}"}
        else:
            # Use local embeddings
            embedding_file = f"data/{dataset_name}/embeddings_matrix.joblib"
            if not os.path.exists(embedding_file):
                return {"status": "error", "message": "Embeddings not found"}
            embeddings = joblib.load(embedding_file)
            selected_embeddings = embeddings[doc_indices_tfidf]
            if query_vector_embedding.shape[1] != embeddings.shape[1]:
                return {"status": "error", "message": "Dimension mismatch between embedding query vector and embeddings"}
            similarities_embedding = cosine_similarity(query_vector_embedding, selected_embeddings)
            doc_indices_embedding = np.argsort(similarities_embedding[0])[::-1][:10]
            
            for idx in doc_indices_embedding:
                original_idx = doc_indices_tfidf[idx]
                if original_idx < tfidf_matrix.shape[0] and original_idx in doc_mapping:
                    doc_id, text = doc_mapping[original_idx]
                    ranked_docs.append({
                        "doc_id": doc_id,
                        "score": float(similarities_embedding[0][idx]),
                        "text": text,
                        "tfidf_score": float(similarities_tfidf[0][original_idx])
                    })

        if not ranked_docs:
            return {"status": "error", "message": "No valid documents found"}
        
        return {"status": "success", "ranked_docs": ranked_docs}
    
    except Exception as e:
        print(f"Ranking error: {str(e)}")
        return {"status": "error", "message": str(e)}

class HybridRankingHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/rank_hybrid':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))

            query_vector = data.get('query_vector')
            dataset_name = data.get('dataset_name')
            use_vector_store = data.get('use_vector_store', False)
            
            if not query_vector or not dataset_name:
                self._send_error(400, "query_vector and dataset_name are required")
                return
            
            response = rank_documents_hybrid(query_vector, dataset_name, use_vector_store)
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
    server_address = ('', 8008)
    httpd = HTTPServer(server_address, HybridRankingHandler)
    print("Hybrid Ranking Service running on port 8008...")
    httpd.serve_forever()