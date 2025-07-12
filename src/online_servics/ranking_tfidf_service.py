import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

def rank_documents_tfidf(query_vector, dataset_name, processed_query):
    try:
        ranked_docs = []

        db_path = f"data/{dataset_name}/index.db"
        if not os.path.exists(db_path):
            return {"status": "error", "message": "Database not found"}
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT doc_id, text FROM documents")
        doc_mapping = {i: (doc_id, text) for i, (doc_id, text) in enumerate(cursor.fetchall())}
        
        if not processed_query:
            return {"status": "error", "message": "Processed query terms are required for TF-IDF"}
        
        doc_ids = set()
        for term in processed_query:
            cursor.execute("SELECT doc_id FROM inverted_index WHERE term = ?", (term,))
            doc_ids.update(row[0] for row in cursor.fetchall())
        
        conn.close()
        
        if not doc_ids:
            return {"status": "success", "ranked_docs": []}
        
        filtered_doc_mapping = {}
        filtered_indices = []
        for idx, (doc_id, text) in doc_mapping.items():
            if doc_id in doc_ids:
                filtered_doc_mapping[len(filtered_doc_mapping)] = (doc_id, text)
                filtered_indices.append(idx)
        
        tfidf_file = f"data/{dataset_name}/tfidf_matrix.joblib"
        if not os.path.exists(tfidf_file):
            return {"status": "error", "message": "TF-IDF matrix not found"}
        
        tfidf_matrix = joblib.load(tfidf_file)
        query_vector = np.array(query_vector).reshape(1, -1)
        if query_vector.shape[1] != tfidf_matrix.shape[1]:
            return {"status": "error", "message": "Dimension mismatch between query vector and TF-IDF matrix"}
        
        filtered_tfidf_matrix = tfidf_matrix[filtered_indices]
        if filtered_tfidf_matrix.shape[0] == 0:
            return {"status": "success", "ranked_docs": []}
        
        similarities = cosine_similarity(query_vector, filtered_tfidf_matrix)
        doc_indices = np.argsort(similarities[0])[::-1][:10]
        
        for idx in doc_indices:
            if idx < len(filtered_doc_mapping):
                doc_id, text = filtered_doc_mapping[idx]
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

class TfidfRankingHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/rank_tfidf':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))

            query_vector = data.get('query_vector')
            dataset_name = data.get('dataset_name')
            processed_query = data.get('processed_query', None)
            
            if not query_vector or not dataset_name:
                self._send_error(400, "query_vector and dataset_name are required")
                return
            
            response = rank_documents_tfidf(query_vector, dataset_name, processed_query)
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
    server_address = ('', 8006)
    httpd = HTTPServer(server_address, TfidfRankingHandler)
    print("TF-IDF Ranking Service running on port 8006...")
    httpd.serve_forever()