import faiss
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import os
import sqlite3

class VectorStoreHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/vector_search':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))

            query_vector = data.get('query_vector')
            dataset_name = data.get('dataset_name')
            
            if not query_vector or not dataset_name:
                self._send_error(400, "query_vector and dataset_name are required")
                return
            
            response = self.search_vector_store(query_vector, dataset_name)
            self._send_response(response)
        else:
            self.send_response(404)
            self.end_headers()

    def search_vector_store(self, query_vector, dataset_name):
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

            # Use Vector Store with FAISS index
            index_path = f"data/{dataset_name}/embedding_index.faiss"
            if not os.path.exists(index_path):
                return {"status": "error", "message": "Vector store index not found"}
            
            index = faiss.read_index(index_path)
            # Convert query_vector to float32 and ensure it's a 2D array
            query_vector_np = np.array([query_vector], dtype=np.float32)
            distances, indices = index.search(query_vector_np, k=10)
            
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(doc_mapping):
                    doc_id, text = doc_mapping[idx]
                    ranked_docs.append({
                        "doc_id": doc_id,
                        "score": float(1.0 / (1.0 + distance)),  # Convert to Python float
                        "text": text
                    })
            
            if not ranked_docs:
                return {"status": "error", "message": "No valid documents found"}
            
            return {"status": "success", "ranked_docs": ranked_docs}
        
        except Exception as e:
            print(f"Vector Store error: {str(e)}")
            return {"status": "error", "message": f"Vector Store search failed: {str(e)}"}

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
    server_address = ('', 8005)
    httpd = HTTPServer(server_address, VectorStoreHandler)
    print("Vector Store Service running on port 8005...")
    httpd.serve_forever()



