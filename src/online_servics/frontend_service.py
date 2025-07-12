from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import urllib.parse
import numpy as np
import requests
import time

class FrontendHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/search'):
            parsed_path = urllib.parse.urlparse(self.path)
            query_params = urllib.parse.parse_qs(parsed_path.query)
            query = query_params.get('query', [''])[0]
            dataset = query_params.get('dataset', ['antique'])[0]
            representation = query_params.get('representation', ['tfidf'])[0]
            enhance_query = query_params.get('enhanceQuery', ['false'])[0].lower() == 'true'
            use_vector_store = query_params.get('useVectorStore', ['false'])[0].lower() == 'true'

            print(f"Processing request: query={query}, dataset={dataset}, representation={representation}, enhanceQuery={enhance_query}, useVectorStore={use_vector_store}")
            
            # فحص توافق useVectorStore مع representation
            if use_vector_store and representation not in ['embedding', 'hybrid']:
                self.send_response(400)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Vector Store can only be used with embedding or hybrid representation"}).encode('utf-8'))
                return

            # معالجة الاستعلام
            process_url = "http://localhost:8003/process_query"
            process_data = {"query": query, "dataset_name": dataset, "representation_type": representation, "enhance": enhance_query}
            start_time = time.time()
            try:
                process_response = requests.post(process_url, json=process_data)
                process_response.raise_for_status()
                process_data = process_response.json()
                print(f"Query processing took {time.time() - start_time:.2f} seconds")
                if process_data.get("status") != "success":
                    raise ValueError("Query processing failed with status: " + process_data.get("message", "Unknown error"))
                
                query_vector = process_data.get("query_vector")
                processed_query = process_data.get("processed_query", None) if representation == 'tfidf' else None
                if not query_vector:
                    raise ValueError("Invalid or empty query vector")
                
                if representation == 'hybrid':
                    if not isinstance(query_vector, dict) or 'tfidf' not in query_vector or 'embedding' not in query_vector:
                        raise ValueError("Invalid query vector format for hybrid representation")
                    print(f"Hybrid query vector: tfidf shape={len(query_vector['tfidf'])}, embedding shape={len(query_vector['embedding'])}")
                else:
                    if hasattr(query_vector, 'tolist'):
                        query_vector = query_vector.tolist()
                    elif isinstance(query_vector, np.ndarray):
                        query_vector = query_vector.tolist()
                    print(f"Query vector: {query_vector[:10]}...")
            
            except (requests.RequestException, ValueError) as e:
                print(f"Error in Query Processing: {str(e)}")
                self.send_response(500)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Query processing failed: {str(e)}"}).encode('utf-8'))
                return

            # ترتيب الوثائق
            rank_url = f"http://localhost:{8006 if representation == 'tfidf' else 8007 if representation == 'embedding' else 8008}/rank_{representation}"
            rank_data = {
                "query_vector": query_vector,
                "dataset_name": dataset,
                "processed_query": processed_query if representation == 'tfidf' else None,
                "use_vector_store": use_vector_store
            }
            start_time = time.time()
            try:
                rank_response = requests.post(rank_url, json=rank_data)
                rank_response.raise_for_status()
                rank_data = rank_response.json()
                print(f"Ranking took {time.time() - start_time:.2f} seconds")
                if rank_data.get("status") != "success":
                    raise ValueError("Ranking failed with status: " + rank_data.get("message", "Unknown error"))
                ranked_docs = rank_data.get("ranked_docs", [])
                if not ranked_docs or not isinstance(ranked_docs, list):
                    raise ValueError("Invalid or empty ranked documents")
                print(f"Ranked docs count: {len(ranked_docs)}")
                
            except (requests.RequestException, ValueError) as e:
                print(f"Error in Ranking: {str(e)}")
                self.send_response(500)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Ranking failed: {str(e)}"}).encode('utf-8'))
                return

            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self.end_headers()
            response = {
                "results": ranked_docs,
            }
            print(f"Response to client: {response}")
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
        
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == "__main__":
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, FrontendHandler)
    print("Frontend Service running on port 8000...")
    httpd.serve_forever()