import asyncio
import platform
from urllib.parse import parse_qs, urlparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from scipy import sparse
import joblib
import sqlite3
import faiss
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
from io import BytesIO
from tqdm import tqdm
from pymongo import MongoClient
from gridfs import GridFS
from io import BytesIO
import joblib
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('omw-1.4')

global_model = None

def custom_tokenizer(text):
    if not isinstance(text, str) or not text.strip():
        return []
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)   
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english')) - {'not', 'no', 'never'}  
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2] 
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token, pos='v') for token in tokens] 
    return tokens

def load_documents(dataset_name):
    try:
        db_path = f"data/{dataset_name}/index.db"
        if not os.path.exists(db_path):
            return []

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT text FROM documents")
        rows = cursor.fetchall()
        conn.close()

        documents = [row[0] for row in rows]
        return documents
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []
    

def load_documents_emb(dataset_name):
    try:
        db_path = f"data/{dataset_name}/index.db"
        if not os.path.exists(db_path):
            return []

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT processed_text FROM documents")
        rows = cursor.fetchall()
        conn.close()

        documents = [row[0] for row in rows]
        return documents
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []

def get_model():
    global global_model
    if global_model is None:
        global_model = SentenceTransformer('all-mpnet-base-v2')
    return global_model

# def store_representation_in_db(dataset_name, representation_type, data):
#     try:
#         db_path = f"data/{dataset_name}/index.db"
#         conn = sqlite3.connect(db_path)
#         cursor = conn.cursor()
        
#         cursor.execute('''
#             CREATE TABLE IF NOT EXISTS representations (
#                 representation_type TEXT,
#                 data BLOB,
#                 shape TEXT,
#                 PRIMARY KEY (representation_type)
#             )
#         ''')
        
#         buffer = BytesIO()
#         joblib.dump(data, buffer)
#         buffer.seek(0)
#         serialized_data = buffer.read()
#         shape = str(data.shape) if hasattr(data, 'shape') else 'model'
#         cursor.execute('INSERT OR REPLACE INTO representations (representation_type, data, shape) VALUES (?, ?, ?)',
#                       (representation_type, serialized_data, shape))
        
#         conn.commit()
#         conn.close()
#     except Exception as e:
#         print(f"Error storing representation in database: {e}")

# def load_representation_from_db(dataset_name, representation_type):
#     try:
#         db_path = f"data/{dataset_name}/index.db"
#         conn = sqlite3.connect(db_path)
#         cursor = conn.cursor()
        
#         cursor.execute('SELECT data FROM representations WHERE representation_type = ?', (representation_type,))
#         row = cursor.fetchone()
#         conn.close()
        
#         if row:
#             buffer = BytesIO(row[0])
#             return joblib.load(buffer)
#         return None
#     except Exception as e:
#         print(f"Error loading representation from database: {e}")
#         return None



def load_representation_from_mongo(dataset_name, representation_type):
    try:
        client = MongoClient("mongodb://localhost:27017/")
        db = client[dataset_name]
        fs = GridFS(db, collection="representation")

        # مطابقة مباشرة حسب نوع التمثيل
        filename_map = {
            "tfidf_vectorizer": "tfidf_vectorizer.joblib",
            "tfidf_matrix": "tfidf_matrix.joblib",
            "embeddings_matrix": "embeddings_matrix.joblib",
            "embeddings_vectorizer": "embeddings_vectorizer.joblib",
            "embedding_index": "embedding_index.faiss"
        }

        filename = filename_map.get(representation_type)
        if not filename:
            print(f"❌ نوع التمثيل غير معروف: {representation_type}")
            return None

        file_doc = fs.find_one({"filename": filename})
        if not file_doc:
            print(f"⚠️ الملف {filename} غير موجود في MongoDB ({dataset_name}.representation)")
            return None

        buffer = BytesIO(file_doc.read())
        if filename.endswith(('.joblib',)):
            return joblib.load(buffer)
        elif filename.endswith('.faiss'):
            return buffer.read()  # نعيد البيانات الثنائية فقط لتأكيد وجودها
        else:
            return buffer.read()

    except Exception as e:
        print(f"❌ خطأ أثناء تحميل التمثيل من MongoDB: {e}")
        return None


def create_tfidf(dataset_name):
    try:
        tfidf_file = f"data/{dataset_name}/tfidf_matrix.joblib"
        if os.path.exists(tfidf_file):
            return {"status": "success", "message": "TF-IDF already exists"}
        
        docs = load_documents(dataset_name)
        if not docs:
            return {"status": "error", "message": "No documents found"}
        
        vectorizer = TfidfVectorizer(
        tokenizer=custom_tokenizer,
        lowercase=False,
        preprocessor=None,
        token_pattern=None,
        ngram_range=(1,2),  
        max_features=15000,  
        min_df=2,  
        max_df=0.85, 
        sublinear_tf=True   
    )
        tfidf_matrix = vectorizer.fit_transform(docs)
        joblib.dump(vectorizer, f"data/{dataset_name}/tfidf_vectorizer.joblib")
        joblib.dump(tfidf_matrix, tfidf_file)
        # store_representation_in_db(dataset_name, 'tfidf', tfidf_matrix)
        
        return {"status": "success", "message": f"TF-IDF created with shape {tfidf_matrix.shape}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}



def create_embedding(dataset_name, batch_size=3000):
    try:
        embedding_matrix_file = f"data/{dataset_name}/embeddings_matrix.joblib"
        embedding_vectorizer_file = f"data/{dataset_name}/embeddings_vectorizer.joblib"
        index_file = f"data/{dataset_name}/embedding_index.faiss"
        
        # Ensure output directory exists
        os.makedirs(f"data/{dataset_name}", exist_ok=True)
        
        # Check if all required files exist
        if os.path.exists(embedding_matrix_file) and os.path.exists(embedding_vectorizer_file) and os.path.exists(index_file):
            return {"status": "success", "message": "Embeddings matrix, vectorizer, and faiss index already exist"}
        
        # Load documents
        docs = load_documents_emb(dataset_name)
        if not docs:
            return {"status": "error", "message": "No documents found"}

        # Generate embeddings if matrix or vectorizer is missing
        if not os.path.exists(embedding_matrix_file) or not os.path.exists(embedding_vectorizer_file):
            model = get_model()
            embeddings = []
            total_batches = (len(docs) + batch_size - 1) // batch_size
            with tqdm(total=total_batches, desc=f"Encoding embeddings for {dataset_name}", unit="batch") as pbar:
                for i in range(0, len(docs), batch_size):
                    batch = docs[i:i + batch_size]
                    batch_embeddings = model.encode(batch, convert_to_numpy=True)
                    embeddings.append(batch_embeddings)
                    pbar.update(1)
            embeddings = np.vstack(embeddings).astype(np.float32)
            
            # Save embeddings matrix
            joblib.dump(embeddings, embedding_matrix_file)
            
            # Save the SentenceTransformer model (vectorizer)
            joblib.dump(model, embedding_vectorizer_file)
        else:
            # Load existing embeddings
            embeddings = joblib.load(embedding_matrix_file)
            print(f"Loaded existing embeddings with shape {embeddings.shape}")

        # Generate faiss index if it doesn't exist
        if not os.path.exists(index_file):
            print("start creating")
            try:
                dimension = embeddings.shape[1]
                nlist = 1000
                m = 8
                print(f"Initializing faiss index with dimension={dimension}, nlist={nlist}, m={m}")
                quantizer = faiss.IndexFlatL2(dimension)
                index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, 8)
                
                # Progress bar for faiss index creation
                faiss_steps = ["Training", "Adding embeddings", "Saving index"]
                with tqdm(total=len(faiss_steps), desc=f"Creating faiss index for {dataset_name}", unit="step") as faiss_pbar:
                    print("Training faiss index...")
                    index.train(embeddings[:160000])
                    faiss_pbar.update(1)
                    
                    print("Adding embeddings to faiss index...")
                    index.add(embeddings)
                    faiss_pbar.update(1)
                    
                    print("Saving faiss index...")
                    faiss.write_index(index, index_file)
                    faiss_pbar.update(1)
                
                print("finish creating")
            except Exception as faiss_error:
                return {"status": "error", "message": f"Faiss index creation failed: {str(faiss_error)}"}
        
        return {"status": "success", "message": f"Embeddings matrix created with shape {embeddings.shape} and vectorizer/index saved"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def check_and_generate_representations(dataset_name):
    try:
        tfidf_file = f"data/{dataset_name}/tfidf_matrix.joblib"
        embedding_matrix_file = f"data/{dataset_name}/embeddings_matrix.joblib"
        embedding_vectorizer_file = f"data/{dataset_name}/embeddings_vectorizer.joblib"
        index_file = f"data/{dataset_name}/embedding_index.faiss"
        
        representations = [
            ("TF-IDF", tfidf_file, create_tfidf),
            ("Embeddings", embedding_matrix_file, create_embedding)
        ]
        
        files_to_generate = []
        if not os.path.exists(tfidf_file):
            files_to_generate.append(("TF-IDF", create_tfidf))
        if not os.path.exists(embedding_matrix_file) or not os.path.exists(embedding_vectorizer_file) or not os.path.exists(index_file):
            files_to_generate.append(("Embeddings", create_embedding))
        
        if not files_to_generate:
            return {"status": "success", "message": "All representations already exist"}
        
        with tqdm(total=len(files_to_generate), desc=f"Generating representations for {dataset_name}", unit="file") as pbar:
            for rep_name, create_func in files_to_generate:
                print(f"Generating {rep_name} for {dataset_name}")
                result = create_func(dataset_name)
                if result["status"] != "success":
                    return result
                pbar.update(1)
        
        return {"status": "success", "message": "Representations checked and generated if needed"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
class RepresentationHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/create_representation':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            dataset_name = data.get('dataset_name')
            representation_type = data.get('representation_type', 'tfidf')
            
            if not dataset_name:
                self.send_response(400)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                response = {"status": "error", "message": "dataset_name is required"}
                self.wfile.write(json.dumps(response).encode('utf-8'))
                return
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self.end_headers()
            if representation_type == 'tfidf':
                response = create_tfidf(dataset_name)
            elif representation_type == 'embedding':
                response = create_embedding(dataset_name)
            else:
                response = {"status": "error", "message": "Invalid representation_type"}
            self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path.startswith('/get_representation'):
            parsed_path = urlparse(self.path)
            params = parse_qs(parsed_path.query)
            dataset_name = params.get('dataset_name', [None])[0]
            representation_type = params.get('representation_type', [None])[0]

            self.send_response(200)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self.end_headers()

            if not dataset_name or not representation_type:
                response = {"status": "error", "message": "dataset_name and representation_type are required"}
                self.wfile.write(json.dumps(response).encode('utf-8'))
                return

            try:
                data = load_representation_from_mongo(dataset_name, representation_type)
                if data is None:
                    # fallback: local file
                    local_paths = {
                        "tfidf_vectorizer": f"data/{dataset_name}/tfidf_vectorizer.joblib",
                        "tfidf_matrix": f"data/{dataset_name}/tfidf_matrix.joblib",
                        "embeddings_matrix": f"data/{dataset_name}/embeddings_matrix.joblib",
                        "embeddings_vectorizer": f"data/{dataset_name}/embeddings_vectorizer.joblib",
                        "embedding_index": f"data/{dataset_name}/embedding_index.faiss"
                    }
                    file_path = local_paths.get(representation_type)

                    if not file_path or not os.path.exists(file_path):
                        response = {"status": "error", "message": f"{representation_type} not found in MongoDB or local"}
                        self.wfile.write(json.dumps(response).encode('utf-8'))
                        return

                    if file_path.endswith(".joblib"):
                        loaded = joblib.load(file_path)
                        shape = getattr(loaded, "shape", None)
                        response = {
                            "status": "success",
                            "source": "local",
                            "shape": shape if shape is not None else str(type(loaded))
                        }

                    elif file_path.endswith(".faiss"):
                        response = {
                            "status": "success",
                            "source": "local",
                            "message": "Index file exists (binary)",
                            "shape": None
                        }

                else:
                    # Loaded from MongoDB
                    if isinstance(data, bytes):
                        response = {
                            "status": "success",
                            "source": "mongo",
                            "message": "Binary file loaded from MongoDB",
                            "shape": None
                        }
                    else:
                        shape = getattr(data, "shape", None)
                        response = {
                            "status": "success",
                            "source": "mongo",
                            "shape": shape if shape is not None else str(type(data))
                        }

                self.wfile.write(json.dumps(response).encode('utf-8'))

            except Exception as e:
                response = {"status": "error", "message": str(e)}
                self.wfile.write(json.dumps(response).encode('utf-8'))
       

    # def do_GET(self):
    #     if self.path.startswith('/get_representation'):
    #         parsed_path = urlparse(self.path)
    #         params = parse_qs(parsed_path.query)
    #         dataset_name = params.get('dataset_name', [None])[0]
    #         representation_type = params.get('representation_type', ['tfidf'])[0]
            
    #         if not dataset_name:
    #             self.send_response(400)
    #             self.send_header('Content-type', 'application/json; charset=utf-8')
    #             self.end_headers()
    #             response = {"status": "error", "message": "dataset_name is required"}
    #             self.wfile.write(json.dumps(response).encode('utf-8'))
    #             return
            
    #         self.send_response(200)
    #         self.send_header('Content-type', 'application/json; charset=utf-8')
    #         self.end_headers()
    #         if representation_type == 'tfidf':
    #             tfidf_matrix = load_representation_from_mongo(dataset_name, 'tfidf')
    #             if tfidf_matrix is None:
    #                 tfidf_matrix = joblib.load(f"data/{dataset_name}/tfidf_matrix.joblib")
    #             response = {"status": "success", "shape": tfidf_matrix.shape}
    #         elif representation_type == 'embedding':
    #             embeddings = load_representation_from_mongo(dataset_name, 'embeddings_matrix')
    #             if embeddings is None:
    #                 embeddings = joblib.load(f"data/{dataset_name}/embeddings_matrix.joblib")
    #             response = {"status": "success", "shape": embeddings.shape}
    #         else:
    #             response = {"status": "error", "message": "Invalid representation_type"}
    #         self.wfile.write(json.dumps(response).encode('utf-8'))
    #     else:
    #         self.send_response(404)
    #         self.end_headers()

if __name__ == "__main__":
    server_address = ('', 8002)
    httpd = HTTPServer(server_address, RepresentationHandler)
    print("Representation Service running on port 8002...")
    
    data_dir = "data"
    if os.path.exists(data_dir):
        datasets = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]
        with tqdm(total=len(datasets), desc="Processing datasets", unit="dataset") as pbar:
            for dataset_name in datasets:
                print(f"Checking representations for dataset: {dataset_name}")
                check_and_generate_representations(dataset_name)
                pbar.update(1)
    
    httpd.serve_forever()