import ast
from datetime import datetime
import io
import sqlite3
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import joblib
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import requests
from sklearn.metrics.pairwise import cosine_similarity
import torch
# تحميل بيانات NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('omw-1.4', quiet=True)



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

def preprocess_text(text):
    return custom_tokenizer(text)

global_model_cache = {}

# def get_model(dataset_name):

#     global global_model_cache
#     if dataset_name not in global_model_cache:
#         vectorizer_file = f"data/{dataset_name}/embeddings_vectorizer.joblib"
#         if os.path.exists(vectorizer_file):
#             global_model_cache[dataset_name] = joblib.load(vectorizer_file)
#             print(f"Loaded model from {vectorizer_file}")
#         else:
#             global_model_cache[dataset_name] = SentenceTransformer('all-mpnet-base-v2')
#             joblib.dump(global_model_cache[dataset_name], vectorizer_file)
#             print(f"Initialized new model and saved to {vectorizer_file}")
#     return global_model_cache[dataset_name]

def get_model(dataset_name):
    global global_model_cache
    if dataset_name not in global_model_cache:
        vectorizer_file = f"data/{dataset_name}/embeddings_vectorizer.joblib"
        if os.path.exists(vectorizer_file):
            try:
                # Load the model with map_location to ensure CPU compatibility
                with open(vectorizer_file, 'rb') as f:
                    global_model_cache[dataset_name] = joblib.load(f)
                # Move the model to CPU explicitly
                global_model_cache[dataset_name].to('cpu')
                print(f"Loaded model from {vectorizer_file} and moved to CPU")
            except Exception as e:
                print(f"Error loading model from {vectorizer_file}: {str(e)}")
                # Fallback to initializing a new model
                global_model_cache[dataset_name] = SentenceTransformer('all-mpnet-base-v2')
                global_model_cache[dataset_name].to('cpu')
                joblib.dump(global_model_cache[dataset_name], vectorizer_file)
                print(f"Initialized new model and saved to {vectorizer_file}")
        else:
            global_model_cache[dataset_name] = SentenceTransformer('all-mpnet-base-v2')
            global_model_cache[dataset_name].to('cpu')
            joblib.dump(global_model_cache[dataset_name], vectorizer_file)
            print(f"Initialized new model and saved to {vectorizer_file}")
    return global_model_cache[dataset_name]

def load_embeddings(dataset_name):
    embedding_file = f"data/{dataset_name}/embeddings_matrix.joblib"
    
    if os.path.exists(embedding_file):
        return joblib.load(embedding_file)
    else:
        # Handle the case where the embedding file does not exist
        raise FileNotFoundError(f"Embedding file {embedding_file} not found. Please generate embeddings first.")
        


def process_query(query, dataset_name, representation_type='tfidf', enhance=False):
    processed_query = preprocess_text(query)
    log_file = f"logs/preproc_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    if enhance:
            #بحمل تمثيلا الماتريكس للداتا 
            query_embeddings = load_embeddings(dataset_name)
            ########
            if len(query_embeddings) == 0:
                print(f"No query embeddings available for {dataset_name}")
                with open(log_file, "a", encoding="utf-8") as log:
                    log.write(f"No query embeddings available for {dataset_name} at {datetime.now()}\n")
                return query, processed_query
            #بحمل فيكتورايزي ال embedding 
            model = get_model(dataset_name)
            #بيعمل الكويري  كلمة وحده 
            query_text = ' '.join(processed_query)
            # بحولها لفيكتور 
            query_vector = model.encode([query_text], convert_to_numpy=True)
            similarities = cosine_similarity(query_vector, query_embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:5]
            print(f"Top similarities: {similarities[top_indices]}")
            ###########################################
            with open(log_file, "a", encoding="utf-8") as log:
                log.write(f"Top similarities for {dataset_name}: {similarities[top_indices]} at {datetime.now()}\n")
            #بيتاكد من وجود الداتا بيز
            db_file = f"data/{dataset_name}/index.db"
            ####################
            if not os.path.exists(db_file):
                print(f"Error: Database file {db_file} not found")
                with open(log_file, "a", encoding="utf-8") as log:
                    log.write(f"Error: Database file {db_file} not found at {datetime.now()}\n")
                return query, processed_query

            # تحميل ملف doc_id_mapping.joblib
            mapping_file = f"data/{dataset_name}/doc_id_mapping.joblib"
            ####################################
            if not os.path.exists(mapping_file):
                print(f"Error: Mapping file {mapping_file} not found")
                with open(log_file, "a", encoding="utf-8") as log:
                    log.write(f"Error: Mapping file {mapping_file} not found at {datetime.now()}\n")
                return query, processed_query

            try:
                index_to_doc_id = joblib.load(mapping_file)
                print(f"Loaded doc_id_mapping with {len(index_to_doc_id)} entries")
                with open(log_file, "a", encoding="utf-8") as log:
                    log.write(f"Loaded doc_id_mapping with {len(index_to_doc_id)} entries at {datetime.now()}\n")
            except Exception as e:
                print(f"Error loading doc_id_mapping: {str(e)}")
                with open(log_file, "a", encoding="utf-8") as log:
                    log.write(f"Error loading doc_id_mapping: {str(e)} at {datetime.now()}\n")
                return query, processed_query

            try:
                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()
                cursor.execute('SELECT doc_id, processed_text FROM documents')
                doc_data = {row[0]: row[1] for row in cursor.fetchall()}
                conn.close()

                if not doc_data:
                    print(f"Warning: No processed documents found in database for {dataset_name}")
                    with open(log_file, "a", encoding="utf-8") as log:
                        log.write(f"No processed documents found in database for {dataset_name} at {datetime.now()}\n")
                    return query, processed_query

                print(f"Total documents in doc_data: {len(doc_data)}")
                print(f"Sample doc_ids: {list(doc_data.keys())[:5]}")
                print(f"Top indices: {top_indices}")

                similar_queries = []
                max_add = 5
                added = 0
                top_doc_words = []

                for idx in top_indices:
                    # تحويل idx إلى doc_id باستخدام doc_id_mapping
                    doc_id = index_to_doc_id.get(idx)
                    if doc_id is None:
                        print(f"Warning: No doc_id mapped for index {idx}")
                        with open(log_file, "a", encoding="utf-8") as log:
                            log.write(f"No doc_id mapped for index {idx} at {datetime.now()}\n")
                        continue

                    if doc_id in doc_data:
                        try:
                            sim_score = similarities[idx]
                            if 0.4 < sim_score < 0.999:
                                processed_query_text = ast.literal_eval(doc_data[doc_id]) if doc_data[doc_id] else []
                                if not processed_query_text:
                                    print(f"Warning: Empty processed_text for doc_id {doc_id}")
                                    with open(log_file, "a", encoding="utf-8") as log:
                                        log.write(f"Empty processed_text for doc_id {doc_id} at {datetime.now()}\n")
                                    continue
                                print(f"Selected document at doc_id {doc_id}, similarity: {sim_score}, text: {processed_query_text}")
                                with open(log_file, "a", encoding="utf-8") as log:
                                    log.write(f"Selected document at doc_id {doc_id}, similarity: {sim_score}, text: {processed_query_text} at {datetime.now()}\n")
                                similar_queries.append(processed_query_text)
                                top_doc_words.extend(processed_query_text)
                                added += 1
                                if added >= max_add:
                                    break
                        except (ValueError, SyntaxError) as e:
                            print(f"Error parsing processed_text at doc_id {doc_id}: {str(e)}")
                            with open(log_file, "a", encoding="utf-8") as log:
                                log.write(f"Error parsing processed_text at doc_id {doc_id}: {str(e)} at {datetime.now()}\n")
                            continue
                    else:
                        print(f"Warning: doc_id {doc_id} not found in doc_data")

                if not similar_queries:
                    print(f"No similar documents found for {dataset_name} with sufficient similarity")
                    with open(log_file, "a", encoding="utf-8") as log:
                        log.write(f"No similar documents found for {dataset_name} with sufficient similarity at {datetime.now()}\n")
                    return query, processed_query

                new_words = []
                seen = set(processed_query)
                for word in top_doc_words:
                    if word not in seen and word not in new_words:
                        new_words.append(word)
                        if len(new_words) >= 4:
                            break

                additional_words = ' '.join(new_words)
                print(f"Additional words from top 5 documents: {additional_words}")
                with open(log_file, "a", encoding="utf-8") as log:
                    log.write(f"Additional words for {dataset_name}: {additional_words} at {datetime.now()}\n")

                if additional_words:
                    query = query + ' ' + additional_words
                    processed_query = custom_tokenizer(query)
                    print(f"🔍 Enhanced Query: {query}")
                    with open(log_file, "a", encoding="utf-8") as log:
                        log.write(f"Enhanced Query for {dataset_name}: {query} at {datetime.now()}\n")
                else:
                    print(f"No new words to add to query for {dataset_name}")
                    with open(log_file, "a", encoding="utf-8") as log:
                        log.write(f"No new words to add to query for {dataset_name} at {datetime.now()}\n")

            except Exception as e:
                print(f"Error enhancing query for {dataset_name}: {str(e)}")
                with open(log_file, "a", encoding="utf-8") as log:
                    log.write(f"Error enhancing query for {dataset_name}: {str(e)} at {datetime.now()}\n")
                if 'conn' in locals():
                    conn.close()
    if representation_type == 'tfidf':
        vectorizer_file = f"data/{dataset_name}/tfidf_vectorizer.joblib"
        if not os.path.exists(vectorizer_file):
            return {"status": "error", "message": "TF-IDF vectorizer not found"}
        vectorizer = joblib.load(vectorizer_file)
        if vectorizer.tokenizer != custom_tokenizer:
            print("تحديث TfidfVectorizer لاستخدام custom_tokenizer")
            vectorizer.tokenizer = custom_tokenizer
            vectorizer.lowercase = False
            vectorizer.preprocessor = None
            vectorizer.token_pattern = None
            vectorizer.ngram_range = (1, 2)  # إضافة bigrams
            vectorizer.max_features = 15000  # زيادة عدد الميزات
            vectorizer.min_df = 2  # تجاهل الكلمات النادرة
            vectorizer.max_df = 0.85  # تجاهل الكلمات الشائعة جدًا
            vectorizer.sublinear_tf = True  # تسجيل لوغاريتمي
        query_vector = vectorizer.transform([query])
        query_vector_array = query_vector.toarray()[0]
        query_vector = query_vector_array.tolist()
        non_zero_indices = np.nonzero(query_vector_array)[0]
        non_zero_values = query_vector_array[non_zero_indices]
        non_zero_terms = [vectorizer.get_feature_names_out()[idx] for idx in non_zero_indices]
        print(f"TF-IDF query vector shape: {query_vector_array.shape}")
        print(f"Non-zero elements count: {len(non_zero_indices)}")
        if len(non_zero_indices) > 0:
            print(f"Non-zero terms and values: {list(zip(non_zero_terms, non_zero_values))}")
        else:
            print("Warning: Query vector is all zeros!")
            print(f"Processed query tokens: {processed_query}")
            print(f"Vocabulary sample: {list(vectorizer.vocabulary_.items())[:10]}")
        return {
            "status": "success",
            "query_vector": query_vector,
            "processed_query": processed_query
        }
    
    elif representation_type == 'embedding':
        model = get_model(dataset_name)
        query_text = ' '.join(processed_query)
        query_vector = model.encode([query_text]).tolist()[0]
        print(f"Embedding query vector shape: {np.array(query_vector).shape}")
        return {
            "status": "success", 
            "query_vector": query_vector,
            "processed_query": processed_query
            }
    
    elif representation_type == 'hybrid':
        vectorizer_file = f"data/{dataset_name}/tfidf_vectorizer.joblib"
        if not os.path.exists(vectorizer_file):
            return {"status": "error", "message": "TF-IDF vectorizer not found"}
        vectorizer = joblib.load(vectorizer_file)
        if vectorizer.tokenizer != custom_tokenizer:
            vectorizer.tokenizer = custom_tokenizer
            vectorizer.lowercase = False
            vectorizer.preprocessor = None
            vectorizer.token_pattern = None
            vectorizer.min_df = 1
        query_vector_tfidf = vectorizer.transform([query])
        query_vector_tfidf_array = query_vector_tfidf.toarray()[0]
        query_vector_tfidf = query_vector_tfidf_array.tolist()
        non_zero_indices = np.nonzero(query_vector_tfidf_array)[0]
        non_zero_values = query_vector_tfidf_array[non_zero_indices]
        non_zero_terms = [vectorizer.get_feature_names_out()[idx] for idx in non_zero_indices]
        print(f"TF-IDF query vector shape: {query_vector_tfidf_array.shape}")
        print(f"Non-zero elements count: {len(non_zero_indices)}")
        if len(non_zero_indices) > 0:
            print(f"Non-zero terms and values: {list(zip(non_zero_terms, non_zero_values))}")
        else:
            print("Warning: Query vector is all zeros!")
            print(f"Processed query tokens: {processed_query}")
            print(f"Vocabulary sample: {list(vectorizer.vocabulary_.items())[:10]}")
        model = get_model(dataset_name)
        query_text = ' '.join(processed_query)
        query_vector_embedding = model.encode([query_text]).tolist()[0]
        print(f"Embedding query vector shape: {np.array(query_vector_embedding).shape}")
        return {
            "status": "success",
            "query_vector": {
                "tfidf": query_vector_tfidf,
                "embedding": query_vector_embedding,
                "processed_query": processed_query
            }
        }
    
    return {"status": "error", "message": "Invalid representation_type"}

class QueryProcessingHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/process_query':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode('utf-8'))
            except json.JSONDecodeError as e:
                self.send_response(400)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": f"Invalid JSON: {str(e)}"}).encode('utf-8'))
                return
            
            query = data.get('query')
            dataset_name = data.get('dataset_name')
            representation_type = data.get('representation_type', 'tfidf')
            enhance = data.get('enhance', False)
            
            if not query or not dataset_name:
                self.send_response(400)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                response = {"status": "error", "message": "query and dataset_name are required"}
                self.wfile.write(json.dumps(response).encode('utf-8'))
                return
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self.end_headers()
            response = process_query(query, dataset_name, representation_type, enhance)
            self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == "__main__":
    server_address = ('', 8003)
    httpd = HTTPServer(server_address, QueryProcessingHandler)
    print("Query Processing Service running on port 8003...")
    httpd.serve_forever()