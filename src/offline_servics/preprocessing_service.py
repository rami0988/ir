

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import os
from datetime import datetime
import sqlite3
from collections import defaultdict
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

# تحميل بيانات NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('omw-1.4', quiet=True)

def ensure_directories_and_files(dataset_name):
    # إنشاء مجلد logs إذا لم يكن موجودًا
    if not os.path.exists("logs"):
        os.makedirs("logs")
    
    # إنشاء مجلد البيانات ومجلد فرعي لمجموعة البيانات
    data_dir = f"data/{dataset_name}"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # إنشاء ملف CSV للوثائق إذا لم يكن موجودًا
    docs_csv_file = f"{data_dir}/docs_{dataset_name}.csv"
    if not os.path.exists(docs_csv_file):
        empty_df = pd.DataFrame(columns=['doc_id', 'text'])
        empty_df.to_csv(docs_csv_file, index=False)
    
    # إنشاء ملف CSV للاستعلامات إذا لم يكن موجودًا
    queries_csv_file = f"{data_dir}/queries_{dataset_name}.csv"
    if not os.path.exists(queries_csv_file):
        empty_df = pd.DataFrame(columns=['query_id', 'text'])
        empty_df.to_csv(queries_csv_file, index=False)
    
    # إنشاء قاعدة بيانات إذا لم تكن موجودة
    db_file = f"{data_dir}/index.db"
    if not os.path.exists(db_file):
        conn = sqlite3.connect(db_file)
        conn.close()

log_file = f"logs/preproc_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

def preprocess_text(text):
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

def preprocess_dataset(dataset_name):
    try:
        print(f"Starting preprocessing for dataset: {dataset_name}")
        conn = sqlite3.connect(f"data/{dataset_name}/index.db")
        cursor = conn.cursor()

        # إنشاء جدول الوثائق
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                text TEXT,
                processed_text TEXT
            )
        ''')

        # إنشاء جدول الاستعلامات
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS queries (
                query_id TEXT PRIMARY KEY,
                text TEXT,
                processed_text TEXT
            )
        ''')

        # التحقق من وجود بيانات منضفة في جدول الوثائق
        cursor.execute('SELECT COUNT(*) FROM documents')
        docs_count = cursor.fetchone()[0]
        if docs_count == 0:
            print(f"No processed documents found, preprocessing documents for {dataset_name}")
            docs_df = pd.read_csv(f"data/{dataset_name}/docs_{dataset_name}.csv")
            if 'doc_id' not in docs_df.columns or 'text' not in docs_df.columns:
                raise ValueError("Documents CSV file must contain 'doc_id' and 'text' columns")
            
            docs_df['processed_text'] = docs_df['text'].apply(preprocess_text)
            
            cursor.execute('DELETE FROM documents')
            for _, row in docs_df.iterrows():
                cursor.execute('INSERT INTO documents (doc_id, text, processed_text) VALUES (?, ?, ?)',
                              (row['doc_id'], row['text'], str(row['processed_text'])))
            conn.commit()
            
            # إنشاء الفهرس العكسي
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS inverted_index (
                    term TEXT,
                    doc_id TEXT,
                    frequency INTEGER,
                    PRIMARY KEY (term, doc_id)
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_term ON inverted_index(term)')
            cursor.execute('DELETE FROM inverted_index')
            
            term_freq = defaultdict(lambda: defaultdict(int))
            cursor.execute('SELECT doc_id, processed_text FROM documents')
            for doc_id, processed_text in cursor.fetchall():
                tokens = eval(processed_text) if processed_text else []
                for token in tokens:
                    term_freq[token][doc_id] += 1
            
            for term, doc_counts in term_freq.items():
                for doc_id, freq in doc_counts.items():
                    cursor.execute('INSERT INTO inverted_index (term, doc_id, frequency) VALUES (?, ?, ?)',
                                  (term, doc_id, freq))
            conn.commit()
            
            with open(log_file, "a", encoding="utf-8") as log:
                log.write(f"Completed preprocessing {len(docs_df)} documents for {dataset_name} at {datetime.now()}\n")
        else:
            print(f"Found {docs_count} processed documents for {dataset_name}, skipping document preprocessing")
            with open(log_file, "a", encoding="utf-8") as log:
                log.write(f"Skipped document preprocessing for {dataset_name}: {docs_count} documents already processed\n")

        # التحقق من وجود بيانات منضفة في جدول الاستعلامات
        cursor.execute('SELECT COUNT(*) FROM queries')
        queries_count = cursor.fetchone()[0]
        if queries_count == 0:
            print(f"No processed queries found, preprocessing queries for {dataset_name}")
            queries_df = pd.read_csv(f"data/{dataset_name}/queries_{dataset_name}.csv")
            if 'query_id' not in queries_df.columns or 'text' not in queries_df.columns:
                raise ValueError("Queries CSV file must contain 'query_id' and 'text' columns")
            
            queries_df['processed_text'] = queries_df['text'].apply(preprocess_text)
            
            cursor.execute('DELETE FROM queries')
            for _, row in queries_df.iterrows():
                cursor.execute('INSERT INTO queries (query_id, text, processed_text) VALUES (?, ?, ?)',
                              (row['query_id'], row['text'], str(row['processed_text'])))
            conn.commit()
            
            with open(log_file, "a", encoding="utf-8") as log:
                log.write(f"Completed preprocessing {len(queries_df)} queries for {dataset_name} at {datetime.now()}\n")
        else:
            print(f"Found {queries_count} processed queries for {dataset_name}, skipping query preprocessing")
            with open(log_file, "a", encoding="utf-8") as log:
                log.write(f"Skipped query preprocessing for {dataset_name}: {queries_count} queries already processed\n")

        conn.close()
        return {"status": "success", "message": f"Processed dataset {dataset_name}"}
    except Exception as e:
        with open(log_file, "a", encoding="utf-8") as log:
            log.write(f"Error preprocessing {dataset_name}: {str(e)}\n")
        conn.close()
        return {"status": "error", "message": str(e)}

class PreprocessingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self.end_headers()
            response = {"status": "running"}
            self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/preprocess':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode('utf-8'))
            except json.JSONDecodeError as e:
                self.send_response(400)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                response = {"status": "error", "message": f"Invalid JSON: {str(e)}"}
                self.wfile.write(json.dumps(response).encode('utf-8'))
                return
            
            dataset_name = data.get('dataset_name')
            if not dataset_name:
                self.send_response(400)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                response = {"status": "error", "message": "dataset_name is required"}
                self.wfile.write(json.dumps(response).encode('utf-8'))
                return
            
            # Ensure required directories and files exist
            ensure_directories_and_files(dataset_name)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self.end_headers()
            response = preprocess_dataset(dataset_name)
            self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == "__main__":
    server_address = ('', 8001)
    httpd = HTTPServer(server_address, PreprocessingHandler)
    print("Preprocessing Service running on port 8001...")
    httpd.serve_forever()