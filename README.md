نظرة عامة على المشروع
هذا المشروع عبارة عن نظام لاسترداد المعلومات يهدف إلى معالجة البيانات النصية وإنشاء تمثيلات للوثائق باستخدام تقنيات متقدمة لدعم الاستعلامات بكفاءة. يتكون المشروع من عدة خدمات تعمل معًا لتوفير وظائف المعالجة المسبقة، إنشاء التمثيلات، معالجة الاستعلامات، وترتيب الوثائق. يستخدم النظام تقنيات مثل:

TF-IDF
SentenceTransformer
FAISS

كما يدعم النظام تحسين الاستعلامات وتخزين التمثيلات في قواعد بيانات:

SQLite
MongoDB

هيكلية المشروع
project_root/
│
├── data/
│   └── [dataset_name]/
│       ├── docs_[dataset_name].csv
│       ├── queries_[dataset_name].csv
│       ├── index.db
│       ├── tfidf_vectorizer.joblib
│       ├── tfidf_matrix.joblib
│       ├── embeddings_matrix.joblib
│       ├── embeddings_vectorizer.joblib
│       ├── embedding_index.faiss
│       └── doc_id_mapping.joblib
│
├── logs/
│   └── preproc_log_[timestamp].txt
│
├── src/
│   └── offline_services/
│       ├── preprocessing_service.py
│       ├── representation_service.py
│   └── online_services/
│       ├── frontend_service.py
│       ├── query_processing_service.py
│       ├── ranking_tfidf_service.py
│       ├── ranking_embedding_service.py
│       ├── ranking_hybrid_service.py
│       └── vector_store_service.py


وصف المكونات
مجلد data/
يحتوي على بيانات كل مجموعة بيانات 
(beir ,antique)، 
وتشمل:

ملفات CSV للوثائق والاستعلامات
قاعدة بيانات SQLite
ملفات تمثيلات النماذج  وتضمينات


مجلد src/online_services/
يحتوي على نصوص Python لتشغيل الخدمات:

frontend_service.py (الواجهة الأمامية على المنفذ 8000)
query_processing_service.py (معالجة الاستعلامات على المنفذ 8003)
ranking_tfidf_service.py (ترتيب الوثائق باستخدام TF-IDF على المنفذ 8006)
ranking_embedding_service.py (ترتيب الوثائق باستخدام التضمينات على المنفذ 8007)
ranking_hybrid_service.py (ترتيب الوثائق باستخدام نهج هجين على المنفذ 8008)
vector_store_service.py (تخزين المتجهات باستخدام FAISS على المنفذ 8005)
run_online.py لتشغيل كل البورتات 
مجلد src/offline_services/

representation_service.py يحتوي على ملفات تمثيل البيانات 
preprocessing_service.py وملف تنضيف البيانات 

ملاحظات إضافية

يستخدم النظام مكتبات مثل:
pandas
nltk
sklearn
sentence_transformers
faiss

يتم تشغيل الخدمات في بيئة افتراضية لضمان عزل التبعيات
