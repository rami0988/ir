# import csv
# from tqdm import tqdm

# # دالة لقراءة queries.txt
# def read_queries(file_path):
#     queries = []
#     with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
#         for line in f:
#             query_id, text = line.strip().split('\t', 1)
#             queries.append((query_id, text))
#     return queries

# # دالة لقراءة qrels.txt مع استخراج query_id, doc_id, relevance فقط
# def read_qrels(file_path):
#     qrels = []
#     with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
#         for line_number, line in enumerate(f, 1):
#             # تخطي الأسطر الفارغة
#             if not line.strip():
#                 print(f"تحذير: تخطي السطر الفارغ في السطر رقم {line_number}")
#                 continue
#             # تقسيم السطر إلى حقول
#             parts = line.strip().split('\t')
#             # التحقق من أن السطر يحتوي على 4 حقول بالضبط
#             if len(parts) != 4:
#                 print(f"تحذير: تخطي السطر غير الصحيح في السطر رقم {line_number}: {line.strip()} (تم العثور على {len(parts)} حقول)")
#                 continue
#             # استخراج query_id, doc_id, relevance وتجاهل الحقل الثاني (Q0)
#             query_id, _, doc_id, relevance = parts
#             qrels.append((query_id, doc_id, relevance))
#     return qrels

# # مسارات الملفات
# queries_file = "queries.txt"
# qrels_file = "qrels"

# # حفظ الاستعلامات في ملف CSV
# with open("queries_antique.csv", "w", newline='', encoding="utf-8", errors="replace") as f:
#     writer = csv.writer(f)
#     writer.writerow(["query_id", "text"])
#     for query_id, text in tqdm(read_queries(queries_file), desc="Antique Queries"):
#         writer.writerow([query_id, text])

# # حفظ qrels في ملف CSV مع query_id, doc_id, relevance فقط
# with open("qrels_antique.csv", "w", newline='', encoding="utf-8", errors="replace") as f:
#     writer = csv.writer(f)
#     writer.writerow(["query_id", "doc_id", "relevance"])
#     for query_id, doc_id, relevance in tqdm(read_qrels(qrels_file), desc="Antique Qrels"):
#         writer.writerow([query_id, doc_id, relevance])



import pandas as pd

# Read the input data from a text file (assuming space-separated values)
input_file = 'qrels'  # Replace with your input file name
data = pd.read_csv(input_file, sep='\s+', header=None, names=['query_id', 'q0', 'doc_id', 'relevance'])

# Select only the required columns
output_data = data[['query_id', 'doc_id', 'relevance']]

# Save to CSV
output_data.to_csv('qrels_antique.csv', index=False)