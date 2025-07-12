import ir_datasets
import csv
from tqdm import tqdm

# تحميل مجموعة ANTIQUE
antique = ir_datasets.load("antique/test/non-offensive")

# with open("docs_antique.csv", "w", newline='', encoding="utf-8", errors="replace") as f:
#     writer = csv.writer(f)
#     writer.writerow(["doc_id", "text"])
#     for doc in tqdm(antique.docs_iter(), desc="ANTIQUE Docs"):
#         writer.writerow([doc.doc_id, doc.text])

# with open("queries_antique.csv", "w", newline='', encoding="utf-8", errors="replace") as f:
#     writer = csv.writer(f)
#     writer.writerow(["query_id", "text"])
#     for query in tqdm(antique.queries_iter(), desc="ANTIQUE Queries"):
#         writer.writerow([query.query_id, query.text])

with open("qrels_antique.csv", "w", newline='', encoding="utf-8", errors="replace") as f:
    writer = csv.writer(f)
    writer.writerow(["query_id", "doc_id", "relevance"])
    for qrel in tqdm(antique.qrels_iter(), desc="ANTIQUE Qrels"):
        writer.writerow([qrel.query_id, qrel.doc_id, qrel.relevance])

# # تحميل مجموعة ArgsMe
# argsme = ir_datasets.load("argsme/2020-04-01/processed/touche-2022-task-1")

# with open("docs_argsme.csv", "w", newline='', encoding="utf-8", errors="replace") as f:
#     writer = csv.writer(f)
#     writer.writerow(["doc_id", "text"])
#     for doc in tqdm(argsme.docs_iter(), desc="ArgsMe Docs"):
#         writer.writerow([doc.doc_id, doc.text])

# with open("queries_argsme.csv", "w", newline='', encoding="utf-8", errors="replace") as f:
#     writer = csv.writer(f)
#     writer.writerow(["query_id", "text"])
#     for query in tqdm(argsme.queries_iter(), desc="ArgsMe Queries"):
#         writer.writerow([query.query_id, query.text])

# with open("qrels_argsme.csv", "w", newline='', encoding="utf-8", errors="replace") as f:
#     writer = csv.writer(f)
#     writer.writerow(["query_id", "doc_id", "relevance"])
#     for qrel in tqdm(argsme.qrels_iter(), desc="ArgsMe Qrels"):
#         writer.writerow([qrel.query_id, qrel.doc_id, qrel.relevance])




# # lotte/lifestyle/dev/forum
# import ir_datasets
# import csv
# from tqdm import tqdm

# # تحميل مجموعة wapo
# trec = ir_datasets.load("wapo/v2/trec-core-2018")

# # تصدير الوثائق
# with open("docs_wapo.csv", "w", newline='', encoding="utf-8", errors="replace") as f:
#     writer = csv.writer(f)
#     writer.writerow(["doc_id", "text"])
#     for doc in tqdm(trec.docs_iter(), desc="wapo Docs"):
#         writer.writerow([doc.doc_id, doc.text])

# # تصدير الاستعلامات
# with open("queries_wapo.csv", "w", newline='', encoding="utf-8", errors="replace") as f:
#     writer = csv.writer(f)
#     writer.writerow(["query_id", "title"])
#     for query in tqdm(trec.queries_iter(), desc="wapo Queries"):
#         writer.writerow([query.query_id, query.title])

# # تصدير ملفات التقييم
# with open("qrels_wapo.csv", "w", newline='', encoding="utf-8", errors="replace") as f:
#     writer = csv.writer(f)
#     writer.writerow(["query_id", "doc_id", "relevance"])
#     for qrel in tqdm(trec.qrels_iter(), desc="wapo Qrels"):
#         writer.writerow([qrel.query_id, qrel.doc_id, qrel.relevance])




# from datasets import load_dataset

# queries = load_dataset('irds/wapo_v2_trec-core-2018', 'queries')
# for record in queries:
#     record # {'query_id': ..., 'title': ..., 'description': ..., 'narrative': ...}

# qrels = load_dataset('irds/wapo_v2_trec-core-2018', 'qrels')
# for record in qrels:
#     record # {'query_id': ..., 'doc_id': ..., 'relevance': ..., 'iteration': ...}




# # Load the BEIR webis-touche2020/v2 dataset
# dataset = ir_datasets.load("beir/webis-touche2020/v2")

# # Save documents to CSV
# with open("docs_touche2020.csv", "w", newline='', encoding="utf-8", errors="replace") as f:
#     writer = csv.writer(f)
#     writer.writerow(["doc_id", "text"])
#     for doc in tqdm(dataset.docs_iter(), desc="Touche2020 Docs"):
#         writer.writerow([doc.doc_id, doc.text])

# # Save queries to CSV
# with open("queries_touche2020.csv", "w", newline='', encoding="utf-8", errors="replace") as f:
#     writer = csv.writer(f)
#     writer.writerow(["query_id", "text"])
#     for query in tqdm(dataset.queries_iter(), desc="Touche2020 Queries"):
#         writer.writerow([query.query_id, query.text])

# # Save qrels to CSV
# with open("qrels_touche2020.csv", "w", newline='', encoding="utf-8", errors="replace") as f:
#     writer = csv.writer(f)
#     writer.writerow(["query_id", "doc_id", "relevance"])
#     for qrel in tqdm(dataset.qrels_iter(), desc="Touche2020 Qrels"):
#         writer.writerow([qrel.query_id, qrel.doc_id, qrel.relevance])
