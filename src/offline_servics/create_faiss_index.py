import os
import faiss
import numpy as np
import joblib
import argparse
from tqdm import tqdm

def create_faiss_index(embeddings, index_file, dataset_name):
    """
    Create and save a faiss index for the given embeddings.

    Args:
        embeddings: numpy array of embeddings
        index_file: path to save the faiss index
        dataset_name: name of the dataset for progress display

    Returns:
        dict: status and message indicating success or error
    """
    try:
        print("Start creating FAISS index...")
        dimension = embeddings.shape[1]  # التحقق تلقائياً من عدد الأبعاد
        nlist = 1000
        m = 8

        print(f"Initializing faiss index with dimension={dimension}, nlist={nlist}, m={m}")
        print(f"FAISS version: {faiss.__version__}")
        print(f"IndexFlatL2 available: {hasattr(faiss, 'IndexFlatL2')}")

        try:
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, 8)
        except AttributeError:
            print("Falling back to index_factory due to missing IndexFlatL2")
            index = faiss.index_factory(dimension, f"IVF{nlist},PQ{m}", faiss.METRIC_L2)

        # شريط التقدم
        faiss_steps = ["Training", "Adding embeddings", "Saving index"]
        with tqdm(total=len(faiss_steps), desc=f"Creating FAISS index for {dataset_name}", unit="step") as faiss_pbar:
            print("Training FAISS index...")
            index.train(embeddings[:min(160000, len(embeddings))])  # تدريب على جزء إن لزم
            faiss_pbar.update(1)

            print("Adding embeddings to FAISS index...")
            index.add(embeddings)
            faiss_pbar.update(1)

            print("Saving FAISS index...")
            faiss.write_index(index, index_file)
            faiss_pbar.update(1)

        print("Finished creating FAISS index.")
        return {"status": "success", "message": "FAISS index created and saved"}
    except Exception as faiss_error:
        return {"status": "error", "message": f"FAISS index creation failed: {str(faiss_error)}"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a FAISS index from embeddings")
    parser.add_argument("--embeddings_file", type=str, required=True, help="Path to the embeddings file (.joblib)")
    parser.add_argument("--index_file", type=str, required=True, help="Path to save the FAISS index (.faiss)")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset for progress display")

    args = parser.parse_args()

    try:
        # تحميل التمثيلات
        embeddings = joblib.load(args.embeddings_file)
        print(f"Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")

        if not isinstance(embeddings, np.ndarray):
            raise ValueError("Loaded embeddings are not a numpy array.")
        if not np.all(np.isfinite(embeddings)):
            raise ValueError("Embeddings contain NaN or Inf values.")

        # التأكد من وجود مجلد الإخراج
        os.makedirs(os.path.dirname(args.index_file), exist_ok=True)

        # إنشاء الفهرس
        result = create_faiss_index(embeddings, args.index_file, args.dataset_name)
        print(result)
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)
