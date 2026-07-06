import json
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from collections import defaultdict
import math
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Path Configuration (Standalone)
# ============================================================

# Get the directory where this file is located
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data_transformations"
CHROMA_DIR = BASE_DIR / "chroma_db"

# ============================================================
# STEP 1: Load Data
# ============================================================

with open(DATA_DIR / 'index_chunks_combined.json', 'r', encoding='utf-8') as f:
    index_chunks = json.load(f)

print(f"✅ Loaded {len(index_chunks)} chunks from index")

# ============================================================
# STEP 2: Prepare Data for Indexing
# ============================================================

ids = [chunk['doc_id'] for chunk in index_chunks]
documents = [chunk['question'] + "\n" + chunk['answer'] for chunk in index_chunks]
metadatas = [
    {
        'doc_id': chunk['doc_id'],
        'source_type': chunk['source_type'],
        'product': chunk.get('product', ''),
        'menu_area': chunk.get('menu_area', ''),
        'functional_area': chunk.get('functional_area', ''),
        'question': chunk['question'],
        'answer': chunk['answer']
    }
    for chunk in index_chunks
]

# ============================================================
# STEP 3: Initialize ChromaDB Client
# ============================================================

# Create chroma_db directory if it doesn't exist
CHROMA_DIR.mkdir(exist_ok=True)

client = chromadb.PersistentClient(path=str(CHROMA_DIR))

# ============================================================
# STEP 4: Create Dense (Semantic) Collection
# ============================================================

try:
    client.delete_collection("fio_account_dense")
except:
    pass

dense_ef = SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

dense_collection = client.get_or_create_collection(
    name="fio_account_dense",
    embedding_function=dense_ef,
    metadata={"hnsw:space": "cosine"}
)

dense_collection.add(
    ids=ids,
    documents=documents,
    metadatas=metadatas
)

print(f"✅ Dense collection indexed with {len(ids)} chunks")

# ============================================================
# STEP 5: Build BM25 Index
# ============================================================

class BM25:
    def __init__(self, documents, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.tokenized_docs = [doc.lower().split() for doc in documents]
        self.doc_lengths = [len(doc) for doc in self.tokenized_docs]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 1

        self.term_freqs = []
        doc_freq = defaultdict(int)

        for doc in self.tokenized_docs:
            term_freq = defaultdict(int)
            for term in doc:
                term_freq[term] += 1
            self.term_freqs.append(term_freq)
            for term in set(doc):
                doc_freq[term] += 1

        N = len(documents)
        self.idf = {}
        for term, freq in doc_freq.items():
            self.idf[term] = math.log((N - freq + 0.5) / (freq + 0.5) + 1)

    def get_scores(self, query):
        query_tokens = query.lower().split()
        scores = []

        for idx, doc in enumerate(self.tokenized_docs):
            score = 0
            doc_len = self.doc_lengths[idx]
            term_freq = self.term_freqs[idx]

            for term in query_tokens:
                if term in term_freq:
                    tf = term_freq[term]
                    idf = self.idf.get(term, 0)
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_length))
                    score += idf * (numerator / denominator)

            scores.append(score)

        return scores

bm25 = BM25(documents)
print(f"✅ BM25 index built with {len(documents)} documents")

# ============================================================
# STEP 6: Hybrid Search Functions
# ============================================================

def get_dense_results(query, top_k=20):
    results = dense_collection.query(
        query_texts=[query],
        n_results=top_k
    )
    return {doc_id: dist for doc_id, dist in zip(results['ids'][0], results['distances'][0])}

def get_sparse_results(query, top_k=20):
    scores = bm25.get_scores(query)
    doc_ids_scores = sorted(
        [(ids[i], scores[i]) for i in range(len(scores))],
        key=lambda x: x[1],
        reverse=True
    )[:top_k]
    return {doc_id: score for doc_id, score in doc_ids_scores}

def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    scores = {}

    for rank, (doc_id, _) in enumerate(dense_results.items(), 1):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)

    for rank, (doc_id, _) in enumerate(sparse_results.items(), 1):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def hybrid_search(query, top_k=5, dense_top_k=20, sparse_top_k=20):
    dense_results = get_dense_results(query, top_k=dense_top_k)
    sparse_results = get_sparse_results(query, top_k=sparse_top_k)

    fused_scores = reciprocal_rank_fusion(dense_results, sparse_results)
    top_doc_ids = [doc_id for doc_id, _ in fused_scores[:top_k]]

    final_results = dense_collection.get(ids=top_doc_ids)

    results = []
    for doc_id, score in fused_scores[:top_k]:
        if doc_id in final_results['ids']:
            idx = final_results['ids'].index(doc_id)
            results.append({
                'doc_id': doc_id,
                'score': score,
                'metadata': final_results['metadatas'][idx],
                'document': final_results['documents'][idx] if final_results['documents'] else None
            })

    return results

# ============================================================
# STEP 7: Export for Import
# ============================================================

__all__ = [
    'index_chunks',
    'hybrid_search',
    'dense_collection',
    'bm25',
    'ids',
    'client'
]