# search_gdocs.py

import os
import json
import numpy as np
from typing import List, Dict

from openai import OpenAI

EMBED_MODEL = "text-embedding-3-large"
EMBED_FILE = "gdocs_embeddings.jsonl"


def load_embeddings(path: str) -> List[Dict]:
    """
    Load JSONL records from the embeddings file.
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_matrix(records: List[Dict]):
    """
    Build a numpy matrix of embeddings and return (matrix, records).
    """
    emb_list = [r["embedding"] for r in records]
    emb_matrix = np.array(emb_list, dtype=np.float32)
    return emb_matrix


def embed_query(client: OpenAI, query: str) -> np.ndarray:
    """
    Get embedding vector for the query using OpenAI.
    """
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=[query],
    )
    vec = np.array(response.data[0].embedding, dtype=np.float32)
    return vec


def search(query: str, top_k: int = 5):
    client = OpenAI()

    print("Loading embeddings...")
    records = load_embeddings(EMBED_FILE)
    if not records:
        print(f"No records found in {EMBED_FILE}")
        return

    emb_matrix = build_matrix(records)
    print(f"Loaded {len(records)} chunks.")

    # Embed the query
    q_vec = embed_query(client, query)

    # Cosine similarity
    dot = emb_matrix @ q_vec
    norms = np.linalg.norm(emb_matrix, axis=1) * np.linalg.norm(q_vec)
    sims = dot / norms

    # Get top_k indices
    top_indices = sims.argsort()[::-1][:top_k]

    print()
    print(f"Top {top_k} results for query: {query}")
    print("-" * 80)

    for idx in top_indices:
        rec = records[idx]
        score = sims[idx]

        print(f"Doc: {rec['doc_name']}  (chunk {rec['chunk_index']})")
        print(f"Similarity: {score:.4f}")
        print(rec["text"][:400].replace("\n", " "))
        print("-" * 80)


if __name__ == "__main__":
    # simple CLI usage:
    # python search_gdocs.py
    # then type your query when prompted
    query = input("Enter your search query: ")
    search(query, top_k=5)
