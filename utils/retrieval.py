import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def create_embeddings(texts):
    return model.encode(texts, show_progress_bar=True)

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def search(index, query, ticket_texts, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [(ticket_texts[i], distances[0][j]) for j, i in enumerate(indices[0])]
