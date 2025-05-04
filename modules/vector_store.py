import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.metadata = {}

    def build_index(self, text_chunks):
        vectors = self.embedder.encode(text_chunks).astype('float32')
        self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(vectors)
        self.metadata = {i: {"text": text} for i, text in enumerate(text_chunks)}

    def save_index(self, index_path, metadata_path):
        faiss.write_index(self.index, index_path)
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f)

    def load_index(self, index_path, metadata_path):
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

    def retrieve_context(self, query, top_k=5, subject=None):
        # Dynamically load subject-specific index/metadata if provided
        if subject:
            index_path = f"data/{subject.lower()}.index"
            metadata_path = f"data/{subject.lower()}_metadata.json"
            self.load_index(index_path, metadata_path)

        # Proceed with similarity search
        query_vector = self.embedder.encode(query).reshape(1, -1).astype('float32')
        _, indices = self.index.search(query_vector, top_k)
        context = "\n".join([self.metadata.get(str(i), {}).get("text", "") for i in indices[0]])
        return context
