import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from .config import ACTIONS_JSON_PATH, TOP_K

class ActionRAG:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        with open(ACTIONS_JSON_PATH, "r") as f:
            self.actions = json.load(f)

        self.texts = []
        self.mapping = []

        for idx, action in enumerate(self.actions):
            for ex in action["examples"]:
                self.texts.append(ex)
                self.mapping.append(idx)

        embeddings = self.embedder.encode(self.texts, normalize_embeddings=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(np.array(embeddings, dtype=np.float32))

    def retrieve(self, query):
        q = self.embedder.encode([query], normalize_embeddings=True)
        scores, idxs = self.index.search(np.array(q, dtype=np.float32), TOP_K)

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            action = self.actions[self.mapping[int(idx)]]
            results.append({
                "intent": action["intent"],
                "score": float(score)
            })
        return results
