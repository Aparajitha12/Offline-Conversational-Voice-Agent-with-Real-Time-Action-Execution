import os
import sys
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from .config import ACTIONS_JSON_PATH, TOP_K


def get_models_path():
    """
    Returns correct models path.
    Works in both normal Python and PyInstaller .exe.
    """
    if getattr(sys, "frozen", False):
        base_path = sys._MEIPASS
    else:
        base_path = os.getcwd()

    return os.path.join(base_path, "models")


class ActionRAG:
    def __init__(self):
        model_dir = get_models_path()
        embed_model_path = os.path.join(model_dir, "all-MiniLM-L6-v2")

        if not os.path.isdir(embed_model_path):
            raise FileNotFoundError(
                f"Embedding model not found at {embed_model_path}.\n"
                "Download it once with internet ON and store inside models/."
            )

        # Load ONLY from local folder — no HuggingFace lookup
        self.embedder = SentenceTransformer(
            embed_model_path,
            device="cpu",
            model_kwargs={"local_files_only": True},
            tokenizer_kwargs={"local_files_only": True},
        )

        # Load action definitions
        with open(ACTIONS_JSON_PATH, "r", encoding="utf-8") as f:
            self.actions = json.load(f)

        self.texts = []
        self.mapping = []

        for idx, action in enumerate(self.actions):
            for ex in action["examples"]:
                self.texts.append(ex)
                self.mapping.append(idx)

        embeddings = self.embedder.encode(
            self.texts,
            normalize_embeddings=True
        )

        embeddings = np.array(embeddings, dtype=np.float32)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def retrieve(self, query):
        q_embedding = self.embedder.encode(
            [query],
            normalize_embeddings=True
        )

        q_embedding = np.array(q_embedding, dtype=np.float32)

        scores, idxs = self.index.search(q_embedding, TOP_K)

        results = []

        for score, idx in zip(scores[0], idxs[0]):
            action = self.actions[self.mapping[int(idx)]]
            results.append({
                "intent": action["intent"],
                "score": float(score)
            })

        return results