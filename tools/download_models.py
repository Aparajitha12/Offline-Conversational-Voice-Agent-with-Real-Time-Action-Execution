import os
from huggingface_hub import snapshot_download

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
EMB_DIR = os.path.join(MODELS_DIR, "all-MiniLM-L6-v2")

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("Downloading SentenceTransformer model into:", EMB_DIR)

    snapshot_download(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        local_dir=EMB_DIR,
        local_dir_use_symlinks=False,
    )

    print("Done. You can now run fully offline.")

if __name__ == "__main__":
    main()