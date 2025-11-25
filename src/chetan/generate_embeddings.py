import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def load_extracted_texts(input_file):
    # Load the extracted texts from JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def generate_embeddings(texts_data, model_name='all-MiniLM-L6-v2'):
    # Generate embeddings for all documents
    print(f"Loading model: {model_name}\n")
    model = SentenceTransformer(model_name)

    texts = [doc['text'] for doc in texts_data]

    print(f"Generating embeddings for {len(texts)} documents")
    embeddings = model.encode(texts, show_progress_bar=True)

    return embeddings

def save_embeddings(embeddings, output_dir):
    # Save embeddings
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(output_path / "embeddings.npy", embeddings)

    print(f"Saved embeddings: shape {embeddings.shape}")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    input_file = "data/processed/extracted_texts.json"
    output_dir = "data/processed"

    texts_data = load_extracted_texts(input_file)

    embeddings = generate_embeddings(texts_data)

    save_embeddings(embeddings, output_dir)
