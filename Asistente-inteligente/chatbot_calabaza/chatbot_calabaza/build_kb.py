"""
Construye la base de conocimiento vectorial (Chroma) a partir de los
documentos markdown en knowledge_base/.

Usa embeddings LOCALES (sentence-transformers) — no necesita API key ni
tarjeta, corre gratis en tu PC.

Correr UNA VEZ (o cada vez que edites los documentos):
    python build_kb.py
"""

from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

KB_DIR = Path("knowledge_base")
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "calabaza_enfermedades"
CHUNK_SIZE = 600          # caracteres aprox. por chunk
CHUNK_OVERLAP = 100

# Modelo local de embeddings, gratis, multilingüe (funciona bien en español)
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Mapea el nombre del archivo -> nombre de clase EXACTO que devuelve YOLO
FILENAME_TO_CLASS = {
    "bacterial_leaf_spot.md": "Bacterial_Leaf_Spot",
    "downy_mildew.md": "Downy_Mildew",
    "healthy_leaf.md": "Healthy_Leaf",
    "mosaic_disease.md": "Mosaic_Disease",
    "powdery_mildew.md": "Powdery_Mildew",
}


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split simple por párrafos, respetando el tamaño máximo de chunk."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for p in paragraphs:
        if len(current) + len(p) + 2 <= chunk_size:
            current = f"{current}\n\n{p}" if current else p
        else:
            if current:
                chunks.append(current)
            current = p
    if current:
        chunks.append(current)
    return chunks


def main():
    print(f"Cargando modelo de embeddings local ({EMBEDDING_MODEL})...")
    print("(la primera vez descarga el modelo, ~1 minuto, luego queda en caché)")

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    ids, documents, metadatas = [], [], []

    for md_file in sorted(KB_DIR.glob("*.md")):
        if md_file.name not in FILENAME_TO_CLASS:
            print(f"⚠️  Saltando {md_file.name} (no está en FILENAME_TO_CLASS)")
            continue

        clase = FILENAME_TO_CLASS[md_file.name]
        text = md_file.read_text(encoding="utf-8")
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

        for i, chunk in enumerate(chunks):
            ids.append(f"{clase}_{i}")
            documents.append(chunk)
            metadatas.append({"clase": clase, "fuente": md_file.name})

        print(f"✅ {md_file.name} -> {len(chunks)} chunks (clase: {clase})")

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"\nBase de conocimiento lista: {len(ids)} chunks totales")
    print(f"Guardada en: {CHROMA_DIR}/")


if __name__ == "__main__":
    main()
