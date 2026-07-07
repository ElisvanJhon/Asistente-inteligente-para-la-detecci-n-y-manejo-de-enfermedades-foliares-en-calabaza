"""
Retriever: dado una clase detectada y una pregunta, busca los chunks más
relevantes en la base de conocimiento vectorial.

Usa el mismo modelo de embeddings LOCAL que build_kb.py (gratis, sin API).
"""

import chromadb
from chromadb.utils import embedding_functions

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "calabaza_enfermedades"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


class KnowledgeRetriever:
    def __init__(self):
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
        )

    def retrieve(self, query: str, clase: str | None = None, k: int = 3) -> list[str]:
        """
        Busca los k chunks más relevantes. Si se pasa `clase`, filtra
        primero por esa clase (para no traer info de otras enfermedades).
        """
        where = {"clase": clase} if clase and clase != "Healthy_Leaf" else None

        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=where,
        )
        return results["documents"][0] if results["documents"] else []
