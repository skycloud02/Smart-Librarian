from typing import List, Dict
import  os, json
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "book_summaries")
BOOKS_JSON = os.getenv("BOOKS_JSON", "book_summaries.json")


#factory embedding function cu OpenAI
def _openai_embedder():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Lipsește OPENAI_API_KEY în .env")
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
    )

#incarca JSON și adauga in Chroma cu embeddings
def build_index():
    ef = _openai_embedder()
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    coll = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)

    with open(BOOKS_JSON, "r", encoding="utf-8") as f:
        books = json.load(f)

    ids, docs, metas = [], [], []
    for i, b in enumerate(books):
        title = (b.get("title") or "").strip()
        author = (b.get("author") or "").strip()
        genre = (b.get("genre") or "").strip()
        tags = (b.get("tags") or [])
        # Rezumatul (documentul pe care îl căutăm semantic):
        summary = (b.get("summary") or "").strip()
        if not title or not summary:
            continue
        ids.append(f"book-{i}")
        docs.append(summary)
        metas.append({
            "title": title,
            "author": author,
            "genre": genre,
            "tags": ";".join(tags) if isinstance(tags, list) else str(tags),
        })

    if ids:
        try:
            existing = coll.get(ids=ids)
            if existing and existing.get("ids"):
                coll.delete(ids=existing["ids"])
        except Exception:
            pass
        coll.add(ids=ids, documents=docs, metadatas=metas)
    return len(ids)

#retriever compatibil cu chat_core.ask_chat(...)
class ChromaRetriever:
    def __init__(self, k: int = 10):
        self.k = k
        ef = _openai_embedder()
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = client.get_or_create_collection(
            name=COLLECTION_NAME, embedding_function=ef
        )

    def query(self, q: str, k: int | None = None) -> List[Dict]:
        topk = k or self.k
        res = self.collection.query(
            query_texts=[q],
            n_results=topk,
            include=["documents", "metadatas", "distances"],
        )
        out = []
        if not res or not res.get("ids"):
            return out
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res.get("distances", [[ ]])[0]  # distanță; o convertim în „score”
        for doc, meta, dist in zip(docs, metas, dists):
            # transformăm distanța (mai mic = mai bun) într-un scor [0..1] aproximativ
            score = 1 / (1 + float(dist)) if dist is not None else 0.0
            out.append({
                "score": float(round(score, 4)),
                "meta": {
                    "title": meta.get("title", ""),
                    "author": meta.get("author", ""),
                    "genre": meta.get("genre", ""),
                    "tags": meta.get("tags", ""),
                    "info_url": "",
                },
                "doc": f"Title: {meta.get('title','')}\nAuthor: {meta.get('author','')}\n"
                       f"Genre: {meta.get('genre','')}\nSummary: {doc}\nTags: {meta.get('tags','')}",
            })
        return out

if __name__ == "__main__":
    n = build_index()
    print(f"Index construit: {n} documente în '{COLLECTION_NAME}' la {CHROMA_PATH}")