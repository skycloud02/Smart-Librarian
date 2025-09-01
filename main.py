import chromadb
import json

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./chroma")
collection = client.get_or_create_collection(name="book_summaries")
SUMMARY_FILE = "book_summaries.json"

def load_book_summaries(path=SUMMARY_FILE):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def store_books():
    books = load_book_summaries()

    # Prepare separate lists for ids, summaries, and metadata (title)
    ids = []
    documents = []
    metadatas = []

    for idx, book in enumerate(books):
        title = book.get("title", "").strip()
        summary = book.get("summary", "").strip()

        # Sanity check
        if not title or not summary:
            print(f"⚠️ Skipping book at index {idx} due to missing title or summary.")
            continue

        ids.append(f"id{idx}")
        documents.append(summary)
        metadatas.append({"title": title})

    if not ids:
        print("❌ No valid book entries found.")
        return

    # Add to ChromaDB collection
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

    print(f"✅ Stored {len(documents)} documents in collection 'book_summaries'.")


def get_collection():
    get_collection = client.get_collection("book_summaries")

    # Get all documents
    data = get_collection.get()

    # Print stored documents
    for doc_id, doc_text, metadata in zip(data["ids"], data["documents"], data["metadatas"]):
        print(f"📘 {metadata['title']} (id: {doc_id}):\n{doc_text}\n")


if __name__ == "__main__":
    store_books()
    get_collection()
