import chromadb

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chromadb")
collection = chroma_client.get_or_create_collection(name="hatespeech")

# Add documents to the collection
collection.upsert(
    documents=[
        "[Target] are a plague and should be removed!", 
        "All followers of [Target] are dangerous!",
        "[Target] cannot do it right.", 
    ],
    metadatas=[
        {"Label": "Hatespeech", "Target": "Xenophobia"},
        {"Label": "Hatespeech", "Target": "Antisemitism"},
        {"Label": "Hatespeech", "Target": "Sexism"},
    ],
    ids=["id1", "id2", "id3"]
)

# Query collection (for testing purposes)
results = collection.query(
    query_texts=["All followers of [Target] are annoying!"],
    n_results=1
)
print(results)
