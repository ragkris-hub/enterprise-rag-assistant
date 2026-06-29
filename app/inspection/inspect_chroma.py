import config
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def inspect_chroma():
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL
    )

    vector_store = Chroma(
        persist_directory=config.VECTOR_DB_PATH,
        embedding_function=embeddings
    )

    collection = vector_store._collection

    print("\n===== ChromaDB Inspection =====")
    print(f"Collection name: {collection.name}")
    print(f"Total records: {collection.count()}")

    data = collection.get(
        include=["documents", "metadatas", "embeddings"]
    )

    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])
    embeddings_data = data.get("embeddings", [])

    print("\n===== Stored Chunks =====")

    for index, document in enumerate(documents):
        print(f"\n--- Chunk {index + 1} ---")
        print("Content:")
        print(document[:500])

        print("\nMetadata:")
        print(metadatas[index])

        if embeddings_data is not None and len(embeddings_data) > index:
            embedding = embeddings_data[index]
            print("\nEmbedding:")
            print(f"Vector dimensions: {len(embedding)}")
            print(f"First 10 values: {embedding[:10]}")

    print("\n===== Inspection Complete =====")


if __name__ == "__main__":
    inspect_chroma()