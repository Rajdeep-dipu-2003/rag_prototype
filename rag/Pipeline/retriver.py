from chromadb import PersistentClient

def get_documents(question):
    
    client = PersistentClient(path="rag/VectorStore/chroma")

    collection = client.get_or_create_collection(
        name="timetable_collection",
        metadata={"hnsw:space":"cosine"}
    )

    return collection.query(
        query_texts=[question], 
        n_results=100              
    )