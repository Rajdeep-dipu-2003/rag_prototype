import json
import uuid

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from rag.Ingestion.preprocess import load_json

INPUT_FILE = "rag/TimetableData/processed/clean_chunks.json"


def load_chunks():
    with open(INPUT_FILE, "r") as f:
        return json.load(f)

def safe(v):
    return "" if v is None else v
    
def get_metadata():
    data = load_json()

    metadatas = []

    for row in data:
        metadata = {
            "day": safe(row.get("day")),
            "slot": safe(row.get("slot")),
            "subject": safe(row.get("subject")),
            "subjectCode": safe(row.get("subjectCode")),
            "subjectFullName": safe(row.get("subjectFullName")),
            "subjectType": safe(row.get("subjectType")),
            "subjectCredit": safe(row.get("subjectCredit")),
            "faculty": safe(row.get("faculty")),
            "subjectDept": safe(row.get("subjectDept")),
            "offeringDept": safe(row.get("offeringDept")),
            "year": safe(row.get("year")),
            "room": safe(row.get("room")),
            "sem": safe(row.get("sem")),
            "code": safe(row.get("code")),
            "session": safe(row.get("session")),
            "mergedClass": safe(row.get("mergedClass")),
            "created_at": safe(row.get("created_at")),
            "updated_at": safe(row.get("updated_at")),
            "degree": safe(row.get("degree"))
        }

        metadatas.append(metadata)
    
    return metadatas

def embed_chunks():
    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    chunks = load_chunks()

    texts = [row["text"] for row in chunks]

    embeddings = model.encode(texts, show_progress_bar=True)

    ## Initialize Chroma DB
    client = PersistentClient(path="rag/VectorStore/chroma")

    collection = client.get_or_create_collection(
        name="timetable_collection",
        metadata={"hnsw:space":"cosine"}
    )

    ids = [str(uuid.uuid4()) for _ in texts]

    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=get_metadata(),
        ids=ids
    )

    