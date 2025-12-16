from rag.Ingestion.preprocess import preprocess
from rag.Ingestion.embed import embed_chunks

def orchestrate():
    preprocess()
    embed_chunks()
    # initialize_model()
    # chat()
