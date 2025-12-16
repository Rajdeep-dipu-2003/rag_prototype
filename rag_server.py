from rag.Pipeline.rag import rag
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question:str

@app.get("/")
async def root():
    return {"text":"Hello world"}

@app.post("/ask")
async def ask_rag(query: Query):
    answer = rag.chat(query.question)
    return {"answer" : answer}

