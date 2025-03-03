from fastapi import FastAPI, Query
from pinecone import Pinecone, ServerlessSpec
import os
import numpy as np
from sentence_transformers import SentenceTransformer 
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# Load API Key
  
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pc.Index("product-search")
class QueryRequest(BaseModel):
    query_embedding: list  

@app.get("/")
def read_root():
    return {"message": "Pinecone FastAPI is running!"}



# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

class QueryRequest(BaseModel):
    query: str  

@app.post("/search/")
def search_similar_vectors(request: QueryRequest, top_n: int = Query(10, title="Top N results")):
    """
    Convert query to embedding and search Pinecone for similar vectors.
    """
    # Converting query text to an embedding
    query_embedding = embedding_model.encode([request.query])
    query_vector = np.array(query_embedding).tolist()[0]  

    # Searching for similar vectors in Pinecone
    search_results = index.query(
        namespace="",  
        vector=query_vector,  
        top_k=top_n,
        include_metadata=True
    )

   
    top_results = [
        {
            "product_id": match.id,
            "metadata": match.metadata,
            "score": match.score
        }
        for match in search_results.matches
    ]

    return {"query": request.query, "top_results": top_results}