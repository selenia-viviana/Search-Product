import os
import logging
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load API Keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "product-search")

# Initialize Pinecone Client
try:
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    index_list = pc.list_indexes().names()

    
    if INDEX_NAME not in index_list:
        logger.error(f"Index '{INDEX_NAME}' not found in Pinecone. Please create it first.")
        raise RuntimeError(f"Index '{INDEX_NAME}' does not exist.")

    index = pc.Index(INDEX_NAME)  
    logger.info(f"Connected to Pinecone index: {INDEX_NAME}")

except Exception as e:
    logger.error(f"Error initializing Pinecone: {e}")
    raise RuntimeError("Failed to connect to Pinecone.")

#Load Sentence Transformer Model
try:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Loaded SentenceTransformer model successfully.")
except Exception as e:
    logger.error(f"Error loading SentenceTransformer model: {e}")
    raise RuntimeError("Failed to load the embedding model.")

# Model for Query Request
class QueryRequest(BaseModel):
    query: str

# Search Class
class SemanticSearch:
    def __init__(self, index, model):
        self.index = index
        self.model = model

    def get_query_embedding(self, query: str):
        """Convert a query into an embedding."""
        return np.array(self.model.encode([query])).tolist()[0]

    def search_top_n(self, query: str, top_n: int = 10):
        """Retrieve the top N most relevant results."""
        try:
            query_vector = self.get_query_embedding(query)
            search_results = self.index.query(
                namespace="",
                vector=query_vector,
                top_k=top_n,
                include_metadata=True
            )

            if "matches" not in search_results:
                raise ValueError("No matches found in the index.")

            top_results = [
                {
                    "product_id": match["id"],
                    "metadata": match["metadata"],
                    "score": match["score"]
                }
                for match in search_results["matches"]
            ]
            return {"query": query, "top_results": top_results}

        except Exception as e:
            logger.error(f"Search error: {e}")
            raise HTTPException(status_code=500, detail="Error retrieving search results.")

# Initialize Search Engine
search_engine = SemanticSearch(index=index, model=embedding_model)

# Initialize FastAPI App
app = FastAPI(title="WANDS Semantic Search API", description="A microservice for semantic search using Pinecone and FastAPI.")

@app.get("/")
def read_root():
    """Root endpoint to check if API is running."""
    return {"message": "Pinecone FastAPI is running!"}

@app.post("/search/")
def search_similar_vectors(request: QueryRequest, top_n: int = Query(10, title="Top N results")):
    """
    Convert query to embedding and search Pinecone for similar vectors.
    """
    return search_engine.search_top_n(request.query, top_n)

@app.get("/health/")
def health_check():
    """Check if the service is running."""
    return {"status": "healthy", "message": "FastAPI service is up and running!"}
