import os
import json
import logging
import pickle
from pathlib import Path
from dotenv import load_dotenv
from functools import partial
from collections import deque
from datetime import datetime, timedelta
from uuid import uuid4
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.runnables import RunnableLambda, RunnableAssign, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.documents import Document
from langchain.chains.query_constructor.base import AttributeInfo
from rich.console import Console
from rich.style import Style

# Load environment variables from .env
load_dotenv()
NVIDIA_API_KEY_ENV_VAR = "NVIDIA_API_KEY"

# Cache configuration
CACHE_DIR = Path("./cache")
CHUNKS_CACHE_FILE = CACHE_DIR / "pokemon_chunks.json"
VECTORSTORE_CACHE_DIR = CACHE_DIR / "vectorstore"
CACHE_EXPIRY_DAYS = 7  # Cache expires after 7 days

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
logger.debug("Logging initialized")

# Initialize console for pretty printing
console = Console()
base_style = Style(color="#76B900", bold=True)
pprint = partial(console.print, style=base_style)

# Set NVIDIA API key
api_key = os.getenv(NVIDIA_API_KEY_ENV_VAR)
if not api_key:
    logger.error("NVIDIA_API_KEY environment variable not set")
    raise ValueError("Please set the NVIDIA_API_KEY environment variable")

# Cache management functions
def create_cache_dir():
    """Create cache directory if it doesn't exist"""
    CACHE_DIR.mkdir(exist_ok=True)
    logger.info(f"Cache directory created/verified at: {CACHE_DIR}")

def is_cache_valid():
    """Check if cache exists and is not expired"""
    if not CHUNKS_CACHE_FILE.exists() or not VECTORSTORE_CACHE_DIR.exists():
        return False
    
    # Check if cache is expired
    cache_time = datetime.fromtimestamp(CHUNKS_CACHE_FILE.stat().st_mtime)
    expiry_time = cache_time + timedelta(days=CACHE_EXPIRY_DAYS)
    
    if datetime.now() > expiry_time:
        logger.info("Cache expired, will rebuild")
        return False
    
    logger.info("Valid cache found")
    return True

def save_chunks_to_cache(chunks):
    """Save processed chunks to cache"""
    try:
        with open(CHUNKS_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        logger.info(f"Chunks saved to cache: {CHUNKS_CACHE_FILE}")
    except Exception as e:
        logger.error(f"Failed to save chunks to cache: {e}")
        raise

def load_chunks_from_cache():
    """Load processed chunks from cache"""
    try:
        with open(CHUNKS_CACHE_FILE, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        logger.info(f"Loaded {len(chunks)} chunks from cache")
        return chunks
    except Exception as e:
        logger.error(f"Failed to load chunks from cache: {e}")
        raise

def clear_cache():
    """Clear all cache files"""
    try:
        if CHUNKS_CACHE_FILE.exists():
            CHUNKS_CACHE_FILE.unlink()
        if VECTORSTORE_CACHE_DIR.exists():
            import shutil
            shutil.rmtree(VECTORSTORE_CACHE_DIR)
        logger.info("Cache cleared")
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")

# Load and process Wikipedia data
def load_wikipedia_data():
    try:
        docs = WikipediaLoader(query="Pokemon and Everything about Pokemon", load_max_docs=100).load()
        logger.info(f"Loaded {len(docs)} Wikipedia documents")
        return docs
    except Exception as e:
        logger.error(f"Failed to load Wikipedia data: {e}")
        raise

def create_chunks_with_headers(doc, doc_index):
    chunk_size = 800
    chunk_overlap = 100
    chunks = []
    start = 0
    doc_content = doc.page_content
    doc_length = len(doc.page_content)

    while start < doc_length:
        end = min(start + chunk_size, doc_length)
        chunk = doc_content[start:end]
        if start != 0:
            chunk = doc_content[max(start - chunk_overlap, 0):end]
        chunk_json = {
            "meta_data": {
                "title": doc.metadata["title"],
                "summary": doc.metadata['summary'],
                "source_url": doc.metadata['source'],
            },
            "chunk_index": len(chunks) + 1,
            "content": chunk
        }
        chunks.append(chunk_json)
        start += chunk_size
    return chunks

def process_documents():
    docs = load_wikipedia_data()
    all_chunks = []
    for i, doc in enumerate(docs):
        chunks = create_chunks_with_headers(doc, i+1)
        all_chunks.extend(chunks)
        logger.info(f"Created {len(chunks)} chunks for document {i+1}")
    logger.info("All data has been processed")
    return all_chunks

# Setup embeddings and vectorstore
def setup_vectorstore(chunks, use_cache=True):
    try:
        embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        logger.debug("HuggingFaceEmbeddings initialized")
        
        # Check if vectorstore cache exists
        if use_cache and VECTORSTORE_CACHE_DIR.exists() and any(VECTORSTORE_CACHE_DIR.iterdir()):
            logger.info("Loading vectorstore from cache")
            vectorstore = Chroma(
                persist_directory=str(VECTORSTORE_CACHE_DIR),
                embedding_function=embeddings_model
            )
            return vectorstore
        
        def convert_chunks_to_documents(chunks):
            return [Document(page_content=chunk["content"], metadata=chunk["meta_data"]) for chunk in chunks]
        
        documents = convert_chunks_to_documents(chunks)
        
        def embed_documents_in_batches(documents, batch_size=25):
            logger.info("Creating new vectorstore...")
            # Create vectorstore with persistence (auto-persists with persist_directory)
            vectorstore = None
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                logger.info(f"Embedding batch {i // batch_size + 1}/{(len(documents) + batch_size - 1) // batch_size}")
                try:
                    if vectorstore is None:
                        # Create the first vectorstore with persistence
                        vectorstore = Chroma.from_documents(
                            documents=batch,
                            embedding=embeddings_model,
                            persist_directory=str(VECTORSTORE_CACHE_DIR)
                        )
                    else:
                        # Add to existing vectorstore
                        vectorstore.add_documents(batch)
                except Exception as e:
                    logger.error(f"Failed batch {i // batch_size + 1}: {e}")
                    raise
            
            # Vectorstore automatically persists when persist_directory is specified
            if vectorstore:
                logger.info("Vectorstore created and automatically persisted to cache")
            
            return vectorstore
        
        embedded_vectorstore = embed_documents_in_batches(documents, batch_size=25)
        return embedded_vectorstore
        
    except Exception as e:
        logger.error(f"Failed to setup vectorstore: {e}")
        raise

# Setup retriever
def setup_retriever(vectorstore):
    try:
        llm = ChatNVIDIA(model="mistralai/mistral-7b-instruct-v0.2", api_key=api_key)
        logger.debug("ChatNVIDIA initialized for retriever")
        metadata_field_info = [
            AttributeInfo(name="title", description="The name of the article", type="string"),
            AttributeInfo(name="summary", description="The short summary of the article contents", type="string"),
            AttributeInfo(name="source_url", description="The web URI link to the article webpage", type="string"),
        ]
        document_content_description = "Data about Pokemon"
        retriever = SelfQueryRetriever.from_llm(
            llm, vectorstore, document_content_description, metadata_field_info
        )
        return retriever
    except Exception as e:
        logger.error(f"Failed to setup retriever: {e}")
        raise

# Initialize chatbot components with caching
def initialize_chatbot_components():
    try:
        create_cache_dir()
        
        if is_cache_valid():
            logger.info("Using cached data...")
            # Load from cache
            all_chunks = load_chunks_from_cache()
            combined_vectorstore = setup_vectorstore(all_chunks, use_cache=True)
        else:
            logger.info("Processing new data...")
            # Process fresh data
            all_chunks = process_documents()
            save_chunks_to_cache(all_chunks)
            combined_vectorstore = setup_vectorstore(all_chunks, use_cache=False)
        
        retriever = setup_retriever(combined_vectorstore)
        return retriever
        
    except Exception as e:
        logger.error(f"Failed to initialize chatbot components: {e}")
        raise

# Initialize components
try:
    retriever = initialize_chatbot_components()
    logger.info("Chatbot components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize chatbot components: {e}")
    raise

memory = deque(maxlen=5)

def update_memory(user_question, response):
    memory.append({"question": user_question, "response": response})
    pprint(memory)
    return memory

sys_msg = """
You are an intelligent assistant that answers all questions about Pokemon using contextual information from Wikipedia. Your responses should be conversational and informative, providing clear and concise explanations. When relevant, include the source URL of the articles to give users additional reading material.

Always aim to:
1. Answer the question directly and clearly.
2. Provide context and background information when useful but do not give irrelevant information and answer to the point.
3. Suggest related topics or additional points of interest.
4. Be polite and engaging in your responses.
5. Remove the unnecessary context from the context provided if irrelevant to the question

Now, let's get started!
"""

try:
    instruct_chat = ChatNVIDIA(model="meta/llama3-70b-instruct", api_key=api_key)
    llm = instruct_chat | StrOutputParser()
    logger.debug("ChatNVIDIA initialized for LLM")
except Exception as e:
    logger.error(f"Failed to initialize LLM: {e}")
    raise

def generate_embeddings(input_data):
    try:
        embeddings = retriever.invoke(input_data)
        return embeddings if embeddings else "No data available"
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        return "No data available"

def generate_embeddings_query(input_data):
    try:
        prompt = ChatPromptTemplate.from_template(
            """
            User's Question: {input}
            Previous conversation memory {memory}
            Generate only a query sentence and nothing else from the user's question to fetch from the data from embeddings. If the user's question does not have enough context then create a query based on the Knowledge Base.
            """
        )
        embedding_chain = prompt | llm
        embeddings_query = embedding_chain.invoke({"input": input_data, "memory": memory})
        return embeddings_query if embeddings_query else "Process failed"
    except Exception as e:
        logger.error(f"Failed to generate embeddings query: {e}")
        return "Process failed"

def get_response(prompt):
    try:
        return llm.invoke(prompt)
    except Exception as e:
        logger.error(f"Failed to get response: {e}")
        return "Error generating response"

# Chatbot runnable
def create_runnable():
    generate_embeddings_runnable = RunnableLambda(generate_embeddings)
    generate_embeddings_query_runnable = RunnableLambda(generate_embeddings_query)
    runnable = (
        {"input": RunnablePassthrough(), "memory": RunnablePassthrough()}
        | RunnableAssign({"embedding_query": generate_embeddings_query_runnable})
        | RunnableAssign({"context": generate_embeddings_runnable})
        | RunnableAssign({"prompt": lambda x: ChatPromptTemplate.from_template(
            f"""
            {sys_msg}

            User's Question: {{input}}

            Context Information: {{context}}

            Previous Conversation memory: {{memory}}

            Your Response:
            """
        )})
        | RunnableAssign({"response": lambda x: get_response(x["prompt"])})
        | RunnableAssign({"memory": lambda x: update_memory(x["input"]["input"], x["response"])})
    )
    return runnable

try:
    runnable = create_runnable()
except Exception as e:
    logger.error(f"Failed to create runnable: {e}")
    raise

# FastAPI setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

@app.post("/query", response_model=QueryResponse)
async def query_chatbot(request: QueryRequest):
    try:
        response = runnable.invoke({"input": request.query, "memory": memory})
        return QueryResponse(response=response["response"])
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return QueryResponse(response="Error processing your request")

@app.post("/clear-cache")
async def clear_cache_endpoint():
    """Endpoint to manually clear cache and force rebuild on next startup"""
    try:
        clear_cache()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return {"error": "Failed to clear cache"}

@app.get("/cache-status")
async def cache_status():
    """Endpoint to check cache status"""
    try:
        cache_valid = is_cache_valid()
        cache_exists = CHUNKS_CACHE_FILE.exists() and VECTORSTORE_CACHE_DIR.exists()
        
        cache_info = {
            "cache_valid": cache_valid,
            "cache_exists": cache_exists,
            "cache_dir": str(CACHE_DIR),
            "chunks_file": str(CHUNKS_CACHE_FILE),
            "vectorstore_dir": str(VECTORSTORE_CACHE_DIR)
        }
        
        if CHUNKS_CACHE_FILE.exists():
            cache_time = datetime.fromtimestamp(CHUNKS_CACHE_FILE.stat().st_mtime)
            cache_info["cache_created"] = cache_time.isoformat()
            cache_info["expires"] = (cache_time + timedelta(days=CACHE_EXPIRY_DAYS)).isoformat()
        
        return cache_info
    except Exception as e:
        logger.error(f"Error getting cache status: {e}")
        return {"error": "Failed to get cache status"}

if __name__ == "__main__":
    logger.debug("Starting Uvicorn server")
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)