import os
import json
import logging
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
def setup_vectorstore(chunks):
       try:
           embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
           logger.debug("HuggingFaceEmbeddings initialized")
           def convert_chunks_to_documents(chunks):
               return [Document(page_content=chunk["content"], metadata=chunk["meta_data"]) for chunk in chunks]
           documents = convert_chunks_to_documents(chunks)
           def embed_documents_in_batches(documents, batch_size=25):
               vectorstores = []
               for i in range(0, len(documents), batch_size):
                   batch = documents[i:i + batch_size]
                   logger.info(f"Embedding batch {i // batch_size + 1}/{(len(documents) + batch_size - 1) // batch_size}")
                   try:
                       vs = Chroma.from_documents(
                           documents=batch,
                           embedding=embeddings_model,
                           collection_name=f"pokemon_{uuid4()}"
                       )
                       vectorstores.append(vs)
                   except Exception as e:
                       logger.error(f"Failed batch {i // batch_size + 1}: {e}")
               return vectorstores
           embedded_vectorstores = embed_documents_in_batches(documents, batch_size=25)
           return embedded_vectorstores[0] if embedded_vectorstores else None
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

   # Initialize chatbot components
try:
       all_chunks = process_documents()
       combined_vectorstore = setup_vectorstore(all_chunks)
       retriever = setup_retriever(combined_vectorstore)
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

if __name__ == "__main__":
       logger.debug("Starting Uvicorn server")
       import uvicorn
       uvicorn.run(app, host="127.0.0.1", port=8000)