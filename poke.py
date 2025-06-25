import os
from tqdm import tqdm
from rich.style import Style
from collections import deque
from functools import partial
from dotenv import load_dotenv
from rich.console import Console
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_cohere import CohereEmbeddings
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WikipediaLoader
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.runnables import RunnableLambda, RunnableAssign, RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


load_dotenv()
print(os.getenv('NVIDIA_API_KEY'))
os.environ["NVIDIA_API_KEY"] = os.getenv('NVIDIA_API_KEY')


console = Console()
base_style = Style(color="#76B900", bold=True)
pprint = partial(console.print, style=base_style)


def PPrint(preface="State: "):
    def print_and_return(x, preface=""):
        pprint(preface, x)
        return x
    return RunnableLambda(partial(print_and_return, preface=preface))


class WikipediaDocumentProcessor:
    def __init__(self, query, load_max_docs=10, chunk_size=800, chunk_overlap=100):
        self.query = query
        self.load_max_docs = load_max_docs
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.docs = []
        self.all_chunks = []
        
        # Use environment variable for API key instead of hardcoding
        cohere_api_key = os.getenv('COHERE_API_KEY')
        if not cohere_api_key:
            raise ValueError("COHERE_API_KEY environment variable is not set")
            
        self.embeddings_model = CohereEmbeddings(
            cohere_api_key=cohere_api_key,
            model="embed-english-v2.0"
        )
        self.vectorstore = None

    def load_documents(self):
        try:
            loader = WikipediaLoader(
                query=self.query, 
                load_max_docs=self.load_max_docs,
                doc_content_chars_max=10000  # Limit content size
            )
            self.docs = list(tqdm(loader.load(), desc="Loading docs"))
            print(f"{len(self.docs)} documents loaded from Wikipedia.")
            
            if not self.docs:
                print(f"No documents found for query: {self.query}")
                return False
            return True
            
        except Exception as e:
            print(f"Error loading documents: {e}")
            return False

    def create_chunks_with_headers(self, doc, doc_index):
        chunks = []
        start = 0
        doc_content = doc.page_content
        doc_length = len(doc.page_content)

        while start < doc_length:
            end = min(start + self.chunk_size, doc_length)
            chunk = doc_content[start:end]

            if start != 0:
                chunk = doc_content[max(start - self.chunk_overlap, 0):end]

            chunk_json = {
                "meta_data": {
                    "title": doc.metadata.get("title", "Unknown"),
                    "summary": doc.metadata.get('summary', "No summary available"),
                    "source_url": doc.metadata.get('source', "No source available"),
                },
                "chunk_index": len(chunks) + 1,
                "doc_index": doc_index,
                "content": chunk
            }
            chunks.append(chunk_json)

            start += self.chunk_size

        return chunks

    def process_and_create_embeddings(self):
        if not self.docs:
            print("No documents to process. Skipping embedding creation.")
            return False
            
        self.all_chunks = []
        for i, doc in enumerate(self.docs):
            chunks = self.create_chunks_with_headers(doc, i + 1)
            self.all_chunks.extend(chunks)

        if not self.all_chunks:
            print("No chunks created. Cannot create embeddings.")
            return False

        documents = [Document(page_content=chunk["content"],
                              metadata=chunk["meta_data"]) for chunk in self.all_chunks]

        try:
            # Filter out empty documents
            documents = [doc for doc in documents if doc.page_content.strip()]
            
            if not documents:
                print("All documents are empty after filtering.")
                return False
                
            print(f"Creating embeddings for {len(documents)} documents...")
            self.vectorstore = Chroma.from_documents(
                documents, self.embeddings_model)
            print("All data has been processed and stored in Chroma.")
            return True
            
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            return False

    def run(self):
        success = self.load_documents()
        if not success:
            return False
            
        success = self.process_and_create_embeddings()
        if success:
            print("All data has been processed successfully.")
        else:
            print("Failed to process data.")
        return success


class PokemonAssistant:
    def __init__(self, vectorstore):
        # Use environment variable for API key
        cohere_api_key = os.getenv('COHERE_API_KEY')
        if not cohere_api_key:
            raise ValueError("COHERE_API_KEY environment variable is not set")
            
        self.embeddings_model = CohereEmbeddings(
            cohere_api_key=cohere_api_key,
            model="embed-english-v2.0"
        )
        self.vectorstore = vectorstore
        self.initialize_retriever()
        
        # Store conversation history per session
        self.store = {}
        
        self.llm = ChatNVIDIA(
            model="mistralai/mixtral-8x22b-instruct-v0.1")

    def initialize_retriever(self):
        metadata_field_info = [
            AttributeInfo(
                name="title", description="The name of the article", type="string"),
            AttributeInfo(
                name="summary", description="The short summary of the article contents", type="string"),
            AttributeInfo(
                name="source_url", description="The web uri link to the article webpage", type="string"),
        ]
        document_content_description = "Data about pokemon, their types, abilities, and other related information."
        llm = ChatNVIDIA(
            model="mistralai/mistral-7b-instruct-v0.2") | StrOutputParser()
        
        try:
            self.retriever = SelfQueryRetriever.from_llm(
                llm, self.vectorstore, document_content_description, metadata_field_info)
        except Exception as e:
            print(f"Error initializing retriever: {e}")
            # Fallback to basic retriever
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def generate_embeddings(self, input_data):
        try:
            embeddings = self.retriever.invoke(input_data)
            if embeddings:
                # Only return the content, not metadata with URLs
                return [doc.page_content for doc in embeddings[:2]]
            return ["No relevant data available"]
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return ["No relevant data available"]

    def create_conversational_chain(self, session_id: str):
        # System message template
        system_message = """
You are a knowledgeable Pokemon assistant. Answer questions about Pokemon using the provided context and conversation history. 

Guidelines:
1. Answer directly and clearly based on the context provided
2. Use information from previous messages in the conversation when relevant
3. Be conversational and engaging
4. Do NOT mention source URLs, Wikipedia, or where the information comes from
5. If you don't have enough information in the context, say so politely
6. Keep responses concise but informative
7. Remember previous parts of the conversation to maintain context

Context Information: {context}

Answer the user's question naturally without mentioning sources.
"""

        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])

        # Create the chain
        chain = prompt | self.llm | StrOutputParser()

        # Wrap with message history
        conversational_chain = RunnableWithMessageHistory(
            chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )

        return conversational_chain

    def handle_user_input(self, user_input: str, session_id: str = "default"):
        try:
            # Get relevant context from embeddings
            context = self.generate_embeddings(user_input)
            context_text = "\n\n".join(context)
            
            # Create conversational chain
            chain = self.create_conversational_chain(session_id)
            
            # Get response with conversation history
            response = chain.invoke(
                {
                    "input": user_input,
                    "context": context_text
                },
                config={"configurable": {"session_id": session_id}}
            )
            
            return response
            
        except Exception as e:
            print(f"Error handling user input: {e}")
            return "Sorry, I encountered an error processing your request. Please try again."

    def clear_session_history(self, session_id: str = "default"):
        """Clear conversation history for a specific session"""
        if session_id in self.store:
            self.store[session_id].clear()
            
    def get_conversation_history(self, session_id: str = "default"):
        """Get conversation history for debugging purposes"""
        if session_id in self.store:
            return self.store[session_id].messages
        return []