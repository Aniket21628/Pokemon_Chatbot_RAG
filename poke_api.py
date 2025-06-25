from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
import uvicorn
import os
from dotenv import load_dotenv
from poke import WikipediaDocumentProcessor, PokemonAssistant
import uuid

# Load environment variables
load_dotenv()

# Define your CORS policy
# Replace "*" with your specific origin(s) in production
CORS_POLICY = {
    "allow_origins": ["*"],
    "allow_credentials": True,
    "allow_methods": ["*"],
    "allow_headers": ["*"],
}

# Global variables
vector_store = None
pokemon_assistant = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store, pokemon_assistant
    print("Running initialization tasks before server starts.")
    
    try:
        # Try simple Pokemon query first
        print("Attempting to load Pokemon documents...")
        processor = WikipediaDocumentProcessor(query="Pokemon", load_max_docs=5)
        
        if not processor.run():
            print("Main Pokemon query failed, trying alternatives...")
            # Try alternative queries
            alternative_queries = [
                "Pokémon",
                "Pikachu", 
                "Pokemon games",
                "Nintendo Pokemon"
            ]
            
            success = False
            for alt_query in alternative_queries:
                print(f"Trying alternative query: {alt_query}")
                processor = WikipediaDocumentProcessor(query=alt_query, load_max_docs=3)
                if processor.run():
                    print(f"Success with query: {alt_query}")
                    success = True
                    break
            
            if not success:
                raise Exception("Failed to load any Pokemon documents from Wikipedia")
            
        vector_store = processor.vectorstore
        # Initialize the assistant once during startup
        pokemon_assistant = PokemonAssistant(vector_store)
        print("Pokemon initialization complete.")
        
    except Exception as e:
        print(f"Initialization failed: {e}")
        # Create a fallback vector store with Pokemon content
        try:
            processor = WikipediaDocumentProcessor(query="Pokemon", load_max_docs=3)
            processor.run()
            vector_store = processor.vectorstore
            pokemon_assistant = PokemonAssistant(vector_store)
            print("Pokemon fallback initialization complete.")
        except Exception as fallback_error:
            print(f"Pokemon fallback also failed: {fallback_error}")
            # Last resort - create minimal Pokemon content
            processor = WikipediaDocumentProcessor(query="Pikachu", load_max_docs=2)
            processor.run()
            vector_store = processor.vectorstore
            pokemon_assistant = PokemonAssistant(vector_store)
            print("Minimal Pokemon initialization complete.")
    
    yield
    print("Shutting Down.... Adios!!")

app = FastAPI(lifespan=lifespan)

# Add the CORSMiddleware to your FastAPI application
app.add_middleware(
    CORSMiddleware,
    **CORS_POLICY
)


class UserInput(BaseModel):
    text: str
    session_id: str = None  # Optional session ID for conversation continuity


class NewSessionResponse(BaseModel):
    session_id: str


@app.get("/api/v1/health")
async def health():
    return {"response": "Alive and well my friend !"}


@app.post("/api/v1/new-session", response_model=NewSessionResponse)
async def create_new_session():
    """Create a new conversation session"""
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}


@app.post("/query")
async def chat_endpoint(user_input: UserInput):
    try:
        global pokemon_assistant
        print(f"Received query: {user_input.text}")
        
        # Use provided session_id or default
        session_id = user_input.session_id or "default"
        
        response = pokemon_assistant.handle_user_input(user_input.text, session_id)
        return {"response": response, "session_id": session_id}
    except Exception as e:
        print(f"Error processing query: {e}")
        return {"response": "Sorry, I encountered an error processing your request. Please try again."}


@app.post("/api/v1/clear-session")
async def clear_session(request: dict):
    """Clear conversation history for a specific session"""
    try:
        global pokemon_assistant
        session_id = request.get("session_id", "default")
        pokemon_assistant.clear_session_history(session_id)
        return {"message": f"Session {session_id} cleared successfully"}
    except Exception as e:
        print(f"Error clearing session: {e}")
        return {"error": "Failed to clear session"}


@app.get("/api/v1/session-history/{session_id}")
async def get_session_history(session_id: str):
    """Get conversation history for debugging purposes"""
    try:
        global pokemon_assistant
        history = pokemon_assistant.get_conversation_history(session_id)
        return {"session_id": session_id, "message_count": len(history)}
    except Exception as e:
        print(f"Error getting session history: {e}")
        return {"error": "Failed to get session history"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)