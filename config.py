import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys (Add your own API keys)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    
    # Model configurations
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    LLM_MODEL = "gpt-3.5-turbo"  # Can be changed to other models
    
    # Database configurations
    VECTOR_DB_PATH = "data/vector_db"
    PDF_PATH = "data/pdfs"
    
    # Chunking parameters
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    # Retrieval parameters
    TOP_K_CHUNKS = 5
    SIMILARITY_THRESHOLD = 0.7
    
    # API configurations
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    
    # Memory configurations
    MAX_CONVERSATION_HISTORY = 10
