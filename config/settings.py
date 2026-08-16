import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path, override=True)

class Settings(BaseSettings):
    APP_NAME: str = "Multi-Agent Enterprise RAG Platform"
    HUGGINGFACEHUB_API_TOKEN: str = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
    LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "multi-agent-rag-chatbot")
    LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "true")
    
    PRIMARY_MODEL: str = "huggingface/Qwen/Qwen2.5-72B-Instruct"
    FALLBACK_MODEL: str = "huggingface/Qwen/Qwen2.5-Coder-32B-Instruct"
    
    FAISS_INDEX_PATH: str = "faiss_index"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

settings = Settings()
