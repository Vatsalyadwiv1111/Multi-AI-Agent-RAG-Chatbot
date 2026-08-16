import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path, override=True)

LANGSMITH_PROJECT = os.getenv("LANGCHAIN_PROJECT", "multi-agent-rag-chatbot")
PHOENIX_PORT = 6006
PHOENIX_HOST = "127.0.0.1"
