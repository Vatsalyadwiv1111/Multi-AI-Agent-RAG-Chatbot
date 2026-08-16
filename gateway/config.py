import os
from dotenv import load_dotenv

# Ensure environment variables are loaded
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path, override=True)

# Map HuggingFace key for LiteLLM standard convention
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if hf_token:
    os.environ["HUGGINGFACE_API_KEY"] = hf_token

# Unified Model List for LiteLLM Router
MODEL_CONFIGS = [
    {
        "model_name": "primary-llm",
        "litellm_params": {
            "model": "huggingface/Qwen/Qwen2.5-72B-Instruct",
            "api_key": hf_token,
            "max_tokens": 512,
            "temperature": 0.1,
        }
    },
    {
        "model_name": "fallback-llm",
        "litellm_params": {
            "model": "huggingface/Qwen/Qwen2.5-Coder-32B-Instruct",
            "api_key": hf_token,
            "max_tokens": 512,
            "temperature": 0.1,
        }
    }
]

# If optional API keys exist, append them to fallback options dynamically
if os.getenv("GROQ_API_KEY"):
    MODEL_CONFIGS.append({
        "model_name": "fallback-groq",
        "litellm_params": {
            "model": "groq/llama-3.3-70b-versatile",
            "api_key": os.getenv("GROQ_API_KEY"),
            "max_tokens": 512
        }
    })

if os.getenv("OPENAI_API_KEY"):
    MODEL_CONFIGS.append({
        "model_name": "fallback-openai",
        "litellm_params": {
            "model": "openai/gpt-4o-mini",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "max_tokens": 512
        }
    })

# Router Configuration
ROUTER_FALLBACKS = [
    {"primary-llm": ["fallback-llm"]}
]

DEFAULT_NUM_RETRIES = 3
DEFAULT_TIMEOUT = 15
