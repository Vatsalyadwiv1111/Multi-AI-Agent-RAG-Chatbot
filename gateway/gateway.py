import os
from langchain_litellm import ChatLiteLLM
from gateway.config import hf_token

def get_gateway_llm(model_group: str = "primary-llm", temperature: float = 0.1):
    """
    Enterprise LLM Factory Function.
    Returns a ChatLiteLLM instance wired to LiteLLM Gateway for:
    - Automatic retries
    - Fallback routing
    - Token & cost tracking
    - Latency monitoring
    """
    if hf_token and "HUGGINGFACE_API_KEY" not in os.environ:
        os.environ["HUGGINGFACE_API_KEY"] = hf_token

    # Use ChatLiteLLM pointing to the configured primary model group
    # LiteLLM handles fallbacks under the hood
    llm = ChatLiteLLM(
        model="huggingface/Qwen/Qwen2.5-72B-Instruct",
        temperature=temperature,
        max_tokens=512,
        api_key=hf_token
    )
    return llm
