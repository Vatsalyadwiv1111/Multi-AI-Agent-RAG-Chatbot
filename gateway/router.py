import time
import logging
from litellm import Router, success_callback, failure_callback
from gateway.config import MODEL_CONFIGS, ROUTER_FALLBACKS, DEFAULT_NUM_RETRIES, DEFAULT_TIMEOUT

# Setup structured logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LLMGateway")

# In-memory metrics store for enterprise monitoring
gateway_metrics = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "total_tokens": 0,
    "estimated_cost_usd": 0.0,
    "last_latency_seconds": 0.0,
    "retries_count": 0
}

# Define LiteLLM Telemetry Callbacks
def custom_success_callback(kwargs, completion_response, start_time, end_time):
    latency = end_time - start_time
    gateway_metrics["total_requests"] += 1
    gateway_metrics["successful_requests"] += 1
    gateway_metrics["last_latency_seconds"] = round(latency, 3)
    
    # Track usage if returned in response
    usage = getattr(completion_response, "usage", None)
    if usage:
        gateway_metrics["total_tokens"] += getattr(usage, "total_tokens", 0)
    
    # Calculate estimated cost if available
    cost = kwargs.get("response_cost", 0.0) or 0.0
    gateway_metrics["estimated_cost_usd"] += cost
    
    logger.info(f"[Gateway SUCCESS] Model: {kwargs.get('model')} | Latency: {latency:.2f}s | Tokens: {usage.total_tokens if usage else 'N/A'}")

def custom_failure_callback(kwargs, completion_response, start_time, end_time):
    latency = end_time - start_time
    gateway_metrics["total_requests"] += 1
    gateway_metrics["failed_requests"] += 1
    logger.error(f"[Gateway FAILURE] Model: {kwargs.get('model')} | Error: {kwargs.get('exception')}")

# Register callbacks with LiteLLM
success_callback.append(custom_success_callback)
failure_callback.append(custom_failure_callback)

# Initialize the Router
gateway_router = Router(
    model_list=MODEL_CONFIGS,
    fallbacks=ROUTER_FALLBACKS,
    num_retries=DEFAULT_NUM_RETRIES,
    timeout=DEFAULT_TIMEOUT
)

def get_gateway_router():
    """Returns the singleton LiteLLM Router instance."""
    return gateway_router
