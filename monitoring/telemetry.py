import os
import logging
import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register
from monitoring.config import PHOENIX_PORT, PHOENIX_HOST, LANGSMITH_PROJECT

logger = logging.getLogger("TelemetryMonitoring")

_telemetry_initialized = False

def setup_telemetry():
    """
    Initializes production observability:
    1. Launches Arize Phoenix local telemetry dashboard server at http://localhost:6006
    2. Registers OpenInference OTEL tracer for LangChain / LangGraph
    3. Confirms LangSmith tracing configuration
    """
    global _telemetry_initialized
    if _telemetry_initialized:
        return

    try:
        # 1. Launch Phoenix Telemetry App Server
        session = px.launch_app(host=PHOENIX_HOST, port=PHOENIX_PORT)
        logger.info(f"[Telemetry] Arize Phoenix Dashboard running at: http://localhost:{PHOENIX_PORT}")
        
        # 2. Register OpenTelemetry Tracer with Phoenix Provider
        tracer_provider = register(project_name=LANGSMITH_PROJECT)
        
        # 3. Instrument LangChain / LangGraph state graph calls
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
        
        _telemetry_initialized = True
        logger.info("[Telemetry] OpenInference OTEL instrumentation enabled for LangGraph & Gateway.")
    except Exception as e:
        logger.warning(f"[Telemetry] Phoenix initialization notice: {e}")
