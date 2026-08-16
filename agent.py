import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from typing import Annotated, List, Literal, TypedDict

# ---------------------------------------------------------------------------
# Setup (unchanged from the original single-agent version)
# ---------------------------------------------------------------------------
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path, override=True)

from monitoring import setup_telemetry
setup_telemetry()


def validate_api_key():
    """Validates and retrieves the Hugging Face API Token."""
    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if not api_token or api_token == "your_hf_token_here":
        try:
            import streamlit as st
            if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
                api_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
            elif "HF_TOKEN" in st.secrets:
                api_token = st.secrets["HF_TOKEN"]
        except Exception:
            pass

    if not api_token or api_token == "your_hf_token_here":
        return None

    os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_token
    return api_token


_api_token = validate_api_key()


def validate_langsmith_config():
    """Enables LangSmith tracing if a key is present. Optional — the app
    runs fine without it, you just won't get traces in the dashboard."""
    api_key = os.getenv("LANGCHAIN_API_KEY")

    if not api_key:
        try:
            import streamlit as st
            if "LANGCHAIN_API_KEY" in st.secrets:
                api_key = st.secrets["LANGCHAIN_API_KEY"]
        except Exception:
            pass

    if not api_key:
        print("LangSmith tracing disabled (no LANGCHAIN_API_KEY set).")
        return False

    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "multi-agent-rag-chatbot")
    print(f"LangSmith tracing enabled (project: {os.environ['LANGCHAIN_PROJECT']}).")
    return True


_langsmith_enabled = validate_langsmith_config()

# Import tools
from tools.pdf_processor import load_pdf
from tools.wiki_search import search_wikipedia
from tools.url_retriever import retrieve_url_content


class State(TypedDict):
    messages: Annotated[List, add_messages]
    context_files: List[str]        # List of file paths or identifiers
    context_urls: List[str]         # List of URLs
    next: str                       # Which specialist the supervisor routed to


# Only the research agent needs a real bindable tool. The RAG agent gets its
# context by retrieving from the vector store directly (see get_retriever),
# same mechanism the original single agent used.
@tool
def wiki_tool(query: str):
    """Searches Wikipedia for a query."""
    return search_wikipedia(query)


research_tools = [wiki_tool]

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# LLM Gateway Integration — Agents request model through LiteLLM Gateway
# ---------------------------------------------------------------------------
from gateway import get_gateway_llm

try:
    if _api_token is None:
        print("Warning: HuggingFace API Token is missing. Gateway will reject requests.")
        chat_model = None
    else:
        chat_model = get_gateway_llm()
except Exception as e:
    print(f"Error initializing Gateway LLM: {e}")
    chat_model = None

# Only the research agent gets tools bound to it
research_llm = chat_model.bind_tools(research_tools) if chat_model else None

# ---------------------------------------------------------------------------
# Shared RAG retriever (identical logic to before, now used only by rag_agent)
# ---------------------------------------------------------------------------
vector_store = None
current_files_hash = ""


def get_retriever(files, urls):
    """Creates or updates a vector store from the provided files and URLs."""
    global vector_store, current_files_hash

    new_hash = str(sorted(files)) + str(sorted(urls))

    if vector_store is not None and new_hash == current_files_hash:
        return vector_store.as_retriever(search_kwargs={"k": 3})

    documents = []

    if files:
        for file_path in files:
            try:
                docs = load_pdf(file_path)
                for doc in docs:
                    doc.metadata["source"] = f"PDF - {os.path.basename(file_path)}"
                documents.extend(docs)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    if urls:
        for url in urls:
            try:
                docs = retrieve_url_content(url)
                for doc in docs:
                    doc.metadata["source"] = f"URL - {url}"
                documents.extend(docs)
            except Exception as e:
                print(f"Error reading {url}: {e}")

    if not documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(splits, embeddings)
    current_files_hash = new_hash

    return vector_store.as_retriever(search_kwargs={"k": 3})


def _get_user_query(state: State) -> str:
    """Pulls the latest human message out of state, handling both message
    object and dict-style history entries."""
    messages = state.get("messages", [])
    if not messages:
        return ""
    last_msg = messages[-1]
    if isinstance(last_msg, HumanMessage):
        return last_msg.content
    if isinstance(last_msg, dict) and last_msg.get("role") == "user":
        return last_msg.get("content", "")
    return ""


# ---------------------------------------------------------------------------
# SUPERVISOR — classifies the query and routes it to one specialist agent.
# ---------------------------------------------------------------------------
def supervisor(state: State):
    """Decides whether the query goes to the RAG, research, or chat agent."""
    files = state.get("context_files", [])
    urls = state.get("context_urls", [])
    user_query = _get_user_query(state)

    # Fast-path: if the user has loaded documents/URLs, assume the question
    # is about them — this mirrors the original app's always-inject-context
    # behaviour and avoids an extra LLM call on every turn.
    if files or urls:
        return {"next": "rag_agent"}

    if not chat_model:
        # No model available — route to chat_agent, which will surface a
        # clear error message rather than crashing the graph.
        return {"next": "chat_agent"}

    classification_prompt = f"""Classify the user's message into exactly one category.
Reply with only one word, nothing else.

"wiki" -> factual or general-knowledge questions that need looking something up
"chat" -> greetings, jokes, stories, opinions, or casual conversation

Message: "{user_query}"
Category:"""

    try:
        decision = chat_model.invoke([SystemMessage(content=classification_prompt)])
        label = str(decision.content).strip().lower()
    except Exception as e:
        print(f"Supervisor classification failed, defaulting to chat: {e}")
        label = "chat"

    next_agent = "research_agent" if "wiki" in label else "chat_agent"
    return {"next": next_agent}


def route_from_supervisor(state: State) -> Literal["rag_agent", "research_agent", "chat_agent"]:
    return state.get("next", "chat_agent")


# ---------------------------------------------------------------------------
# RAG AGENT — answers from uploaded PDFs / URLs.
# ---------------------------------------------------------------------------
RAG_SYSTEM_PROMPT = """You are a document-retrieval specialist inside a multi-agent chatbot.
Answer strictly using the retrieved context below. Do not invent information.
If the answer isn't in the context, say so clearly.
Always cite the source at the end, e.g. "Source: PDF - filename.pdf" or "Source: URL - https://...".
"""


def rag_agent(state: State):
    if not chat_model:
        return {"messages": [AIMessage(content="Error: Language Model not initialized. Please check your API Token.")]}

    files = state.get("context_files", [])
    urls = state.get("context_urls", [])
    user_query = _get_user_query(state)

    context_block = "\n\n--- CONTEXT START ---\n"
    retriever = get_retriever(files, urls)
    if retriever and user_query:
        relevant_docs = retriever.invoke(user_query)
        for doc in relevant_docs:
            source = doc.metadata.get("source", "Unknown")
            context_block += f"\n[Source: {source}]\n{doc.page_content}\n"
    else:
        context_block += "\nNo documents available to search.\n"
    context_block += "\n--- CONTEXT END ---\n"

    system_prompt = RAG_SYSTEM_PROMPT + context_block
    messages = [SystemMessage(content=system_prompt)] + state["messages"]

    try:
        response = chat_model.invoke(messages)
    except Exception as e:
        return {"messages": [AIMessage(content=f"Error invoking model: {str(e)}")]}
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# RESEARCH AGENT — Wikipedia specialist, runs its own tool-call loop.
# ---------------------------------------------------------------------------
RESEARCH_SYSTEM_PROMPT = """You are a research specialist inside a multi-agent chatbot.
Use the wiki_tool to look up facts on Wikipedia before answering general-knowledge questions.
Be accurate and concise. Do not invent information."""


def research_agent(state: State):
    if not research_llm:
        return {"messages": [AIMessage(content="Error: Language Model not initialized. Please check your API Token.")]}

    messages = [SystemMessage(content=RESEARCH_SYSTEM_PROMPT)] + state["messages"]
    try:
        response = research_llm.invoke(messages)
    except Exception as e:
        return {"messages": [AIMessage(content=f"Error invoking model: {str(e)}")]}
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# CHAT AGENT — open-ended conversation, no tools.
# ---------------------------------------------------------------------------
CHAT_SYSTEM_PROMPT = """You are the conversational specialist inside a multi-agent chatbot.
Be highly engaging, witty, and personable for jokes, stories, greetings, and open-ended chat.
You are not a robot — show personality!"""


def chat_agent(state: State):
    if not chat_model:
        return {"messages": [AIMessage(content="Error: Language Model not initialized. Please check your API Token.")]}

    messages = [SystemMessage(content=CHAT_SYSTEM_PROMPT)] + state["messages"]
    try:
        response = chat_model.invoke(messages)
    except Exception as e:
        return {"messages": [AIMessage(content=f"Error invoking model: {str(e)}")]}
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Build the graph:
#   START -> supervisor -> { rag_agent | research_agent (+tools loop) | chat_agent } -> END
# ---------------------------------------------------------------------------
graph_builder = StateGraph(State)

graph_builder.add_node("supervisor", supervisor)
graph_builder.add_node("rag_agent", rag_agent)
graph_builder.add_node("research_agent", research_agent)
graph_builder.add_node("research_tools", ToolNode(research_tools))
graph_builder.add_node("chat_agent", chat_agent)

graph_builder.add_edge(START, "supervisor")
graph_builder.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {"rag_agent": "rag_agent", "research_agent": "research_agent", "chat_agent": "chat_agent"},
)

# RAG and chat agents answer directly and finish
graph_builder.add_edge("rag_agent", END)
graph_builder.add_edge("chat_agent", END)

# Research agent runs its own tool-call loop before finishing
graph_builder.add_conditional_edges("research_agent", tools_condition, {"tools": "research_tools", END: END})
graph_builder.add_edge("research_tools", "research_agent")

graph = graph_builder.compile()


# ---------------------------------------------------------------------------
# Example usage function — SAME SIGNATURE as before, so app.py needs no changes.
# ---------------------------------------------------------------------------
def run_agent(input_text, files=None, urls=None, thread_id="1"):
    import time
    from guardrails import validate_input, validate_output
    from evaluation import evaluate_agent_response

    start_time = time.time()

    # 1. Execute Input Guardrails Check
    input_check = validate_input(input_text)
    if not input_check.is_valid:
        return f"🚨 Guardrail Alert ({input_check.violation_type}): {input_check.reason}"

    sanitized_input = input_check.sanitized_text

    config = {
        "configurable": {"thread_id": thread_id},
        "run_name": "multi-agent-rag-chatbot",
        "tags": ["multi-agent-rag-chatbot"],
        "metadata": {
            "thread_id": thread_id,
            "has_files": bool(files),
            "has_urls": bool(urls),
        },
    }
    initial_state = {
        "messages": [HumanMessage(content=sanitized_input)],
        "context_files": files or [],
        "context_urls": urls or [],
        "next": "",
    }

    events = graph.stream(initial_state, config=config)
    final_response = ""
    for event in events:
        for node_name in ("rag_agent", "research_agent", "chat_agent"):
            if node_name in event:
                message = event[node_name]["messages"][-1]
                content = message.content

                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and "text" in part:
                            text_parts.append(part["text"])
                        elif isinstance(part, str):
                            text_parts.append(part)
                    if text_parts:
                        final_response = "".join(text_parts)
                elif content:
                    final_response = str(content)

    # 2. Execute Output Guardrails Check
    output_check = validate_output(final_response)
    
    # 3. Execute LLM Evaluation Layer
    elapsed_seconds = time.time() - start_time
    eval_result = evaluate_agent_response(
        query=sanitized_input,
        response=output_check.sanitized_text,
        context_docs=[],
        latency_seconds=elapsed_seconds
    )
    print(f"[EVALUATION] {eval_result.summary}")

    return output_check.sanitized_text
