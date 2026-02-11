import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from typing import Annotated, List, TypedDict

# Load environment variables
# Force loading from the same directory as this script
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

# Verify API Key
# API Key Handling
def validate_api_key():
    """Validates and retrieves the Hugging Face API Token."""
    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if not api_token:
        try:
            import streamlit as st
            if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
                api_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
            elif "HF_TOKEN" in st.secrets:
                api_token = st.secrets["HF_TOKEN"]
        except:
            pass

    if not api_token:
        # Return None instead of raising immediately to allow UI to handle it
        return None
    
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_token
    return api_token

# Attempt verification on import, but don't hard crash yet
_api_token = validate_api_key()


# Import tools
from tools.pdf_processor import load_pdf
from tools.wiki_search import search_wikipedia
from tools.url_retriever import retrieve_url_content

class State(TypedDict):
    messages: Annotated[List, add_messages]
    context_files: List[str] # List of file paths or identifiers
    context_urls: List[str] # List of URLs

# Define tools
from langchain_core.tools import tool

@tool
def pdf_tool(file_path: str):
    """Extracts text from a PDF file."""
    return load_pdf(file_path)

@tool
def wiki_tool(query: str):
    """Searches Wikipedia for a query."""
    return search_wikipedia(query)

@tool
def url_tool(url: str):
    """Retrieves content from a URL."""
    return retrieve_url_content(url)

tools = [pdf_tool, wiki_tool, url_tool]

# Initialize LLM
# Use Hugging Face Inference API
# We use a model that supports tool calling or at least follows instructions well
# Start with a reliable model on the free inference API
# Zephyr 7B Beta is excellent for following instructions and chat
repo_id = "HuggingFaceH4/zephyr-7b-beta"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.1,
    stop_sequences=["\n\n", "User:", "---", "[[user"]
)

chat_model = ChatHuggingFace(llm=llm)

# Bind tools to the chat model
llm_with_tools = chat_model.bind_tools(tools)

# Global vector store to avoid re-embedding on every turn (simple caching)
# In a production app, this should be managed better (e.g., per session or persistent)
vector_store = None
current_files_hash = ""

def get_retriever(files, urls):
    """
    Creates or updates a vector store from the provided files and URLs.
    """
    global vector_store, current_files_hash
    
    # Simple hash to check if files changed (naive implementation)
    new_hash = str(sorted(files)) + str(sorted(urls))
    
    if vector_store is not None and new_hash == current_files_hash:
        return vector_store.as_retriever(search_kwargs={"k": 3})
    
    documents = []
    
    # Load PDFs
    if files:
        for file_path in files:
            try:
                docs = load_pdf(file_path)
                for doc in docs:
                    doc.metadata["source"] = f"PDF - {os.path.basename(file_path)}"
                documents.extend(docs)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                
    # Load URLs
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
        
    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    # Create Vector Store
    # Use a small, fast embedding model optimal for CPU
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(splits, embeddings)
    current_files_hash = new_hash
    
    return vector_store.as_retriever(search_kwargs={"k": 3})

def generate_system_prompt(state: State):
    """Generates a dynamic system prompt based on context retrieved via RAG."""
    files = state.get("context_files", [])
    urls = state.get("context_urls", [])
    messages = state.get("messages", [])
    
    # Get user query from the last message
    user_query = ""
    if messages:
        last_msg = messages[-1]
        if isinstance(last_msg, HumanMessage):
            user_query = last_msg.content
        elif isinstance(last_msg, dict) and last_msg.get("role") == "user":
             user_query = last_msg.get("content", "")
    
    prompt = """You are a multi-model retrieval-augmented generation (RAG) chatbot capable of processing queries across various domains.
Your functionalities are categorized as follows:
1. **PDF Query Handling**: When a user asks a question related to a specific PDF document, retrieve relevant information from the PDF and provide a concise answer.
2. **URL Query Handling**: If the question pertains to content from a specified set of URLs, access the relevant web document and return an accurate answer based on that source.
3. **Wikipedia Queries**: For general knowledge questions, utilize Wikipedia or other reliable knowledge bases to provide accurate and succinct answers.
4. **General LLM Responses**: When the query is more generative (jokes, stories, greetings), respond in a friendly, engaging, and conversational manner. Allow for creativity and personality.
5. **Memory Functionality**: You have access to the conversation history. Reference past interactions to provide contextually relevant and personalized responses.

**Instructions**:
- Determine the appropriate category for the user's question.
- Prioritize responses based on the user's previous interactions.
- Maintain a conversational tone for generative interactions while ensuring factual accuracy in retrieval responses.
- If a question falls outside your knowledge base or context, clarify it.

**Citation**:
- When answering from PDFs or URLs, explicitly state the source at the end of your response (e.g., "Source: PDF - [filename]" or "Source: [URL]").
"""
    
    prompt += "\n\n--- CONTEXT START ---\n"
    
    if user_query and (files or urls):
        retriever = get_retriever(files, urls)
        if retriever:
            relevant_docs = retriever.invoke(user_query)
            for doc in relevant_docs:
                source = doc.metadata.get("source", "Unknown")
                prompt += f"\n[Source: {source}]\n{doc.page_content}\n"
        else:
            prompt += "\nNo documents available to search.\n"
    else:
        prompt += "\nNo context available or no query provided to search.\n"

    prompt += "\n--- CONTEXT END ---\n"
        
    return prompt

def chatbot(state: State):
    """The main chatbot node."""
    system_prompt = generate_system_prompt(state)
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Build the graph
graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools))

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")

graph = graph_builder.compile()

# Example usage function
def run_agent(input_text, files=None, urls=None, thread_id="1"):
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "messages": [HumanMessage(content=input_text)],
        "context_files": files or [],
        "context_urls": urls or []
    }
    # For a simple run we can just invoke, but usually we want to stream or iterate
    # This is a synchronous simple wrapper for the Streamlit app
    events = graph.stream(initial_state, config=config)
    final_response = ""
    for event in events:
        if "chatbot" in event:
            message = event["chatbot"]["messages"][-1]
            content = message.content
            
            # Handle list content (common with Gemini for multimodal/grounding)
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        text_parts.append(part["text"])
                    elif isinstance(part, str):
                        text_parts.append(part)
                final_response = "".join(text_parts)
            else:
                final_response = str(content)
                
    return final_response
