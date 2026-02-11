# Multimodal RAG Agent

This project is a Streamlit-based application that integrates a LangGraph agent capable of answering user queries using PDF documents, Wikipedia, and external URLs.

## Prerequisites

- Python 3.9+
- Hugging Face API Token

## Setup

1.  **Clone the repository** (or navigate to the project directory):
    ```bash
    cd /Users/vatsalyadwivedi/.gemini/antigravity/scratch/multimodal_rag_agent
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables**:
    Create a `.env` file in the project root and add your Hugging Face API token:
    ```
    HUGGINGFACEHUB_API_TOKEN=your_hf_token_here
    ```
    You can get a free token from [Hugging Face Settings](https://huggingface.co/settings/tokens).

## Running the Application

1.  **Start the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

2.  **Access the UI**:
    Open the URL provided in the terminal (usually `http://localhost:8501`).

3.  **Use the Agent**:
    - Upload PDF files using the sidebar.
    - Enter URLs in the sidebar (one per line).
    - Ask questions in the chat interface.

## Project Structure

- `app.py`: Main Streamlit application.
- `agent.py`: LangGraph agent definition and logic.
- `tools/`: Directory containing tool implementations.
    - `pdf_processor.py`: PDF text extraction.
    - `wiki_search.py`: Wikipedia search.
    - `url_retriever.py`: URL text and content retrieval.
- `requirements.txt`: Python dependencies.
