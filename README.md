# 🤖 Multi-AI Agent RAG Chatbot

A production-grade, multi-agent Retrieval-Augmented Generation (RAG) chatbot built with **LangGraph**, **LangChain**, and **Streamlit**. The system routes each user query to the most appropriate specialist AI agent — automatically.

---

## ✨ Features

- 📄 **PDF RAG** — Upload any PDF and ask questions about its contents
- 🌐 **URL RAG** — Paste any webpage URL and query it directly
- 🔍 **Wikipedia Search** — Factual/general-knowledge questions are answered via live Wikipedia lookup
- 💬 **Conversational Chat** — Handles greetings, casual talk, jokes, and open-ended conversation
- 🔀 **Supervisor Agent** — Automatically classifies and routes each query to the right specialist
- 🛡️ **Input Guardrails** — Blocks prompt injection, SQL injection, code injection, and profanity before the LLM ever sees the input
- 🔒 **Output Guardrails** — Redacts PII (emails, phones, SSNs), filters toxic responses, flags potential hallucinations
- 📊 **Evaluation Layer** — Scores every response on faithfulness, answer relevance, context precision/recall, hallucination, and latency
- 🔭 **Observability** — Real-time LangGraph tracing via Arize Phoenix at `http://localhost:6006`
- 🔁 **LLM Gateway** — LiteLLM-powered gateway with auto-retries, model fallbacks, and token/cost tracking

---

## 🏗️ Architecture

```
User Input (Streamlit UI)
        │
        ▼
 [Input Guardrails]   ← blocks injections / profanity
        │
        ▼
   [SUPERVISOR]       ← classifies query & routes to specialist
   ┌────┼────┐
   ▼    ▼    ▼
 RAG  Research  Chat
Agent  Agent   Agent
(PDF/  (Wiki   (LLM
 URL)  search)  only)
   └────┼────┘
        │
        ▼
 [Output Guardrails]  ← PII redaction, toxicity, hallucination check
        │
        ▼
 [Evaluation Layer]   ← relevance, faithfulness, latency scores
        │
        ▼
   Answer shown
```

---

## 📁 Project Structure

```
Multi-AI-Agent-RAG-Chatbot/
│
├── app.py                    # Streamlit web UI
├── agent.py                  # LangGraph multi-agent orchestration
│
├── tools/
│   ├── pdf_processor.py      # PDF text extraction (PyPDFLoader)
│   ├── wiki_search.py        # Wikipedia search tool
│   └── url_retriever.py      # Web URL content retrieval
│
├── gateway/
│   ├── gateway.py            # LiteLLM LLM factory (Qwen2.5-72B)
│   ├── router.py             # Smart router with fallbacks, retries & cost tracking
│   └── config.py             # Model configs, timeout, retry settings
│
├── guardrails/
│   ├── input_guard.py        # Pre-LLM security: injection, profanity, length checks
│   ├── output_guard.py       # Post-LLM safety: PII redaction, toxicity, hallucination
│   └── config.py             # Guardrail patterns and thresholds
│
├── evaluation/
│   ├── evaluator.py          # Main evaluation pipeline
│   ├── metrics.py            # Relevance, faithfulness, precision/recall, hallucination
│   └── config.py             # Score thresholds and latency limits
│
├── monitoring/
│   ├── telemetry.py          # Arize Phoenix + OpenInference OTEL setup
│   └── config.py             # Phoenix host/port, LangSmith project name
│
├── prompts/
│   ├── supervisor_prompt.py  # Supervisor classification prompt
│   ├── rag_prompt.py         # RAG agent system prompt
│   ├── research_prompt.py    # Research agent system prompt
│   └── chat_prompt.py        # Chat agent system prompt
│
├── retrievers/
│   └── vector_store.py       # FAISS vector store builder with caching
│
├── tests/
│   ├── test_guardrails.py    # Guardrails unit tests
│   ├── test_gateway.py       # Gateway unit tests
│   └── test_evaluation.py    # Evaluation metrics unit tests
│
├── config/
│   └── settings.py           # Global app settings
│
├── requirements.txt
├── .env.example
└── list_models.py
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- A free [Hugging Face API Token](https://huggingface.co/settings/tokens)

### 1. Clone the Repository

```bash
git clone https://github.com/Vatsalyadwiv1111/Multi-AI-Agent-RAG-Chatbot.git
cd Multi-AI-Agent-RAG-Chatbot
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

```bash
cp .env.example .env
```

Open `.env` and fill in your tokens:

```env
# Required
HUGGINGFACEHUB_API_TOKEN=hf_your_actual_token_here

# Optional — enables LangSmith tracing dashboard
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=multi-agent-rag-chatbot
```

### 5. Run the App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

> **Bonus:** Arize Phoenix observability dashboard launches automatically at `http://localhost:6006`

---

## 🧠 How the Agents Work

| Agent | Triggered When | What It Does |
|---|---|---|
| **Supervisor** | Every query | Classifies query → routes to specialist |
| **RAG Agent** | PDF or URL is uploaded | Embeds docs into FAISS, retrieves top-3 chunks, answers with citations |
| **Research Agent** | Factual/knowledge question | Calls Wikipedia tool in a loop, answers from Wikipedia content |
| **Chat Agent** | Casual conversation | Direct LLM response with personality |

---

## 🛡️ Guardrails

### Input (before LLM sees the message)
| Check | What it catches |
|---|---|
| `ExcessiveLength` | Inputs over the configured character limit |
| `PromptInjection` | "Ignore previous instructions", jailbreak attempts |
| `SQLInjection` | `SELECT * FROM`, `DROP TABLE`, etc. |
| `CodeInjection` | `exec()`, `eval()`, `os.system()`, etc. |
| `Profanity` | Abusive or harmful language |

### Output (after LLM responds)
| Check | What it does |
|---|---|
| PII Redaction | Replaces emails, phones, SSNs, credit cards with `[REDACTED ...]` |
| Toxicity Filter | Blocks harmful/profane responses |
| Hallucination Heuristic | Flags low-grounding responses when RAG context is available |

---

## 📊 Evaluation Metrics

Every response is automatically scored and logged:

| Metric | Description |
|---|---|
| **Answer Relevance** | How well the response addresses the query |
| **Faithfulness** | How grounded the answer is in retrieved context |
| **Hallucination Score** | Inverse of faithfulness |
| **Context Precision** | Fraction of retrieved docs that were relevant |
| **Context Recall** | Coverage of relevant documents |
| **Latency** | End-to-end response time in seconds |

---

## 🔭 Observability

The app automatically launches **Arize Phoenix** for full LangGraph tracing:

```
http://localhost:6006
```

See every supervisor routing decision, tool call, LLM invocation, and latency breakdown — in real time.

---

## 🔧 Tech Stack

| Component | Technology |
|---|---|
| LLM | Qwen2.5-72B-Instruct (via HuggingFace) |
| Embeddings | `all-MiniLM-L6-v2` (Sentence Transformers) |
| Vector Store | FAISS (CPU) |
| Agent Orchestration | LangGraph |
| LLM Gateway | LiteLLM |
| Observability | Arize Phoenix + OpenInference OTEL |
| UI | Streamlit |
| PDF Parsing | PyPDF |
| Web Scraping | BeautifulSoup4 + WebBaseLoader |
| Tracing (optional) | LangSmith |

---

## 📄 License

MIT License
