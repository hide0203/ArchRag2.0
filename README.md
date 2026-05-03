# ArchRAG v2

A first-principles Retrieval-Augmented Generation system for Arch Linux.
Ask questions in plain English. Get answers sourced from your own documents.
Runs fully offline. No LangChain. No LlamaIndex. No cloud required.

---

## Table of Contents

1. [What It Does](#1-what-it-does)
2. [Directory Structure](#2-directory-structure)
3. [How It Works](#3-how-it-works)
4. [The Three Files You Asked About](#4-the-three-files-you-asked-about)
5. [Installation](#5-installation)
6. [Running the Project](#6-running-the-project)
7. [Adding Your Own Documents](#7-adding-your-own-documents)
8. [Hybrid Inference — Local vs Cloud](#8-hybrid-inference--local-vs-cloud)
9. [Configuration Reference](#9-configuration-reference)
10. [How Each File Connects](#10-how-each-file-connects)
11. [Common Errors and Fixes](#11-common-errors-and-fixes)

---

## 1. What It Does

You drop text files into `data/raw/`. ArchRAG reads them, breaks them into
overlapping chunks, converts each chunk into a mathematical vector (embedding),
and stores everything in a local vector database (ChromaDB).

When you ask a question, ArchRAG converts your question into a vector too,
finds the chunks whose vectors are closest to your question's vector (semantic
similarity search), assembles those chunks into a context block, and passes
that context plus your question to a language model. The model answers using
only the retrieved context — it cannot make things up from outside your documents.

The whole pipeline runs on your machine. Your documents never leave.

---

## 2. Directory Structure

```
archrag_v2/
│
├── main.py                     ← Entry point. Wires everything together.
│                                 Launches the Gradio web UI.
│
├── requirements.txt
│
├── data/
│   ├── raw/                    ← Drop your .txt and .md files here
│   └── processed/
│       ├── chunks.json         ← Output of Ingestor (auto-created)
│       └── chroma_db/          ← ChromaDB vector store (auto-created)
│
└── src/
    ├── __init__.py             ← Package exports
    ├── config.py               ← ALL tunable parameters live here
    ├── ingestor.py             ← Loads, cleans, chunks documents
    ├── vector_store.py         ← Embeds chunks, stores/queries ChromaDB
    ├── retriever.py            ← Similarity search + context assembly
    │
    └── engine/
        ├── __init__.py         ← HybridEngine with fallback logic
        ├── base.py             ← Abstract contract all engines must satisfy
        ├── local_llm.py        ← Ollama (Mistral, Phi3, Gemma2, etc.)
        └── cloud_llm.py        ← Anthropic / OpenAI / Groq
```

---

## 3. How It Works

The pipeline has two phases that run at different times.

### Phase 1 — Indexing (runs once, or when you add new documents)

```
data/raw/*.txt, *.md
        │
        ▼
   [ Ingestor ]
   - Loads each file
   - Cleans whitespace
   - Splits into 500-word chunks with 50-word overlap
   - Saves chunks.json
        │
        ▼
   [ VectorStore.index() ]
   - Loads all-MiniLM-L6-v2 (80MB, downloads once)
   - Converts each chunk to a 384-number vector
   - Stores vectors + text + metadata in ChromaDB
   - Persists to data/processed/chroma_db/
```

### Phase 2 — Query (runs every time you ask a question)

```
Your Question ("how do I remove orphaned packages?")
        │
        ▼
   [ VectorStore.search() ]
   - Embeds your question using the same model
   - Finds the 5 most similar chunk vectors in ChromaDB
   - Filters out any with similarity < 0.25
        │
        ▼
   [ Retriever.retrieve() ]
   - Limits max 2 chunks per source file
   - Assembles chunks into a context string (max 2000 words)
   - Adds source labels and similarity scores to each chunk
        │
        ▼
   [ HybridEngine.generate() ]
   - Builds the full prompt: SYSTEM_PROMPT + context + question
   - Sends to Ollama (local) or API provider (cloud)
   - Streams tokens back to the UI as they arrive
        │
        ▼
Streaming answer in the Gradio UI + source panel showing which
chunks were used and their similarity scores
```

---

## 4. The Three Files You Asked About

### `src/config.py`

This file is the single source of truth for every tunable number in the
project. Nothing else hardcodes values. Every class receives a config
object rather than individual parameters.

```python
# The key design: frozen=True dataclasses
# Frozen means once created, nothing can accidentally change a value mid-run.
# Think of it as read-only settings.

@dataclass(frozen=True)
class IngestionConfig:
    chunk_size: int = 500       # words per chunk
    chunk_overlap: int = 50     # words shared between adjacent chunks
    ...

@dataclass(frozen=True)
class RetrievalConfig:
    min_similarity: float = 0.25   # discard chunks below this score
    max_context_words: int = 2000  # hard cap before LLM sees context
    ...

# AppConfig nests all sub-configs.
# DEFAULT_CONFIG is a ready-to-use singleton.
DEFAULT_CONFIG = AppConfig()
```

**To change a parameter without touching the code:**

```python
from src.config import AppConfig, LocalLLMConfig

config = AppConfig(
    local_llm=LocalLLMConfig(model="phi3", temperature=0.2)
)
```

Only `local_llm` changes. Every other sub-config stays at its default.
This is called "partial override" and is only possible because of the
nested frozen dataclass structure.

---

### `src/engine/base.py`

This file defines the contract that `LocalLLMEngine` and `CloudLLMEngine`
must both satisfy. It uses Python's `ABC` (Abstract Base Class) module.

```python
from abc import ABC, abstractmethod

class BaseEngine(ABC):

    @abstractmethod
    def generate(self, query: str, context: str) -> Iterator[str]:
        # Every engine MUST implement this.
        # It must be a generator that yields string tokens.
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        # Every engine MUST implement this.
        # Returns True if the backend is reachable right now.
        raise NotImplementedError
```

**Why this matters:**

`main.py` and `HybridEngine` only ever call `engine.generate()` and
`engine.is_available()`. They do not know or care whether the engine is
talking to Ollama or Anthropic. You could add a fourth engine tomorrow
(e.g. a HuggingFace local pipeline) and `main.py` would work with it
unchanged — as long as it inherits from `BaseEngine`.

`SYSTEM_PROMPT` and `build_prompt()` also live here so both engines
produce an identical prompt structure. The only thing that differs between
backends is the HTTP call, not the prompt.

---

### `src/engine/local_llm.py`

`LocalLLMEngine` inherits from `BaseEngine` and implements both abstract
methods for the Ollama backend.

```
LocalLLMEngine
├── is_available()    → GET /api/tags on localhost:11434 (3s timeout)
├── list_models()     → Returns ["mistral:latest", "phi3", ...]
├── ensure_model()    → Checks list_models(), pulls if missing
├── generate()        → POST /api/generate with stream=True, yields tokens
└── _pull_model()     → POST /api/pull, streams download progress
```

**The streaming mechanism:**

Ollama's API returns newline-delimited JSON when `stream=True`:

```
{"model":"mistral","response":"To","done":false}
{"model":"mistral","response":" remove","done":false}
{"model":"mistral","response":" orphaned","done":false}
...
{"model":"mistral","response":"","done":true,"total_duration":4200000000}
```

`generate()` reads these lines inside a `with requests.post(..., stream=True)`
context manager and `yield`s each `response` value. The Gradio UI, which
calls `engine.generate()` in a loop, appends each token to the chat bubble
in real time.

**The config injection pattern:**

```python
class LocalLLMEngine(BaseEngine):
    def __init__(self, config: LocalLLMConfig = LocalLLMConfig()) -> None:
        self._cfg = config
```

The engine receives its config rather than reading from a global. This
means `HybridEngine` can pass different configs to `LocalLLMEngine` and
`CloudLLMEngine` independently. It also means you can test the engine
with a mock config that points at a test server.

---

## 5. Installation

### Step 1 — Install Ollama (for local inference)

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Start the server:

```bash
ollama serve
```

Pull a model. Choose based on your hardware:

```bash
# RTX 3050 (4GB VRAM) — recommended
ollama pull mistral          # 4.1GB, best quality

# ThinkPad CPU-only — use these instead
ollama pull phi3             # 2.3GB, fast on CPU
ollama pull gemma2:2b        # 1.6GB, lightest option
```

### Step 2 — Install Python dependencies

```bash
# On Arch Linux
pip install -r requirements.txt --break-system-packages

# On any other system
pip install -r requirements.txt
```

### Step 3 — (Optional) Set a cloud API key

Only needed if you want to use cloud inference. Pick one:

```bash
# Groq — free tier, very fast, recommended for cloud fallback
export GROQ_API_KEY="gsk_..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenAI
export OPENAI_API_KEY="sk-..."
```

Add the export to your `~/.bashrc` or `~/.zshrc` to make it permanent.

---

## 6. Running the Project

### Standard run (local Ollama, Mistral)

```bash
python main.py
```

Open http://localhost:7860 in your browser.

### Switch to cloud backend at startup

```bash
# Use Groq (free, fast — best cloud option for this project)
python main.py --backend cloud --provider groq

# Use Anthropic
python main.py --backend cloud --provider anthropic

# Use OpenAI
python main.py --backend cloud --provider openai
```

### Use a lighter model (for ThinkPad or CPU-only)

```bash
python main.py --model phi3
python main.py --model gemma2:2b
```

### Force re-ingest and re-embed (after adding new documents)

```bash
python main.py --reindex
```

### Run with debug logging to see every query and token

```bash
python main.py --log-level DEBUG
```

### All options

```
--backend    local | cloud          Preferred inference backend (default: local)
--provider   anthropic | openai | groq   Cloud provider (default: anthropic)
--model      any Ollama model tag   Override local model (default: mistral)
--reindex    flag                   Force re-ingestion and re-embedding
--port       integer                Gradio port (default: 7860)
--log-level  DEBUG|INFO|WARNING|ERROR    Logging verbosity (default: INFO)
```

---

## 7. Adding Your Own Documents

Drop any `.txt` or `.md` file into `data/raw/`, then re-index:

```bash
python main.py --reindex
```

Or use the **Manage Documents** accordion in the Gradio UI to upload
files and click the Re-index button without touching the terminal.

**Good documents to add:**

- Arch Wiki pages (copy the text content and save as `.txt`)
- Your own notes and dotfiles
- Man pages: `man pacman | col -b > data/raw/pacman_man.txt`
- README files from tools you use
- Any technical documentation in plain text

**The more specific and well-structured your documents, the better the
answers.** A focused 5-page document about one tool outperforms a
vague 50-page dump on everything.

---

## 8. Hybrid Inference — Local vs Cloud

The `HybridEngine` in `src/engine/__init__.py` wraps both backends.

```
HybridEngine
├── .local  →  LocalLLMEngine  (Ollama)
├── .cloud  →  CloudLLMEngine  (Anthropic / OpenAI / Groq)
└── .generate()
      │
      ├─ Try preferred backend
      │     └─ If it fails AND fallback_to_local=True:
      │            yield "⚠️ switching to fallback…"
      │            yield from fallback.generate(...)
      └─ If both fail: raise RuntimeError
```

**Switching backends from the UI:**

The Settings accordion in the Gradio UI has a Local / Cloud radio button.
Selecting Cloud switches the `HybridEngine` to the cloud backend for all
subsequent queries. Switching back to Local takes effect immediately.

**Automatic fallback:**

If you have `fallback_to_local=True` (the default) and your cloud API
key expires or the provider is down, ArchRAG will automatically switch to
Ollama mid-session and show a warning in the chat bubble. You will not
get an error page.

**Recommended cloud option for this hardware setup (RTX 3050 + ThinkPad):**

Use Groq as your cloud fallback. It is free up to generous rate limits,
returns responses in under 2 seconds, and supports Llama 3 and Mistral.

```bash
export GROQ_API_KEY="gsk_..."
python main.py --backend local --provider groq
# Local = primary, Groq = automatic fallback
```

---

## 9. Configuration Reference

All parameters live in `src/config.py`. To override any value, pass a
custom config to `main.py`'s `build_pipeline()` or modify the defaults
directly in the config file.

| Parameter | Default | What it controls |
|---|---|---|
| `chunk_size` | 500 | Words per chunk. Larger = more context, less precise retrieval |
| `chunk_overlap` | 50 | Words shared between adjacent chunks. Prevents boundary cutoff |
| `min_similarity` | 0.25 | Below this cosine score, a chunk is discarded |
| `top_k` | 5 | Max chunks retrieved before deduplication |
| `max_context_words` | 2000 | Hard cap on words passed to the LLM |
| `max_chunks_per_source` | 2 | Max chunks from any single file |
| `local_llm.model` | `"mistral"` | Ollama model tag |
| `local_llm.temperature` | 0.1 | Lower = more factual, higher = more creative |
| `local_llm.num_ctx` | 4096 | Context window in tokens (Ollama setting) |
| `cloud_llm.provider` | `"anthropic"` | Cloud provider |
| `embedding.model_name` | `"all-MiniLM-L6-v2"` | Sentence transformer model |
| `embedding.batch_size` | 64 | Chunks embedded per GPU forward pass |

---

## 10. How Each File Connects

Reading this diagram top-to-bottom shows the dependency chain.
An arrow means "imports from / depends on".

```
main.py
  ├── src/config.py           (reads AppConfig, builds sub-configs)
  ├── src/ingestor.py         (Ingestor class)
  ├── src/vector_store.py     (VectorStore class)
  ├── src/retriever.py        (Retriever class, depends on VectorStore)
  └── src/engine/__init__.py  (HybridEngine)
        ├── src/engine/base.py       (BaseEngine ABC, SYSTEM_PROMPT, build_prompt)
        ├── src/engine/local_llm.py  (LocalLLMEngine → inherits BaseEngine)
        └── src/engine/cloud_llm.py  (CloudLLMEngine → inherits BaseEngine)
```

`main.py` is the only file that touches everything.
All `src/` modules are decoupled from each other — they communicate
only through the types defined in `config.py` and the data classes
(`Chunk`, `SearchResult`, `RetrievalResult`).

---

## 11. Common Errors and Fixes

**`RuntimeError: Vector store is empty`**

The index has not been built yet. Run:
```bash
python main.py --reindex
```

**`Ollama is not reachable at http://localhost:11434`**

Ollama is not running. Start it:
```bash
ollama serve
```
If you are on the ThinkPad and Ollama is running on the TUF over the
network, change `base_url` in `LocalLLMConfig`:
```python
# src/config.py
LocalLLMConfig(base_url="http://192.168.1.x:11434")
```

**`Cloud backend unavailable: environment variable 'ANTHROPIC_API_KEY' is not set`**

You switched to cloud mode but have not exported the key. Either export
it or switch back to local in the UI's Settings accordion.

**`No module named 'chromadb'` / `No module named 'sentence_transformers'`**

Dependencies are not installed. Run:
```bash
pip install -r requirements.txt --break-system-packages
```

**Answers are poor quality or say "I don't have enough information"**

Your documents do not cover the topic you asked about. Add more
relevant documents to `data/raw/` and run `python main.py --reindex`.
Lowering `min_similarity` from `0.25` to `0.15` in `config.py` will
also retrieve more chunks, at the cost of potentially noisy context.

**Inference is very slow**

You are running on CPU. Either:
- Switch to a smaller model: `python main.py --model gemma2:2b`
- Use the cloud backend: `python main.py --backend cloud --provider groq`
- Or run Ollama on the TUF (RTX 3050) and point the ThinkPad at it
  over the network as described in the Ollama error fix above.
