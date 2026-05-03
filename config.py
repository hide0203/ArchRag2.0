"""Centralized configuration for the ArchRAG pipeline.

All tunable parameters live here. No magic numbers anywhere else in the codebase.
Import this module to access settings rather than hardcoding values in individual classes.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class IngestionConfig:
    """Parameters controlling document loading and chunking.

    Attributes:
        raw_data_dir: Directory scanned for source documents.
        processed_dir: Directory where intermediate outputs are written.
        chunk_size: Target word count per chunk.
        chunk_overlap: Word count shared between adjacent chunks to prevent
            context loss at boundaries.
        supported_extensions: File extensions the ingestor will load.
    """

    raw_data_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    chunk_size: int = 500
    chunk_overlap: int = 50
    supported_extensions: tuple = (".txt", ".md")


@dataclass(frozen=True)
class EmbeddingConfig:
    """Parameters for the embedding model and vector store.

    Attributes:
        model_name: HuggingFace sentence-transformers model identifier.
        chroma_dir: Persistent storage path for the ChromaDB collection.
        collection_name: Name of the ChromaDB collection.
        batch_size: Number of chunks embedded per forward pass.
        normalize: Whether to L2-normalise embeddings before storage.
            When True, cosine similarity reduces to a dot product.
    """

    model_name: str = "all-MiniLM-L6-v2"
    chroma_dir: Path = Path("data/processed/chroma_db")
    collection_name: str = "arch_docs"
    batch_size: int = 64
    normalize: bool = True


@dataclass(frozen=True)
class RetrievalConfig:
    """Parameters controlling similarity search and context assembly.

    Attributes:
        top_k: Maximum number of chunks to retrieve per query.
        min_similarity: Cosine similarity floor. Chunks below this score
            are discarded before context assembly.
        max_context_words: Hard cap on total words passed to the LLM.
            Prevents overflowing the model's context window.
        max_chunks_per_source: Deduplication limit per source file.
    """

    top_k: int = 5
    min_similarity: float = 0.25
    max_context_words: int = 2000
    max_chunks_per_source: int = 2


@dataclass(frozen=True)
class LocalLLMConfig:
    """Parameters for the Ollama local inference backend.

    Attributes:
        base_url: Ollama REST API base URL.
        model: Model tag as it appears in `ollama list`.
        temperature: Sampling temperature. Lower = more deterministic.
        top_p: Nucleus sampling probability mass.
        num_ctx: Context window size in tokens passed to Ollama.
        timeout: Request timeout in seconds.
    """

    base_url: str = "http://localhost:11434"
    model: str = "mistral"
    temperature: float = 0.1
    top_p: float = 0.9
    num_ctx: int = 4096
    timeout: int = 120


@dataclass(frozen=True)
class CloudLLMConfig:
    """Parameters for the cloud inference backend.

    Attributes:
        provider: One of ``"anthropic"``, ``"openai"``, or ``"groq"``.
        model: Provider-specific model identifier.
        api_key_env: Name of the environment variable holding the API key.
        temperature: Sampling temperature forwarded to the provider.
        max_tokens: Maximum tokens the provider will generate per response.
        timeout: HTTP request timeout in seconds.
        base_url: Override the default provider endpoint (optional).
    """

    provider: str = "groq"                   
    model: str = "llama-3.1-8b-instant"      
    api_key_env: str = "GROQ_API_KEY"       
    temperature: float = 0.1
    max_tokens: int = 1024
    timeout: int = 60
    base_url: Optional[str] = None


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration.

    Attributes:
        ingestion: Ingestion pipeline settings.
        embedding: Embedding model and vector store settings.
        retrieval: Similarity search settings.
        local_llm: Ollama backend settings.
        cloud_llm: Cloud API backend settings.
        preferred_backend: Starting backend, either ``"local"`` or ``"cloud"``.
        fallback_to_local: Automatically fall back to local if cloud fails.
        log_level: Standard library logging level string.
        gradio_port: Port on which the Gradio UI listens.
        gradio_host: Host address for the Gradio server.
    """

    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    local_llm: LocalLLMConfig = field(default_factory=LocalLLMConfig)
    cloud_llm: CloudLLMConfig = field(default_factory=CloudLLMConfig)
    preferred_backend: str = "local"
    fallback_to_local: bool = True
    log_level: str = "INFO"
    gradio_port: int = 7860
    gradio_host: str = "0.0.0.0"


DEFAULT_CONFIG = AppConfig()
