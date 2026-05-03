"""ArchRAG application entry point.

Initialises all pipeline components, wires them together, and launches
the Gradio web interface. This is the only file that imports from every
sub-module — all other modules remain decoupled from one another.

Usage::

    python main.py                        # default local backend
    python main.py --backend cloud        # start with cloud backend
    python main.py --provider groq        # use Groq as cloud provider
    python main.py --model phi3           # override local model
    python main.py --reindex              # force re-ingest and re-embed
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterator

import gradio as gr

from src.config import (
    AppConfig,
    CloudLLMConfig,
    EmbeddingConfig,
    IngestionConfig,
    LocalLLMConfig,
    RetrievalConfig,
)
from src.engine import HybridEngine
from src.ingestor import Ingestor
from src.retriever import Retriever
from src.vector_store import VectorStore


def configure_logging(level: str) -> None:
    """Configure the root logger with a consistent format.

    Args:
        level: Standard library log level string (e.g. ``"INFO"``).
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def build_pipeline(config: AppConfig, reindex: bool = False):
    """Instantiate and wire all pipeline components.

    Args:
        config: Top-level application configuration.
        reindex: When ``True``, forces re-ingestion and re-embedding even
            if cached data already exists on disk.

    Returns:
        A tuple of ``(retriever, engine)`` ready to handle queries.
    """
    logger = logging.getLogger(__name__)

    logger.info("Initialising ingestion pipeline …")
    ingestor = Ingestor(config.ingestion)
    chunks = ingestor.ingest(force=reindex)

    logger.info("Initialising vector store …")
    store = VectorStore(config.embedding)
    store.index(chunks, force=reindex)

    retriever = Retriever(store, config.retrieval)
    engine = HybridEngine(config)

    logger.info(
        "Pipeline ready — %d vectors indexed, preferred backend: %s",
        store.vector_count,
        config.preferred_backend,
    )
    return retriever, engine


def format_sources_markdown(result) -> str:
    """Render retrieved chunk metadata as a Gradio-compatible Markdown string.

    Args:
        result: A :class:`~src.retriever.RetrievalResult` instance.

    Returns:
        Markdown string displayed in the sources panel. Returns a
        placeholder message if no chunks were retrieved.
    """
    if result is None or result.is_empty:
        return "*No relevant sources found above the similarity threshold.*"

    lines = [
        f"**{len(result.chunks)} chunk(s) retrieved** — "
        f"**{result.total_words} words** of context\n"
    ]

    for i, chunk in enumerate(result.chunks):
        source_name = Path(chunk.source).stem.replace("_", " ").title()
        bar_filled = int(chunk.similarity * 20)
        bar = "█" * bar_filled + "░" * (20 - bar_filled)
        preview = chunk.text[:280].replace("\n", " ")
        if len(chunk.text) > 280:
            preview += " …"

        lines.append(
            f"**{i + 1}. {source_name}**\n"
            f"`{bar}` {chunk.similarity:.3f}\n\n"
            f"> {preview}\n"
        )

    return "\n---\n".join(lines)


def build_ui(retriever: Retriever, engine: HybridEngine) -> gr.Blocks:
    """Construct the Gradio interface.

    Args:
        retriever: Wired retrieval pipeline.
        engine: Hybrid inference engine.

    Returns:
        A :class:`gradio.Blocks` application ready to ``.launch()``.
    """
    logger = logging.getLogger(__name__)

    def chat(
        query: str,
        history: list,
        backend: str,
        show_sources: bool,
    ) -> Iterator[tuple[list, str]]:
        """Handle a single chat turn, streaming tokens to the UI.

        Args:
            query: User's message from the Gradio chat input.
            history: Gradio chat history list of ``(user, assistant)`` tuples.
            backend: Selected backend from the radio widget.
            show_sources: Whether to populate the sources panel.

        Yields:
            Tuples of ``(updated_history, sources_markdown)``.
        """
        if not query.strip():
            yield history, "*Enter a question above.*"
            return

        engine.set_backend(backend.lower())

        try:
            result = retriever.retrieve(query)
        except RuntimeError as exc:
            history.append((query, f"❌ Retrieval error: {exc}"))
            yield history, ""
            return

        sources_md = format_sources_markdown(result) if show_sources else ""

        if result.is_empty:
            history.append((
                query,
                "I couldn't find relevant information in my knowledge base. "
                "Try adding more documents to `data/raw/` and re-indexing.",
            ))
            yield history, sources_md
            return

        history.append((query, ""))
        partial = ""

        try:
            for token in engine.generate(query, result.context):
                partial += token
                history[-1] = (query, partial)
                yield history, sources_md
        except RuntimeError as exc:
            history[-1] = (query, f"❌ Generation error: {exc}")
            yield history, sources_md

    def reindex_documents() -> str:
        """Re-ingest and re-embed all documents in ``data/raw/``."""
        try:
            ingestor = Ingestor()
            chunks = ingestor.ingest(force=True)
            retriever._store.index(chunks, force=True)
            return f"✅ Re-indexed {len(chunks)} chunks successfully."
        except Exception as exc:
            logger.error("Re-index failed: %s", exc)
            return f"❌ Re-index failed: {exc}"

    def upload_document(filepath: str, status: str) -> str:
        """Copy an uploaded file into ``data/raw/``."""
        import shutil
        if not filepath:
            return status
        dest = Path("data/raw") / Path(filepath).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(filepath, dest)
        return f"✅ Uploaded: {dest.name}\n\nClick **Re-index** to make it searchable."

    with gr.Blocks(
        title="ArchRAG — Local AI Knowledge Base",
        theme=gr.themes.Base(
            primary_hue="slate",
            secondary_hue="zinc",
            neutral_hue="zinc",
            font=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
        ),
        css="""
        .gradio-container { max-width: 1440px !important; }
        footer { display: none !important; }
        #title { text-align: center; padding: 1rem 0 0.5rem; }
        #title h1 { font-size: 2.2em; letter-spacing: -0.02em; }
        #title p  { color: #888; margin-top: 0.25em; font-size: 0.9em; }
        """,
    ) as demo:

        gr.HTML("""
        <div id="title">
            <h1>⚡ ArchRAG</h1>
            <p>Local AI Knowledge Base · Hybrid Local/Cloud Inference · Runs Offline</p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=540,
                    bubble_full_width=False,
                    show_label=False,
                    avatar_images=(None, "🤖"),
                )
                with gr.Row():
                    query_box = gr.Textbox(
                        placeholder="Ask anything about Arch Linux, Hyprland, pacman …",
                        show_label=False,
                        scale=5,
                        lines=1,
                    )
                    submit_btn = gr.Button("Ask", variant="primary", scale=1)
                    clear_btn = gr.Button("Clear", scale=1)

                with gr.Accordion("⚙️ Settings", open=False):
                    backend_radio = gr.Radio(
                        choices=["Local", "Cloud"],
                        value="Local" if engine.active_backend == "local" else "Cloud",
                        label="Inference Backend",
                        info=(
                            "Local = Ollama on your machine (free, private). "
                            "Cloud = API provider (faster, requires API key)."
                        ),
                    )
                    show_sources = gr.Checkbox(
                        value=True,
                        label="Show retrieved sources",
                        info="Display which chunks informed each answer.",
                    )
                    gr.Markdown(
                        f"**Vector store:** {retriever._store.vector_count} chunks indexed  \n"
                        f"**Local model:** `{engine.local._cfg.model}`  \n"
                        f"**Cloud provider:** `{engine.cloud._cfg.provider}` / "
                        f"`{engine.cloud._cfg.model}`"
                    )

            with gr.Column(scale=2):
                gr.Markdown("### 📎 Retrieved Sources")
                gr.Markdown(
                    "*Sources retrieved from your knowledge base appear here. "
                    "Higher similarity = more relevant.*",
                    elem_classes=["source-panel"],
                )
                sources_panel = gr.Markdown(
                    value="*Ask a question to see retrieved sources.*"
                )

                with gr.Accordion("📁 Manage Documents", open=False):
                    file_upload = gr.File(
                        label="Upload .txt or .md",
                        file_types=[".txt", ".md"],
                        type="filepath",
                    )
                    upload_status = gr.Textbox(
                        label="Status", interactive=False, lines=3
                    )
                    reindex_btn = gr.Button("🔄 Re-index Documents", variant="secondary")

        gr.Examples(
            examples=[
                ["How do I install a package with pacman?"],
                ["My screen is black after booting Arch. What should I check?"],
                ["How do I set up keybindings in Hyprland?"],
                ["How do I remove orphaned packages to free up disk space?"],
                ["Audio is not working. How do I fix pipewire?"],
            ],
            inputs=query_box,
            label="Example Questions",
        )

        submit_btn.click(
            fn=chat,
            inputs=[query_box, chatbot, backend_radio, show_sources],
            outputs=[chatbot, sources_panel],
        ).then(lambda: "", outputs=query_box)

        query_box.submit(
            fn=chat,
            inputs=[query_box, chatbot, backend_radio, show_sources],
            outputs=[chatbot, sources_panel],
        ).then(lambda: "", outputs=query_box)

        clear_btn.click(
            lambda: ([], "*Ask a question to see retrieved sources.*"),
            outputs=[chatbot, sources_panel],
        )

        file_upload.change(
            fn=upload_document,
            inputs=[file_upload, upload_status],
            outputs=upload_status,
        )

        reindex_btn.click(fn=reindex_documents, outputs=upload_status)

    return demo


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed :class:`argparse.Namespace` object.
    """
    parser = argparse.ArgumentParser(description="ArchRAG — Local AI Knowledge Base")
    parser.add_argument(
        "--backend",
        choices=["local", "cloud"],
        default="local",
        help="Preferred inference backend (default: local)",
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "groq"],
        default="anthropic",
        help="Cloud provider when --backend=cloud (default: anthropic)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the local Ollama model (e.g. phi3, gemma2:2b)",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force re-ingestion and re-embedding of all documents",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Gradio server port (default: 7860)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args()


def main() -> None:
    """Application entry point."""
    args = parse_args()
    configure_logging(args.log_level)
    logger = logging.getLogger(__name__)

    local_llm_cfg = LocalLLMConfig(
        model=args.model or LocalLLMConfig().model
    )
    cloud_llm_cfg = CloudLLMConfig(provider=args.provider)

    config = AppConfig(
        ingestion=IngestionConfig(),
        embedding=EmbeddingConfig(),
        retrieval=RetrievalConfig(),
        local_llm=local_llm_cfg,
        cloud_llm=cloud_llm_cfg,
        preferred_backend=args.backend,
        fallback_to_local=True,
        log_level=args.log_level,
        gradio_port=args.port,
    )

    retriever, engine = build_pipeline(config, reindex=args.reindex)

    logger.info("Launching Gradio UI on http://0.0.0.0:%d", config.gradio_port)
    demo = build_ui(retriever, engine)
    demo.launch(
        server_name=config.gradio_host,
        server_port=config.gradio_port,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
