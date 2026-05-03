"""Document ingestion and chunking for the ArchRAG pipeline.

Responsible for loading raw text files from disk, normalising their content,
and splitting them into overlapping chunks that downstream embedding models
can consume efficiently.
"""

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

from .config import IngestionConfig

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single text chunk produced by the ingestion pipeline.

    Attributes:
        chunk_id: Stable unique identifier derived from the source filename
            and chunk index.
        source: Absolute or relative path to the originating document.
        chunk_index: Zero-based position of this chunk within its document.
        total_chunks: Total number of chunks produced from the same document.
        text: The raw text content of this chunk.
        word_count: Number of whitespace-separated tokens in ``text``.
    """

    chunk_id: str
    source: str
    chunk_index: int
    total_chunks: int
    text: str
    word_count: int


class Ingestor:
    """Loads, cleans, and chunks plain-text documents.

    The chunking algorithm slides a fixed-size window over the token list
    of each document. Adjacent chunks share ``overlap`` tokens so that
    sentences spanning a boundary appear in full in at least one chunk.

    Args:
        config: Ingestion parameters. Defaults to ``IngestionConfig()``.

    Example::

        ingestor = Ingestor()
        chunks = ingestor.ingest()
        print(f"Produced {len(chunks)} chunks")
    """

    def __init__(self, config: IngestionConfig = IngestionConfig()) -> None:
        self._cfg = config
        self._cfg.processed_dir.mkdir(parents=True, exist_ok=True)

    def ingest(self, force: bool = False) -> list[Chunk]:
        """Run the full ingestion pipeline and persist the output.

        Loads all supported documents from ``config.raw_data_dir``, chunks
        them, and writes ``chunks.json`` to ``config.processed_dir``.

        Args:
            force: When ``True``, re-ingest even if ``chunks.json`` already
                exists. When ``False``, returns the cached result if present.

        Returns:
            A list of :class:`Chunk` objects ready for embedding.

        Raises:
            FileNotFoundError: If ``raw_data_dir`` does not exist and no
                sample documents can be created.
        """
        output_path = self._cfg.processed_dir / "chunks.json"

        if output_path.exists() and not force:
            logger.info("Loading cached chunks from %s", output_path)
            return self._load_chunks(output_path)

        documents = self._load_documents()

        if not documents:
            logger.warning(
                "No documents found in %s — creating sample corpus.",
                self._cfg.raw_data_dir,
            )
            self._write_sample_corpus()
            documents = self._load_documents()

        chunks = list(self._chunk_documents(documents))
        self._save_chunks(chunks, output_path)

        logger.info(
            "Ingested %d documents → %d chunks → %s",
            len(documents),
            len(chunks),
            output_path,
        )
        return chunks

    def _load_documents(self) -> list[dict]:
        """Scan ``raw_data_dir`` and return a list of ``{filename, content}`` dicts."""
        documents = []
        for ext in self._cfg.supported_extensions:
            for path in self._cfg.raw_data_dir.glob(f"**/*{ext}"):
                try:
                    content = path.read_text(encoding="utf-8", errors="ignore")
                    documents.append({"filename": str(path), "content": content})
                    logger.debug("Loaded document: %s", path.name)
                except OSError as exc:
                    logger.warning("Could not read %s: %s", path, exc)

        seen: set[str] = set()
        unique = []
        for doc in documents:
            if doc["filename"] not in seen:
                seen.add(doc["filename"])
                unique.append(doc)

        return unique

    def _chunk_documents(self, documents: list[dict]) -> Iterator[Chunk]:
        """Yield :class:`Chunk` objects for every document in ``documents``."""
        for doc in documents:
            cleaned = self._clean_text(doc["content"])
            raw_chunks = self._split_text(cleaned)
            total = len(raw_chunks)
            stem = Path(doc["filename"]).stem

            for idx, text in enumerate(raw_chunks):
                yield Chunk(
                    chunk_id=f"{stem}_chunk_{idx}",
                    source=doc["filename"],
                    chunk_index=idx,
                    total_chunks=total,
                    text=text,
                    word_count=len(text.split()),
                )

    def _clean_text(self, text: str) -> str:
        """Normalise whitespace without destroying document structure.

        Args:
            text: Raw document content.

        Returns:
            Cleaned text with collapsed blank lines and stripped trailing
            whitespace on each line.
        """
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+\n", "\n", text)
        return text.strip()

    def _split_text(self, text: str) -> list[str]:
        """Split ``text`` into overlapping word-level chunks.

        Implements a sliding window where each step advances by
        ``chunk_size - overlap`` words, ensuring adjacent chunks share
        ``overlap`` words of context.

        Args:
            text: Cleaned document text.

        Returns:
            List of chunk strings. The final chunk may be shorter than
            ``chunk_size`` if the document does not divide evenly.
        """
        tokens = text.split()
        step = self._cfg.chunk_size - self._cfg.chunk_overlap
        chunks: list[str] = []
        start = 0

        while start < len(tokens):
            chunk_tokens = tokens[start : start + self._cfg.chunk_size]
            chunks.append(" ".join(chunk_tokens))
            start += step

        return chunks

    def _save_chunks(self, chunks: list[Chunk], path: Path) -> None:
        """Serialise ``chunks`` to JSON at ``path``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump([asdict(c) for c in chunks], fh, indent=2, ensure_ascii=False)

    def _load_chunks(self, path: Path) -> list[Chunk]:
        """Deserialise chunks from a JSON file produced by :meth:`_save_chunks`."""
        with open(path, "r", encoding="utf-8") as fh:
            return [Chunk(**item) for item in json.load(fh)]

    def _write_sample_corpus(self) -> None:
        """Write a minimal Arch Linux sample corpus for demo and testing."""
        self._cfg.raw_data_dir.mkdir(parents=True, exist_ok=True)

        samples = {
            "arch_pacman.txt": """\
Pacman is the package manager for Arch Linux. It uses binary packages in the tar.zst format.

Installing packages:
To install a single package or list of packages: pacman -S package_name
To install without confirmation: pacman -S --noconfirm package_name

Removing packages:
Remove a package: pacman -R package_name
Remove with unused dependencies: pacman -Rs package_name
Remove with config files: pacman -Rns package_name

Upgrading the system:
Synchronize the package database and upgrade all packages: pacman -Syu
Always run a full system upgrade before installing new packages.

Searching for packages:
Search in the database: pacman -Ss keyword
Search locally installed: pacman -Qs keyword
Display package information: pacman -Si package_name

Cleaning the cache:
Pacman stores downloads in /var/cache/pacman/pkg/
Remove packages not installed: pacman -Sc
Remove all cached packages: pacman -Scc

AUR helpers like yay or paru can install from the Arch User Repository.
""",
            "hyprland_config.txt": """\
Hyprland is a dynamic tiling Wayland compositor. Configuration: ~/.config/hypr/hyprland.conf

Monitor configuration:
monitor=name,resolution,position,scale
Example: monitor=DP-1,1920x1080@144,0x0,1

Keybindings:
bind=SUPER,Return,exec,kitty
bind=SUPER,Q,killactive,
bind=SUPER,M,exit,
bind=SUPER,R,exec,wofi --show drun

Moving windows:
bind=SUPER SHIFT,1,movetoworkspace,1
bind=SUPER,1,workspace,1

Window rules:
windowrule=float,^(pavucontrol)$
windowrulev2=opacity 0.8,class:^(kitty)$

Environment variables for Wayland:
env=XCURSOR_SIZE,24
env=QT_QPA_PLATFORM,wayland
""",
            "arch_troubleshooting.txt": """\
Common Arch Linux issues and solutions:

Black screen after boot:
Check display manager: systemctl status sddm
Check logs: journalctl -xe
For NVIDIA: ensure nvidia-drm.modeset=1 is in kernel parameters

No sound:
Install pipewire: pacman -S pipewire pipewire-pulse wireplumber
Start services: systemctl --user enable --now pipewire pipewire-pulse wireplumber
Check with: pactl info

Network issues:
NetworkManager: systemctl enable --now NetworkManager
List connections: nmcli connection show
Connect to wifi: nmcli device wifi connect SSID password PASSWORD

Disk space issues:
Check usage: df -h
Find large files: du -sh /* 2>/dev/null | sort -rh | head -20
Clean pacman cache: pacman -Sc
Remove orphans: pacman -Rns $(pacman -Qtdq)
""",
        }

        for name, content in samples.items():
            (self._cfg.raw_data_dir / name).write_text(content.strip(), encoding="utf-8")

        logger.info("Wrote %d sample documents to %s", len(samples), self._cfg.raw_data_dir)
