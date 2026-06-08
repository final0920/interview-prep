"""Ingest pipeline runner.

Accepts a list of source paths (zip archives, source files, or directories),
processes each through extract -> chunk, deduplicates by content_hash, and
writes evidence_units.jsonl to the configured data directory.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

from coach.config import data_dir, get
from coach.ingest.chunk import (
    chunk_code,
    chunk_pdf,
    extract_create_tables,
)
from coach.ingest.extract import DEFAULT_NOISE_GLOBS, is_noise, safe_unzip
from coach.schemas import EvidenceUnit

# ---- Language detection -----------------------------------------------------

_LANG_BY_EXT: dict[str, str] = {
    ".java": "java", ".kt": "kotlin", ".scala": "scala",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp",
    ".c": "c", ".h": "cpp", ".hpp": "cpp", ".hxx": "cpp",
    ".py": "python", ".pyi": "python",
    ".go": "go", ".rs": "rust",
    ".ts": "typescript", ".tsx": "tsx",
    ".js": "javascript", ".jsx": "javascript",
    ".vue": "javascript",
    ".sql": "sql",
    ".md": "markdown", ".markdown": "markdown", ".rst": "markdown",
    ".xml": "xml", ".yml": "yaml", ".yaml": "yaml",
    ".json": "json", ".properties": "properties",
    ".toml": "toml", ".ini": "ini", ".cfg": "ini",
    ".sh": "bash", ".bat": "bash", ".ps1": "powershell",
    ".gradle": "groovy", ".cmake": "cmake",
    ".txt": "text",
}

_KEEP_NOEXT_PREFIX = ("Dockerfile", "Makefile")


def _lang_of(name: str) -> str:
    base = Path(name).name
    for pre in _KEEP_NOEXT_PREFIX:
        if base.startswith(pre):
            return "dockerfile" if pre == "Dockerfile" else "make"
    return _LANG_BY_EXT.get(Path(name).suffix.lower(), "text")


def _decode(data: bytes) -> str | None:
    for enc in ("utf-8", "gbk"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return None


# ---- Single-file chunker dispatch -------------------------------------------

def _chunk_file(file_path: Path, rel_path: str) -> list[EvidenceUnit]:
    """Chunk a single extracted file; returns [] on unrecognised or empty."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return chunk_pdf(str(file_path))

    data = file_path.read_bytes()
    text = _decode(data)
    if not text:
        return []

    lang = _lang_of(file_path.name)
    if lang == "sql":
        # extract CREATE TABLE statements; fall back to generic code chunks
        ct = extract_create_tables(text, rel_path)
        if ct:
            return ct
    return chunk_code(text, lang, rel_path)


# ---- ZIP ingestion ----------------------------------------------------------

def _ingest_zip(
    zip_path: Path,
    tmp_dir: Path,
    noise_globs: list[str],
    seen: set[str],
) -> list[EvidenceUnit]:
    """Extract zip, chunk each member, deduplicate."""
    try:
        extracted = safe_unzip(zip_path, tmp_dir / zip_path.stem, noise_globs)
    except (ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        print(f"  [skip] {zip_path.name}: {exc}")
        return []

    units: list[EvidenceUnit] = []
    for fpath in extracted:
        try:
            rel = fpath.relative_to(tmp_dir).as_posix()
        except ValueError:
            rel = fpath.name
        for eu in _chunk_file(fpath, rel):
            if eu.content_hash not in seen:
                seen.add(eu.content_hash)
                eu = eu.model_copy(update={"repo": zip_path.stem})
                units.append(eu)
    return units


# ---- Direct file/dir ingestion ----------------------------------------------

def _ingest_path(
    src: Path,
    noise_globs: list[str],
    seen: set[str],
    repo: str = "",
) -> list[EvidenceUnit]:
    """Chunk a plain file or every file under a directory."""
    files: list[Path] = []
    if src.is_dir():
        files = [f for f in src.rglob("*") if f.is_file()]
    elif src.is_file():
        files = [src]

    units: list[EvidenceUnit] = []
    for fpath in files:
        rel = fpath.as_posix()
        if is_noise(rel, noise_globs):
            continue
        for eu in _chunk_file(fpath, rel):
            if eu.content_hash not in seen:
                seen.add(eu.content_hash)
                if repo:
                    eu = eu.model_copy(update={"repo": repo})
                units.append(eu)
    return units


# ---- Public API -------------------------------------------------------------

def ingest(paths: list[str], cfg: dict) -> str:
    """Run the full ingest pipeline.

    Processes each entry in *paths* (zip / file / directory), writes
    deduplicated EvidenceUnits as newline-JSON to
    ``<data_dir>/evidence_units.jsonl``, and returns the output path as a str.

    Args:
        paths:  Source paths to ingest.
        cfg:    Loaded config dict (from coach.config.load_config).

    Returns:
        Absolute path to the written .jsonl file.
    """
    noise_globs: list[str] = get(cfg, "ingest.noise_globs", DEFAULT_NOISE_GLOBS)
    ddir = data_dir(cfg)
    out_path = ddir / "evidence_units.jsonl"
    tmp_dir = ddir / "_ingest_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    all_units: list[EvidenceUnit] = []

    for raw in paths:
        src = Path(raw)
        if not src.exists():
            print(f"  [warn] path does not exist: {src}")
            continue

        if src.suffix.lower() == ".zip":
            units = _ingest_zip(src, tmp_dir, noise_globs, seen)
        else:
            units = _ingest_path(src, noise_globs, seen)

        all_units.extend(units)
        print(f"  {src.name}: {len(units)} units")

    # Write streaming (avoid loading multi-GB into memory for large corpora)
    with open(out_path, "w", encoding="utf-8") as fp:
        for eu in all_units:
            fp.write(eu.model_dump_json() + "\n")

    print(f"Ingest done: {len(all_units)} units -> {out_path}")
    return str(out_path)
