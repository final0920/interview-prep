"""Ingest extract: safe ZIP unzip + noise filtering.

Zip-slip and zip-bomb guards, binary sniffing, extension/dir deny-list,
and streaming extraction to a destination directory.
"""
from __future__ import annotations

import fnmatch
import hashlib
import stat
import zipfile
from pathlib import Path

# ---- Zip safety limits -------------------------------------------------------
_MAX_TOTAL_UNCOMPRESSED = 512 * 1024 ** 2  # 512 MB total uncompressed
_MAX_FILES = 20_000
_MAX_FILE_SIZE = 5 * 1024 ** 2            # 5 MB per member
_MAX_RATIO = 200                          # compression ratio bomb guard

# ---- Default noise globs (deny-list) ----------------------------------------
DEFAULT_NOISE_GLOBS: list[str] = [
    # build artefacts and generated output
    "*/target/*", "*/build/*", "*/out/*", "*/dist/*", "*/.git/*",
    "*/node_modules/*", "*/__pycache__/*", "*/venv/*", "*/.venv/*",
    "*/vendor/*", "*/bin/*", "*/obj/*",
    # binary / media
    "*.class", "*.jar", "*.war", "*.ear", "*.pyc", "*.pyo",
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.svg", "*.ico",
    "*.exe", "*.dll", "*.so", "*.dylib", "*.bin", "*.o",
    "*.zip", "*.tar", "*.gz", "*.7z", "*.rar",
    "*.xlsx", "*.xls", "*.doc", "*.docx", "*.ppt", "*.pptx",
    "*.woff", "*.woff2", "*.ttf", "*.eot", "*.mp4", "*.mp3",
    "*.db", "*.lock",
    # specific build/tool config files
    "*/pom.xml", "*/package-lock.json", "*/yarn.lock",
]


def is_noise(path: str, noise_globs: list[str]) -> bool:
    """Return True if *path* matches any pattern in *noise_globs*.

    Matching uses fnmatch against the full path string (forward-slash
    normalised).  For patterns starting with ``*/``, we also try without
    the leading ``*/`` so that top-level members like ``node_modules/x``
    are caught the same as ``a/node_modules/x``.
    """
    normalised = path.replace("\\", "/")
    for pat in noise_globs:
        if fnmatch.fnmatch(normalised, pat):
            return True
        # also match paths that have no leading directory segment
        if pat.startswith("*/") and fnmatch.fnmatch(normalised, pat[2:]):
            return True
    return False


def _is_binary(data: bytes) -> bool:
    if b"\x00" in data[:8192]:
        return True
    for enc in ("utf-8", "gbk"):
        try:
            data[:8192].decode(enc)
            return False
        except UnicodeDecodeError:
            continue
    return True


def _fix_cp437_name(raw: str) -> str:
    """Re-decode ZIP member name that was mis-decoded as cp437 from a GBK zip."""
    try:
        return raw.encode("cp437").decode("gbk")
    except Exception:
        return raw


def safe_unzip(
    zip_path: str | Path,
    dest: str | Path,
    noise_globs: list[str] | None = None,
) -> list[Path]:
    """Extract *zip_path* to *dest*, enforcing safety limits.

    Returns the list of extracted file Paths (only non-noise, non-binary,
    non-oversized members).

    Raises:
        ValueError  -- zip-slip attempt detected (path escapes dest).
        RuntimeError -- zip-bomb: total uncompressed size or file count exceeds limits.
    """
    if noise_globs is None:
        noise_globs = DEFAULT_NOISE_GLOBS

    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        # --- pre-scan for bombs / slip ---
        total_unc = 0
        n_files = 0
        for info in zf.infolist():
            if info.is_dir():
                continue
            n_files += 1
            total_unc += info.file_size
            # reject symlinks / other non-regular members. The high 16 bits of
            # external_attr hold the unix st_mode for unix-created zips. Entries
            # written via zipfile.writestr carry no file-type bits (S_IFMT == 0),
            # so only reject when a type is set and it is not a regular file.
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode) or (
                stat.S_IFMT(mode) and not stat.S_ISREG(mode)
                and not stat.S_ISDIR(mode)
            ):
                raise ValueError(
                    f"non-regular zip entry rejected: {info.filename!r}"
                )
            # zip-slip check: resolved path must be dest itself or below it.
            # A plain prefix check lets a sibling like '<dest>_evil' escape.
            member_name = _fix_cp437_name(info.filename)
            resolved = (dest / member_name).resolve()
            if not (resolved == dest or dest in resolved.parents):
                raise ValueError(f"zip-slip attempt: {info.filename!r}")
            # compression-ratio bomb: check every compressed member (no size gate)
            if info.compress_size > 0:
                ratio = info.file_size / info.compress_size
                if ratio > _MAX_RATIO:
                    raise RuntimeError(
                        f"zip-bomb: ratio {ratio:.0f}x in {info.filename!r}"
                    )

        if total_unc > _MAX_TOTAL_UNCOMPRESSED:
            raise RuntimeError(
                f"zip-bomb: total uncompressed {total_unc / 1024**2:.0f} MB "
                f"> {_MAX_TOTAL_UNCOMPRESSED // 1024**2} MB limit"
            )
        if n_files > _MAX_FILES:
            raise RuntimeError(
                f"zip-bomb: {n_files} members > {_MAX_FILES} limit"
            )

        # --- extract accepted members ---
        for info in zf.infolist():
            if info.is_dir():
                continue
            member_name = _fix_cp437_name(info.filename)

            if is_noise(member_name, noise_globs):
                continue
            if info.file_size > _MAX_FILE_SIZE:
                continue

            out_path = (dest / member_name).resolve()
            # second slip guard (paranoid): must be dest or strictly below it
            if not (out_path == dest or dest in out_path.parents):
                continue

            # bounded read: never materialise more than the per-member cap,
            # even if the declared file_size lies about the real content size
            try:
                with zf.open(info, "r") as fh:
                    data = fh.read(_MAX_FILE_SIZE + 1)
            except Exception:
                continue
            if len(data) > _MAX_FILE_SIZE:
                continue

            if _is_binary(data):
                continue

            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(data)
            extracted.append(out_path)

    return extracted
