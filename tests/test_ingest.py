"""Tests for coach/ingest/{extract,chunk,run}.

All tests are offline: no network, no LLM, no model downloads.
tree-sitter paths are guarded so the fallback is always exercised.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from coach.ingest.extract import DEFAULT_NOISE_GLOBS, is_noise, safe_unzip
from coach.ingest.chunk import (
    chunk_code,
    chunk_pdf,
    extract_create_tables,
    _fallback_chunk,
    _chunk_by_window,
    _split_by_brace_blocks,
)
from coach.ingest.run import ingest
from coach.schemas import Channel, EvidenceUnit


# ============================================================
# Fixtures
# ============================================================

SAMPLE_PY = """\
def add(a, b):
    return a + b

class Foo:
    def bar(self):
        pass
"""

SAMPLE_CPP = """\
#include <iostream>

int main() {
    std::cout << "hello" << std::endl;
    return 0;
}

void helper() {
    int x = 42;
}
"""

SAMPLE_SQL = """\
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    total REAL
);

SELECT * FROM users;
"""

SAMPLE_JAVA = """\
public class Hello {
    public void greet(String name) {
        System.out.println("Hello " + name);
    }
}
"""


# ============================================================
# extract.py: is_noise
# ============================================================

class TestIsNoise:
    def test_deny_dir(self):
        assert is_noise("project/node_modules/lib/index.js", DEFAULT_NOISE_GLOBS)

    def test_deny_ext_class(self):
        assert is_noise("src/Foo.class", DEFAULT_NOISE_GLOBS)

    def test_deny_pyc(self):
        assert is_noise("app/__pycache__/mod.pyc", DEFAULT_NOISE_GLOBS)

    def test_allow_python(self):
        assert not is_noise("src/main.py", DEFAULT_NOISE_GLOBS)

    def test_allow_java(self):
        assert not is_noise("src/Main.java", DEFAULT_NOISE_GLOBS)

    def test_deny_lock(self):
        assert is_noise("project/package-lock.json", DEFAULT_NOISE_GLOBS)

    def test_deny_image(self):
        assert is_noise("assets/logo.png", DEFAULT_NOISE_GLOBS)

    def test_deny_git(self):
        assert is_noise("repo/.git/config", DEFAULT_NOISE_GLOBS)

    def test_deny_build_dir(self):
        assert is_noise("project/build/output.txt", DEFAULT_NOISE_GLOBS)

    def test_empty_globs_allows_everything(self):
        assert not is_noise("anything/goes.exe", [])


# ============================================================
# extract.py: safe_unzip
# ============================================================

class TestSafeUnzip:
    def _make_zip(self, tmp_path: Path, members: dict[str, bytes]) -> Path:
        z = tmp_path / "test.zip"
        with zipfile.ZipFile(z, "w") as zf:
            for name, data in members.items():
                zf.writestr(name, data)
        return z

    def test_normal_extraction(self, tmp_path):
        z = self._make_zip(tmp_path, {"src/hello.py": b"print('hi')"})
        dest = tmp_path / "out"
        extracted = safe_unzip(z, dest, [])
        assert len(extracted) == 1
        assert extracted[0].read_bytes() == b"print('hi')"

    def test_zip_slip_rejected(self, tmp_path):
        z = tmp_path / "slip.zip"
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr("../../../evil.txt", b"pwned")
        dest = tmp_path / "out"
        with pytest.raises(ValueError, match="zip-slip"):
            safe_unzip(z, dest, [])

    def test_sibling_prefix_slip_rejected(self, tmp_path):
        # dest=.../proj ; a member resolving to a SIBLING that merely shares the
        # 'proj' prefix (.../proj_evil) must be rejected. A naive
        # str(resolved).startswith(str(dest)) guard would let this escape.
        dest = tmp_path / "proj"
        z = tmp_path / "sibling.zip"
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr("../proj_evil/x.txt", b"pwned")
        with pytest.raises(ValueError, match="zip-slip"):
            safe_unzip(z, dest, [])

    def test_symlink_member_rejected(self, tmp_path):
        # an entry whose unix mode (external_attr >> 16) is a symlink must be
        # rejected before any extraction happens.
        import stat
        z = tmp_path / "link.zip"
        with zipfile.ZipFile(z, "w") as zf:
            zi = zipfile.ZipInfo("link")
            zi.external_attr = (stat.S_IFLNK | 0o777) << 16
            zf.writestr(zi, b"/etc/passwd")
        dest = tmp_path / "out"
        with pytest.raises(ValueError, match="non-regular"):
            safe_unzip(z, dest, [])

    def test_high_ratio_member_not_fatal(self, tmp_path):
        # A single highly-compressible member (e.g. a sparse binary) must NOT
        # abort the whole archive. Real bomb protection is the total-size cap +
        # the bounded per-member read, not a per-member ratio veto (which caused
        # false positives on legitimate sparse binaries).
        z = tmp_path / "ratio.zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("big.txt", b"A" * (1024 ** 2))  # 1 MB, ~1000x ratio
        dest = tmp_path / "out"
        extracted = safe_unzip(z, dest, [])  # must not raise
        assert [p.name for p in extracted] == ["big.txt"]

    def test_noise_binary_high_ratio_skipped_not_fatal(self, tmp_path):
        # Mirrors the real bug: a noise binary (*.db/*.bin) with a huge ratio
        # used to abort ingestion of the whole zip. It must be skipped while
        # genuine source files are still extracted.
        z = tmp_path / "mixed.zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("proj/data.db", b"A" * (1024 ** 2))   # noise + high ratio
            zf.writestr("proj/main.cpp", "int main(){return 0;}\n")
        dest = tmp_path / "out"
        extracted = safe_unzip(z, dest, None)  # default noise globs include *.db
        assert [p.name for p in extracted] == ["main.cpp"]

    def test_total_uncompressed_cap_still_enforced(self, tmp_path, monkeypatch):
        # The aggregate uncompressed budget is the real anti-bomb guard.
        monkeypatch.setattr("coach.ingest.extract._MAX_TOTAL_UNCOMPRESSED", 100)
        z = tmp_path / "big.zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("a.txt", b"x" * 500)  # 500 B non-noise > 100 B cap
        dest = tmp_path / "out"
        with pytest.raises(RuntimeError, match="zip-bomb"):
            safe_unzip(z, dest, [])

    def test_noise_filtered(self, tmp_path):
        z = self._make_zip(tmp_path, {
            "src/main.py": b"x = 1",
            "node_modules/lib.js": b"exports = {}",
        })
        dest = tmp_path / "out"
        extracted = safe_unzip(z, dest, DEFAULT_NOISE_GLOBS)
        names = [p.name for p in extracted]
        assert "main.py" in names
        assert "lib.js" not in names

    def test_binary_skipped(self, tmp_path):
        z = self._make_zip(tmp_path, {
            "data.bin": b"\x00\x01\x02\x03",
            "code.py": b"x = 1",
        })
        dest = tmp_path / "out"
        extracted = safe_unzip(z, dest, [])
        assert all(p.suffix == ".py" for p in extracted)

    def test_empty_zip(self, tmp_path):
        z = self._make_zip(tmp_path, {})
        dest = tmp_path / "out"
        assert safe_unzip(z, dest, []) == []

    def test_returns_paths_under_dest(self, tmp_path):
        z = self._make_zip(tmp_path, {"a/b/c.py": b"pass"})
        dest = tmp_path / "out"
        extracted = safe_unzip(z, dest, [])
        for p in extracted:
            assert str(p).startswith(str(dest.resolve()))


# ============================================================
# chunk.py: _split_by_brace_blocks
# ============================================================

class TestSplitByBraceBlocks:
    def test_single_block(self):
        lines = ["void f() {", "    return;", "}"]
        blocks = _split_by_brace_blocks(lines)
        assert len(blocks) == 1
        start, end, text = blocks[0]
        assert start == 1
        assert end == 3
        assert "void f()" in text

    def test_two_blocks(self):
        lines = ["int a() {", "}", "int b() {", "}"]
        blocks = _split_by_brace_blocks(lines)
        assert len(blocks) == 2

    def test_no_braces_returns_whole_file(self):
        lines = ["hello", "world"]
        blocks = _split_by_brace_blocks(lines)
        assert len(blocks) == 1
        start, end, text = blocks[0]
        assert start == 1
        assert end == 2

    def test_stray_close_brace_does_not_drop_next_block(self):
        # A brace appearing inside a string literal (e.g. a printed '}') makes
        # the naive depth counter go negative; without clamping, the block that
        # follows is silently dropped. Clamping depth to >=0 keeps it.
        lines = [
            'void a() {',
            '    printf("}");',   # stray close-brace char inside a string
            '}',
            'void b() {',
            '    return;',
            '}',
        ]
        blocks = _split_by_brace_blocks(lines)
        texts = [t for _, _, t in blocks]
        assert any("void a()" in t for t in texts)
        assert any("void b()" in t for t in texts), "second block was dropped"


# ============================================================
# chunk.py: _chunk_by_window
# ============================================================

class TestChunkByWindow:
    def test_short_text_single_chunk(self):
        # 3 lines, window=10 covers all -> at least 1 chunk, first starts at line 1
        units = _chunk_by_window("a\nb\nc", "f.py", "python", win=10)
        assert len(units) >= 1
        assert units[0].start_line == 1

    def test_long_text_multiple_chunks(self):
        text = "\n".join(f"line {i}" for i in range(200))
        units = _chunk_by_window(text, "f.py", "python", win=80, overlap=12)
        assert len(units) > 1

    def test_symbol_format(self):
        units = _chunk_by_window("x = 1\ny = 2", "f.py", "python")
        assert units[0].symbol.startswith("lines:")

    def test_channel_code(self):
        units = _chunk_by_window("def f(): pass", "f.py", "python")
        assert units[0].channel == Channel.code

    def test_channel_doc(self):
        units = _chunk_by_window("# heading\nsome text", "README.md", "markdown")
        assert units[0].channel == Channel.doc

    def test_empty_lines_skipped(self):
        units = _chunk_by_window("", "f.py", "python")
        assert units == []


# ============================================================
# chunk.py: chunk_code (fallback path)
# ============================================================

class TestChunkCode:
    def test_python_fallback_returns_units(self):
        units = chunk_code(SAMPLE_PY, "python", "foo.py")
        assert len(units) > 0
        for u in units:
            assert isinstance(u, EvidenceUnit)
            assert u.source_path == "foo.py"
            assert u.lang == "python"

    def test_cpp_brace_split(self):
        units = chunk_code(SAMPLE_CPP, "cpp", "main.cpp")
        assert len(units) > 0
        # each unit should cover a brace block or window
        for u in units:
            assert u.start_line >= 1
            assert u.end_line >= u.start_line

    def test_java_brace_split(self):
        units = chunk_code(SAMPLE_JAVA, "java", "Hello.java")
        assert len(units) > 0

    def test_empty_text_returns_empty(self):
        assert chunk_code("", "python", "empty.py") == []

    def test_whitespace_only_returns_empty(self):
        assert chunk_code("   \n\n   ", "python", "ws.py") == []

    def test_ids_are_unique(self):
        units = chunk_code(SAMPLE_CPP, "cpp", "main.cpp")
        ids = [u.id for u in units]
        assert len(ids) == len(set(ids))

    def test_content_hash_set(self):
        units = chunk_code(SAMPLE_PY, "python", "a.py")
        for u in units:
            assert u.content_hash != ""

    def test_file_line_traceability(self):
        units = chunk_code(SAMPLE_PY, "python", "path/to/mod.py")
        for u in units:
            assert u.source_path == "path/to/mod.py"
            assert u.start_line > 0

    def test_sql_channel(self):
        units = chunk_code("SELECT 1", "sql", "q.sql")
        assert len(units) > 0
        for u in units:
            assert u.channel == Channel.sql

    def test_yaml_channel(self):
        units = chunk_code("key: value\nother: 42", "yaml", "cfg.yaml")
        assert len(units) > 0
        for u in units:
            assert u.channel == Channel.config

    def test_markdown_channel(self):
        units = chunk_code("# Title\nsome text", "markdown", "README.md")
        assert len(units) > 0
        for u in units:
            assert u.channel == Channel.doc


# ============================================================
# chunk.py: extract_create_tables
# ============================================================

class TestExtractCreateTables:
    def test_finds_two_tables(self):
        units = extract_create_tables(SAMPLE_SQL, "schema.sql")
        assert len(units) == 2

    def test_symbol_contains_table_name(self):
        units = extract_create_tables(SAMPLE_SQL, "schema.sql")
        syms = [u.symbol for u in units]
        assert any("users" in s for s in syms)
        assert any("orders" in s for s in syms)

    def test_channel_is_sql(self):
        units = extract_create_tables(SAMPLE_SQL, "schema.sql")
        for u in units:
            assert u.channel == Channel.sql

    def test_no_create_returns_empty(self):
        assert extract_create_tables("SELECT 1;", "q.sql") == []

    def test_if_not_exists(self):
        ddl = "CREATE TABLE IF NOT EXISTS foo (id INT);"
        units = extract_create_tables(ddl, "t.sql")
        assert len(units) == 1
        assert "foo" in units[0].symbol


# ============================================================
# run.py: ingest
# ============================================================

class TestIngestRun:
    def _make_zip(self, path: Path, members: dict[str, str]) -> Path:
        with zipfile.ZipFile(path, "w") as zf:
            for name, content in members.items():
                zf.writestr(name, content)
        return path

    def test_ingest_zip_produces_jsonl(self, tmp_path):
        z = self._make_zip(
            tmp_path / "repo.zip",
            {"src/main.py": SAMPLE_PY, "src/schema.sql": SAMPLE_SQL},
        )
        cfg = {"paths": {"data_dir": str(tmp_path / "data")}}
        out = ingest([str(z)], cfg)
        out_path = Path(out)
        assert out_path.exists()
        lines = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
        assert len(lines) > 0
        # all lines are valid EvidenceUnit JSON
        for line in lines:
            eu = EvidenceUnit(**line)
            assert eu.id != ""

    def test_ingest_deduplicates(self, tmp_path):
        py_content = SAMPLE_PY
        z1 = self._make_zip(tmp_path / "a.zip", {"main.py": py_content})
        z2 = self._make_zip(tmp_path / "b.zip", {"main.py": py_content})
        cfg = {"paths": {"data_dir": str(tmp_path / "data")}}
        out = ingest([str(z1), str(z2)], cfg)
        lines = Path(out).read_text().splitlines()
        ids = [json.loads(l)["id"] for l in lines if l.strip()]
        assert len(ids) == len(set(ids))

    def test_ingest_plain_file(self, tmp_path):
        src = tmp_path / "mod.py"
        src.write_text(SAMPLE_PY, encoding="utf-8")
        cfg = {"paths": {"data_dir": str(tmp_path / "data")}}
        out = ingest([str(src)], cfg)
        lines = Path(out).read_text().splitlines()
        assert len(lines) > 0

    def test_ingest_missing_path_skipped(self, tmp_path):
        cfg = {"paths": {"data_dir": str(tmp_path / "data")}}
        out = ingest([str(tmp_path / "nonexistent.zip")], cfg)
        lines = Path(out).read_text().splitlines()
        assert lines == []

    def test_ingest_empty_list(self, tmp_path):
        cfg = {"paths": {"data_dir": str(tmp_path / "data")}}
        out = ingest([], cfg)
        assert Path(out).exists()
        assert Path(out).read_text().strip() == ""

    def test_ingest_zip_slip_rejected_gracefully(self, tmp_path):
        z = tmp_path / "slip.zip"
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr("../../../evil.txt", "pwned")
        cfg = {"paths": {"data_dir": str(tmp_path / "data")}}
        # should not raise; slip zip is skipped
        out = ingest([str(z)], cfg)
        lines = Path(out).read_text().splitlines()
        assert lines == []

    def test_ingest_directory(self, tmp_path):
        src_dir = tmp_path / "project"
        src_dir.mkdir()
        (src_dir / "a.py").write_text(SAMPLE_PY, encoding="utf-8")
        (src_dir / "b.py").write_text("x = 42\n", encoding="utf-8")
        cfg = {"paths": {"data_dir": str(tmp_path / "data")}}
        out = ingest([str(src_dir)], cfg)
        lines = Path(out).read_text().splitlines()
        assert len(lines) >= 2
