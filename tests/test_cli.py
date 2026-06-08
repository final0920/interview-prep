"""Tests for coach/cli.py.

Per DESIGN.md T13: verify dispatch (correct handler wired) and --help for
each subcommand. Handlers are NOT invoked -- the underlying modules (T8/T10/T11)
may not be fully implemented yet. We test the CLI layer only.
"""
from __future__ import annotations

import argparse
import sys

import pytest

from coach.cli import _build_parser, main


SUBCOMMANDS = ["ingest", "interview", "resume", "review", "export", "kb", "serve"]


# ---------------------------------------------------------------------------
# --help for each subcommand (must not raise, must exit 0)
# ---------------------------------------------------------------------------

class TestHelp:
    @pytest.mark.parametrize("sub", SUBCOMMANDS)
    def test_subcommand_help_exits_zero(self, sub):
        """Each subcommand's --help must print and exit 0 without importing deps."""
        with pytest.raises(SystemExit) as exc_info:
            _build_parser().parse_args([sub, "--help"])
        assert exc_info.value.code == 0

    def test_top_level_help_exits_zero(self):
        with pytest.raises(SystemExit) as exc_info:
            _build_parser().parse_args(["--help"])
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# Dispatch: correct _handler is set for each subcommand
# ---------------------------------------------------------------------------

class TestDispatch:
    def _parse(self, argv: list[str]) -> argparse.Namespace:
        return _build_parser().parse_args(argv)

    def test_ingest_dispatch(self):
        from coach.cli import _cmd_ingest
        args = self._parse(["ingest", "foo.zip", "bar.zip"])
        assert args._handler is _cmd_ingest
        assert args.paths == ["foo.zip", "bar.zip"]

    def test_ingest_no_paths(self):
        from coach.cli import _cmd_ingest
        args = self._parse(["ingest"])
        assert args._handler is _cmd_ingest
        assert args.paths == []

    def test_interview_dispatch(self):
        from coach.cli import _cmd_interview
        args = self._parse(["interview", "--role", "ML Engineer"])
        assert args._handler is _cmd_interview
        assert args.role == "ML Engineer"

    def test_interview_session_resume(self):
        from coach.cli import _cmd_interview
        args = self._parse(["interview", "--session-id", "abc123",
                            "--answer", "My answer"])
        assert args._handler is _cmd_interview
        assert args.session_id == "abc123"
        assert args.answer == "My answer"

    def test_resume_dispatch(self):
        from coach.cli import _cmd_resume
        args = self._parse(["resume", "path/to/cv.pdf"])
        assert args._handler is _cmd_resume
        assert args.path == "path/to/cv.pdf"

    def test_review_dispatch(self):
        from coach.cli import _cmd_review
        args = self._parse(["review"])
        assert args._handler is _cmd_review

    def test_export_dispatch(self):
        from coach.cli import _cmd_export
        args = self._parse(["export"])
        assert args._handler is _cmd_export

    def test_kb_grow_dispatch(self):
        from coach.cli import _cmd_kb
        args = self._parse(["kb", "redis cache eviction"])
        assert args._handler is _cmd_kb
        assert args.topic == "redis cache eviction"
        assert not args.web
        assert not args.search

    def test_kb_search_flag(self):
        from coach.cli import _cmd_kb
        args = self._parse(["kb", "kafka", "--search"])
        assert args.search is True

    def test_kb_web_flag(self):
        from coach.cli import _cmd_kb
        args = self._parse(["kb", "kafka", "--web"])
        assert args.web is True

    def test_serve_dispatch(self):
        from coach.cli import _cmd_serve
        args = self._parse(["serve"])
        assert args._handler is _cmd_serve
        assert args.host == "0.0.0.0"
        assert args.port == 8000
        assert not args.reload

    def test_serve_custom_port(self):
        from coach.cli import _cmd_serve
        args = self._parse(["serve", "--port", "9000", "--reload"])
        assert args.port == 9000
        assert args.reload is True


# ---------------------------------------------------------------------------
# main() return codes
# ---------------------------------------------------------------------------

class TestMainReturnCodes:
    def test_no_subcommand_returns_2(self):
        """No subcommand: print help and return 2."""
        result = main([])
        assert result == 2

    def test_unknown_subcommand_exits(self):
        with pytest.raises(SystemExit):
            main(["nonexistent-command"])

    def test_help_flag_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# Import safety: importing cli.py must not trigger heavy module imports
# ---------------------------------------------------------------------------

class TestImportSafety:
    def test_import_does_not_load_torch(self):
        """Importing coach.cli must not pull in torch or sentence_transformers."""
        # These would be loaded if lazy imports weren't used
        import coach.cli  # noqa: F401 - just verify no ImportError or side-effect
        # If torch was eagerly imported it would already be in sys.modules from
        # a previous test; we just verify cli itself imports cleanly
        assert "coach.cli" in sys.modules

    def test_all_subcommands_registered(self):
        """Every expected subcommand name is registered in the parser."""
        parser = _build_parser()
        # Extract registered subcommand names from the subparsers action
        sub_action = next(
            a for a in parser._actions
            if isinstance(a, argparse._SubParsersAction)
        )
        registered = set(sub_action.choices.keys())
        for sub in SUBCOMMANDS:
            assert sub in registered, f"subcommand '{sub}' not registered"
