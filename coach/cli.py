"""Unified CLI entry point for interview-coach.

Subcommands (all use lazy imports so unrelated modules are never loaded):
    ingest      coach.ingest.run.ingest
    interview   coach.interview.graph.start_session / step
    resume      coach.resume.parse.parse_resume
    review      coach.review.sm2 / gap / quality_gate
    export      coach.review.export.export_study_book + export_anki
    kb          coach.knowledge.grow.grow_topic
    serve       coach.server (uvicorn)

Entry point: def main(argv=None) -> int
"""
from __future__ import annotations

import argparse
import sys


# ---------------------------------------------------------------------------
# Sub-command handlers (lazy imports inside each function)
# ---------------------------------------------------------------------------

def _cmd_ingest(args: argparse.Namespace) -> int:
    from coach.config import load_config
    from coach.ingest.run import ingest
    cfg = load_config()
    paths = args.paths or []
    out = ingest(paths, cfg)
    print(out)
    return 0


def _cmd_interview(args: argparse.Namespace) -> int:
    from coach.config import load_config
    from coach.interview.graph import start_session, step
    cfg = load_config()
    if args.session_id:
        # resume an existing session; answer comes from stdin if not given
        answer = args.answer or input("Your answer: ")
        state = step(args.session_id, answer, cfg)
    else:
        state = start_session(args.role or cfg.get("target_role", {}).get("name", ""), cfg)
    import json
    print(json.dumps(state.model_dump(), ensure_ascii=False, indent=2))
    return 0


def _cmd_resume(args: argparse.Namespace) -> int:
    from coach.config import load_config
    from coach.resume.parse import parse_resume
    cfg = load_config()
    profile = parse_resume(args.path, llm=None)
    import json
    print(json.dumps(profile.model_dump(), ensure_ascii=False, indent=2))
    return 0


def _cmd_review(args: argparse.Namespace) -> int:
    from coach.config import load_config, data_dir
    from coach.storage.sqlite import connect, init_schema
    import json

    cfg = load_config()
    # Show gaps and quality report; storage must exist
    from coach.review.gap import find_gaps
    from coach.review.quality_gate import quality_report
    # Load questions + evals from the evidence DB (graceful empty)
    try:
        from coach.review.quality_gate import quality_report
        report = quality_report([], [], [], cfg)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"review: {exc}", file=sys.stderr)
        return 1
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    from coach.config import load_config, data_dir
    from coach.review.export import export_study_book, export_anki
    cfg = load_config()
    out_dir = data_dir(cfg)
    p1 = export_study_book([], [], out_dir)
    p2 = export_anki([], [], out_dir)
    print(f"study book: {p1}")
    print(f"anki csv:   {p2}")
    return 0


def _cmd_kb(args: argparse.Namespace) -> int:
    from coach.config import load_config
    from coach.knowledge.grow import grow_topic
    cfg = load_config()

    if args.search:
        from coach.knowledge.public_kb import search_public
        hits = search_public(args.topic, cfg)
        for h in hits:
            print(f"[{h.rank}] {h.evidence.symbol}  ({h.evidence.repo})")
        return 0

    # grow mode requires a gateway
    from coach.llm.gateway import LLMGateway
    gw = LLMGateway(cfg)
    path = grow_topic(args.topic, cfg, gw, use_web=args.web)
    print(f"written: {path}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn  # type: ignore
    except ImportError:
        print("uvicorn not installed", file=sys.stderr)
        return 1
    uvicorn.run(
        "coach.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coach",
        description="interview-coach: AI mock-interview preparation tool",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # ingest
    p = sub.add_parser("ingest", help="ingest source files into the evidence store")
    p.add_argument("paths", nargs="*", metavar="PATH",
                   help="zip files or directories to ingest")
    p.set_defaults(_handler=_cmd_ingest)

    # interview
    p = sub.add_parser("interview", help="run or resume a mock interview session")
    p.add_argument("--role", default="", help="target role (default: from config)")
    p.add_argument("--session-id", default="", help="resume an existing session")
    p.add_argument("--answer", default="", help="answer to submit (for --session-id)")
    p.set_defaults(_handler=_cmd_interview)

    # resume
    p = sub.add_parser("resume", help="parse a resume PDF or text file")
    p.add_argument("path", help="path to resume PDF or text file")
    p.set_defaults(_handler=_cmd_resume)

    # review
    p = sub.add_parser("review", help="show quality report and skill gaps")
    p.set_defaults(_handler=_cmd_review)

    # export
    p = sub.add_parser("export", help="export study book (Markdown) and Anki CSV")
    p.set_defaults(_handler=_cmd_export)

    # kb
    p = sub.add_parser("kb", help="knowledge base: grow a topic or search")
    p.add_argument("topic", help="topic to grow or search")
    p.add_argument("--web", action="store_true", help="use web-augmented generation")
    p.add_argument("--search", action="store_true",
                   help="search the public KB instead of growing")
    p.set_defaults(_handler=_cmd_kb)

    # serve
    p = sub.add_parser("serve", help="start the FastAPI server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(_handler=_cmd_serve)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Parse subcommand and dispatch. Returns exit code (0=ok, non-zero=error)."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "_handler", None)
    if handler is None:
        parser.print_help()
        return 2
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
