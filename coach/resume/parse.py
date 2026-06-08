"""Resume parse: PII redaction + heuristic structured extraction.

Public API:
    def redact_pii(text) -> tuple[str, dict]
    def parse_resume(pdf_or_text, *, llm=None) -> ResumeProfile
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from coach.schemas import ResumeProfile, ResumeProject


# ---------------------------------------------------------------------------
# PII patterns: phone / email / ID card / bank card
# Order matters: ID card (18 digits) must precede bank card (16-19 digits)
# ---------------------------------------------------------------------------

_PII_PATTERNS: list[tuple[str, str, str]] = [
    ("phone",     r"(?<!\d)1[3-9]\d{9}(?!\d)",                                  "<PHONE>"),
    ("email",     r"[\w.\-]+@[\w.\-]+\.\w+",                                     "<EMAIL>"),
    ("id_card",
     r"(?<!\d)[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?![\dXx])",
     "<ID_CARD>"),
    ("bank_card", r"(?<!\d)\d{16,19}(?!\d)",                                     "<BANK_CARD>"),
]


def redact_pii(text: str) -> tuple[str, dict]:
    """Regex-redact PII (phone/email/ID card/bank card) with placeholder tokens.

    Returns (redacted_text, coverage_dict).  coverage_dict keys:
        pii_masked  : always True
        counts      : {kind: n, ...}
        total_pii   : total PII matches found
        masked_pii  : same (all are replaced)
        coverage    : 1.0 (redact-on-detect => always 100 %)
        spans       : [{type, placeholder, raw_len}, ...] for auditing
    """
    if text is None:
        text = ""
    masked = str(text)
    counts = {key: 0 for key, _, _ in _PII_PATTERNS}
    spans: list[dict] = []

    for key, pat, placeholder in _PII_PATTERNS:
        rx = re.compile(pat)

        def _sub(m, _k=key, _ph=placeholder):
            counts[_k] += 1
            spans.append({"type": _k, "placeholder": _ph, "raw_len": len(m.group(0))})
            return _ph

        masked = rx.sub(_sub, masked)

    total = sum(counts.values())
    return masked, {
        "pii_masked": True,
        "counts": counts,
        "total_pii": total,
        "masked_pii": total,
        "coverage": 1.0,
        "spans": spans,
    }


# ---------------------------------------------------------------------------
# Section segmentation
# ---------------------------------------------------------------------------

_SECTION_ALIASES: dict[str, list[str]] = {
    "basic":      ["基本信息", "个人信息", "基本资料", "个人资料", "求职意向", "联系方式"],
    "education":  ["教育经历", "教育背景", "学习经历", "教育"],
    "experience": ["工作经历", "工作经验", "实习经历", "职业经历", "工作"],
    "project":    ["项目经历", "项目经验", "项目"],
    "skill":      ["技能", "专业技能", "技能特长", "技术栈", "专业技术", "掌握技能"],
}

_TITLE_MAX_LEN = 20
_TITLE_PREFIX_RE = re.compile(r"^[\s\d.、)\)\-—*#●▪◆■·:：]*")
_KV_RE = re.compile(r"^([一-鿿A-Za-z]{1,12})\s*[:：]\s*(.+)$")


def _match_section_header(line: str) -> Optional[str]:
    """Return canonical section key if line looks like a section heading, else None."""
    raw = (line or "").strip()
    if not raw:
        return None
    core = _TITLE_PREFIX_RE.sub("", raw).strip()
    # key: value lines (colon with non-empty rhs) are fields, not headings
    kv = _KV_RE.match(core)
    if kv and kv.group(2).strip():
        return None
    core_nocolon = core.rstrip(":：").strip()
    if len(core_nocolon) > _TITLE_MAX_LEN:
        return None
    for key, aliases in _SECTION_ALIASES.items():
        for alias in aliases:
            if alias in core_nocolon:
                return key
    return None


def _segment(text: str) -> list[dict]:
    """Split resume text into sections; each dict has section/title/offset/end/text."""
    headers: list[tuple[int, int, str, str]] = []
    pos = 0
    for line in text.splitlines(keepends=True):
        key = _match_section_header(line.rstrip("\r\n"))
        if key is not None:
            headers.append((pos, pos + len(line), key, line.rstrip("\r\n").strip()))
        pos += len(line)

    sections: list[dict] = []
    first_pos = headers[0][0] if headers else len(text)
    if first_pos > 0:
        pre = text[:first_pos]
        if pre.strip():
            sections.append({"section": "preamble", "title": "",
                             "offset": 0, "end": first_pos, "text": pre})

    for idx, (hpos, cstart, key, title) in enumerate(headers):
        cend = headers[idx + 1][0] if idx + 1 < len(headers) else len(text)
        sections.append({"section": key, "title": title,
                         "offset": cstart, "end": cend, "text": text[cstart:cend]})
    return sections


# ---------------------------------------------------------------------------
# Field extractors
# ---------------------------------------------------------------------------

_DATE_RANGE_RE = re.compile(
    r"(\d{4}(?:[.\-/]\d{1,2})?)\s*[\-—~～至到]+\s*"
    r"(\d{4}(?:[.\-/]\d{1,2})?|至今|今|现在|now)"
)
_SKILL_SEP_RE = re.compile(r"[、,，;；/|\s]+")
_TECH_KEYWORDS = [
    "Java", "Go", "Golang", "Python", "C++", "Spring", "Spring Boot", "Spring Cloud",
    "Kafka", "RocketMQ", "RabbitMQ", "Redis", "Memcached", "MySQL", "PostgreSQL",
    "Oracle", "MongoDB", "Elasticsearch",
    "LLM", "RAG", "Embedding", "Faiss", "Milvus", "Prompt", "Agent", "GPT",
    "Docker", "Kubernetes", "K8s", "gRPC", "Nginx", "Netty", "Dubbo", "Zookeeper",
    "分布式", "微服务", "高并发", "高可用", "缓存", "消息队列", "检索增强", "向量检索",
    "大模型", "Prompt工程",
]
_TECH_RE: list[tuple[str, re.Pattern]] = [
    (kw, re.compile(
        re.escape(kw) if re.search(r"[一-鿿]", kw)
        else r"(?<![A-Za-z0-9])" + re.escape(kw) + r"(?![A-Za-z0-9])",
        re.IGNORECASE,
    ))
    for kw in _TECH_KEYWORDS
]


def _nonempty_lines(block: str, base_offset: int):
    """Yield (stripped_line, offset_in_original_text) for non-blank lines."""
    pos = base_offset
    for line in block.splitlines(keepends=True):
        s = line.strip()
        if s:
            yield s, pos
        pos += len(line)


def _guess_skills(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for kw, rx in _TECH_RE:
        if rx.search(text) and kw not in seen:
            seen.add(kw)
            out.append(kw)
    return out


def _guess_org(header: str) -> str:
    if not header:
        return ""
    h = _DATE_RANGE_RE.sub(" ", header).strip()
    parts = [p for p in re.split(r"[\s,，、|]+", h) if p]
    return parts[0] if parts else ""


_FIELD_MAP: dict[str, str] = {
    "姓名": "name", "名字": "name",
    "求职意向": "target_role", "应聘岗位": "target_role", "意向岗位": "target_role",
    "目标岗位": "target_role", "期望职位": "target_role",
    "电话": "phone", "手机": "phone", "联系电话": "phone",
    "邮箱": "email", "邮件": "email", "电子邮箱": "email", "email": "email",
    "学历": "degree", "年龄": "age", "性别": "gender",
    "城市": "city", "所在地": "city", "现居": "city",
}


def _extract_basics(sections: list[dict]) -> dict:
    basics: dict = {}
    for sec in sections:
        if sec["section"] not in ("basic", "preamble"):
            continue
        for line, _ in _nonempty_lines(sec["text"], sec["offset"]):
            m = _KV_RE.match(line)
            if not m:
                continue
            raw_key = m.group(1).strip()
            val = m.group(2).strip()
            key = _FIELD_MAP.get(raw_key) or _FIELD_MAP.get(raw_key.lower())
            if key and key not in basics:
                basics[key] = val
    return basics


def _split_entries_by_date(sec: dict) -> list[dict]:
    """Split a section's lines into entries delimited by date-range lines."""
    entries: list[dict] = []
    cur: Optional[dict] = None
    for line, loff in _nonempty_lines(sec["text"], sec["offset"]):
        dm = _DATE_RANGE_RE.search(line)
        if dm is not None:
            cur = {"header": line, "period": f"{dm.group(1)}-{dm.group(2)}",
                   "offset": loff, "bullets": []}
            entries.append(cur)
        else:
            if cur is None:
                cur = {"header": line, "period": "", "offset": loff, "bullets": []}
                entries.append(cur)
            else:
                cur["bullets"].append((line, loff))
    return entries


def _extract_experiences(sections: list[dict]) -> list[dict]:
    out: list[dict] = []
    for sec in sections:
        if sec["section"] != "experience":
            continue
        for ent in _split_entries_by_date(sec):
            bullets = [b for b, _ in ent["bullets"]]
            boffsets = [o for _, o in ent["bullets"]]
            skills = _guess_skills(" ".join([ent["header"]] + bullets))
            out.append({
                "title": ent["header"],
                "period": ent["period"],
                "company": _guess_org(ent["header"]),
                "role": "",
                "description": "\n".join(bullets),
                "tech": skills,
                "bullets": bullets,
                "skills": skills,
                "offset": ent["offset"],
                "bullet_offsets": boffsets,
            })
    return out


def _extract_projects(sections: list[dict]) -> list[ResumeProject]:
    out: list[ResumeProject] = []
    for sec in sections:
        if sec["section"] != "project":
            continue
        for ent in _split_entries_by_date(sec):
            bullets = [b for b, _ in ent["bullets"]]
            skills = _guess_skills(" ".join([ent["header"]] + bullets))
            out.append(ResumeProject(
                name=ent["header"],
                role="",
                description="\n".join(bullets),
                tech=skills,
                offset=ent["offset"],
            ))
    return out


def _extract_skills(sections: list[dict]) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    for sec in sections:
        if sec["section"] != "skill":
            continue
        for tok in _SKILL_SEP_RE.split(sec["text"]):
            tok = tok.strip(" :：·-—()（）")
            if not tok or len(tok) > 24:
                continue
            if tok not in seen:
                seen.add(tok)
                items.append(tok)
    return items


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def _extract_pdf_text(path: str) -> str:
    """Extract plain text from a PDF with PyMuPDF; returns '' on failure."""
    try:
        import fitz
        doc = fitz.open(path)
        pages = [doc[i].get_text("text") or "" for i in range(len(doc))]
        doc.close()
        return "\n".join(pages).strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_resume(
    pdf_or_text: str,
    *,
    llm=None,
) -> ResumeProfile:
    """Parse a resume from a file path (PDF) or raw text string.

    Steps:
      1. If pdf_or_text looks like an existing file path, read text via PyMuPDF.
      2. redact_pii on the raw text.
      3. Heuristic section segmentation + field extraction -> ResumeProfile.
      4. Optional LLM correction: if llm (LLMGateway) is provided, call it to
         fill in fields the heuristics may have missed; silently falls back to
         the heuristic result on any error.

    Works fully offline without llm (no network, no model download).
    """
    # --- resolve input ---
    raw_text: str
    if pdf_or_text and Path(pdf_or_text).exists():
        raw_text = _extract_pdf_text(pdf_or_text) or pdf_or_text
    else:
        raw_text = pdf_or_text or ""

    masked, _pii = redact_pii(raw_text)
    sections = _segment(masked)

    basics = _extract_basics(sections)
    experiences = _extract_experiences(sections)
    projects = _extract_projects(sections)
    skills = _extract_skills(sections)

    # education: collect raw lines
    education: list[dict] = []
    for sec in sections:
        if sec["section"] == "education":
            for line, _ in _nonempty_lines(sec["text"], sec["offset"]):
                dm = _DATE_RANGE_RE.search(line)
                education.append({
                    "raw": line,
                    "period": f"{dm.group(1)}-{dm.group(2)}" if dm else "",
                })

    profile = ResumeProfile(
        basics=basics,
        education=education,
        experiences=experiences,
        projects=projects,
        skills=skills,
        raw_text=masked,
    )

    # optional LLM enhancement (never raises)
    if llm is not None:
        profile = _llm_enhance(profile, masked, llm)

    return profile


def _llm_enhance(profile: ResumeProfile, masked_text: str, llm) -> ResumeProfile:
    """Ask the LLM to correct/complete the heuristic profile; fall back silently."""
    import json

    sys_msg = (
        "You are a resume-parsing expert. Given a PII-redacted resume and a heuristic "
        "parse result (JSON), correct and complete the parse. "
        "Return only JSON with keys: basics, education, experiences, projects, skills. "
        "Do not invent information not present in the resume text."
    )
    slim = {
        "basics": profile.basics,
        "education": profile.education,
        "experiences": [
            {"title": e.get("title", ""), "period": e.get("period", ""),
             "bullets": e.get("bullets", []), "skills": e.get("skills", [])}
            for e in profile.experiences
        ],
        "projects": [
            {"name": p.name, "tech": p.tech, "description": p.description}
            for p in profile.projects
        ],
        "skills": profile.skills,
    }
    user_msg = (
        f"Resume (PII-redacted):\n{masked_text}\n\n"
        f"Heuristic parse:\n{json.dumps(slim, ensure_ascii=False, indent=2)}\n\n"
        "Return corrected JSON (same structure)."
    )
    try:
        result = llm.structured(
            [{"role": "system", "content": sys_msg},
             {"role": "user",   "content": user_msg}],
            ResumeProfile,
        )
        # preserve raw_text from our own extraction
        return result.model_copy(update={"raw_text": profile.raw_text})
    except Exception:
        return profile
