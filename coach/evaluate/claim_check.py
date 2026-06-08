"""Claim grounding: split an answer into atomic claims and check each against
project evidence (L1 lexical recall + L2 offline NLI heuristic; optional LLM).

Ported from interview-prep/claim_check.py. The pure functions (split_claims,
classify_claim, nli_score, route_verdict) are network-free and deterministic so
they unit-test cleanly. Verdict routing:
  entail -> verified / contradict -> rejected / neutral|miss -> needs_evidence.
``grounding_rate`` is the verified fraction.

Evidence is a list of ``EvidenceUnit`` (schemas.py); we index on its ``.text``
and ``.symbol`` and report matches by ``.id``.
"""
from __future__ import annotations

import re
from typing import Optional

from coach.schemas import ClaimCheck, ClaimVerdict, EvidenceUnit

# --- lexical signals --------------------------------------------------------

# Negation cues (zh + en) for polarity-conflict detection.
_NEG_ASCII = {
    "not", "no", "never", "without", "cannot", "won't",
    "none", "neither", "nor", "lack", "lacks", "missing",
}
_NEG_ZH = ("不", "没", "无", "非", "未", "别", "勿", "禁")

# Quantitative / metric cues: a numeric or quantified assertion needs numeric
# evidence to be entailed, otherwise it stays neutral (needs more evidence).
_QUANT_HINT = re.compile(
    r"\d|百分|倍|万|亿|千|qps|tps|sec|占比|提升|降低|减少|增加|[\d]\s*(?:个|次|ms|秒|率)",
    re.I,
)

_SKILL_HINT = re.compile(
    r"熟悉|掌握|精通|擅长|使用|基于|采用|skill|java|python|kafka|redis|mysql|spring|"
    r"docker|k8s|线程|并发|框架|中间件|语言|栈",
    re.I,
)
_DECISION_HINT = re.compile(
    r"选型|选择|采用|决定|改用|引入|替换|权衡|取舍|为了|因此|所以|方案|架构|设计为|拆分|下沉",
    re.I,
)


def tokenize(text: str) -> list[str]:
    """Mixed zh/en tokenizer: camel/snake split + lowercase; Chinese char + 2-gram."""
    toks: list[str] = []
    for m in re.findall(r"[A-Za-z][A-Za-z0-9_]*", text or ""):
        for p in re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+", m):
            toks.append(p.lower())
    for seg in re.findall(r"[一-鿿]+", text or ""):
        toks.extend([seg] if len(seg) == 1 else [seg[i:i + 2] for i in range(len(seg) - 1)])
    return toks


def _keywords(text: str) -> set[str]:
    """Discriminative keyword set: en identifiers (lowercased) + Chinese 2-grams."""
    kws: set[str] = set()
    for m in re.findall(r"[A-Za-z][A-Za-z0-9_]{1,}", text or ""):
        kws.add(m.lower())
    for seg in re.findall(r"[一-鿿]{2,}", text or ""):
        for i in range(len(seg) - 1):
            kws.add(seg[i:i + 2])
    return kws


def _has_negation(text: str) -> bool:
    t = (text or "").lower()
    if any(w in t for w in _NEG_ZH):
        return True
    return bool(set(tokenize(t)) & _NEG_ASCII)


# --- claim splitting / classification --------------------------------------

def classify_claim(text: str) -> str:
    """Classify a claim as metric/decision/skill/fact (priority in that order)."""
    t = text or ""
    if _QUANT_HINT.search(t):
        return "metric"
    if _DECISION_HINT.search(t):
        return "decision"
    if _SKILL_HINT.search(t):
        return "skill"
    return "fact"


def split_claims(text: str) -> list[str]:
    """Split an answer/resume blob into atomic claim strings (pure, offline).

    Splits on zh/en sentence punctuation, newlines and semicolons after
    stripping bullet/numbering prefixes; drops fragments shorter than 6 chars or
    without any letter/CJK content; de-dupes while preserving order.
    """
    if not text:
        return []
    norm = re.sub(r"^[\s\-\*•\d\.\)、]+", "", text, flags=re.M)
    pieces = re.split(r"[。!?!?；;\n]+|(?<=[一-鿿]),", norm)
    seen: set[str] = set()
    claims: list[str] = []
    for p in pieces:
        s = p.strip(" \t,，.。;；:：-—·")
        if len(s) < 6:
            continue
        if not (re.search(r"[一-鿿]", s) or re.search(r"[A-Za-z]{2,}", s)):
            continue
        if s in seen:
            continue
        seen.add(s)
        claims.append(s)
    return claims


# --- evidence canonicalization ---------------------------------------------

def _symbol_name(symbol: str) -> str:
    """Name part of 'class_definition:Foo' / 'function_definition:bar'; '' for line blocks."""
    if not symbol:
        return ""
    if symbol.startswith("lines:"):
        return ""
    return symbol.split(":", 1)[1] if ":" in symbol else symbol


def canonicalize_evidence(unit: EvidenceUnit, max_lines: int = 6, max_chars: int = 400) -> str:
    """Reduce an EvidenceUnit to entailment-relevant text: symbol name + key lines."""
    if unit is None:
        return ""
    name = _symbol_name(unit.symbol or "")
    snippet = unit.text or ""
    head = [name] if name else []
    picked = []
    for raw in snippet.replace("\r\n", "\n").split("\n"):
        ln = raw.strip()
        if not ln:
            continue
        if re.fullmatch(r"[{}()\[\];,:.]+", ln):
            continue
        picked.append(ln)
        if len(picked) >= max_lines:
            break
    text = re.sub(r"\s+", " ", " ".join(head + picked)).strip()
    return text[:max_chars]


# --- NLI scoring + routing --------------------------------------------------

def nli_score(claim_text: str, evidence_text: str) -> dict:
    """Offline heuristic lexical-entailment score (pure, no network).

    Returns ``{"label": entail|neutral|contradict, "score": 0..1, "overlap": float}``.
    overlap = |claim_kw & ev_kw| / |claim_kw|. High overlap with matching
    negation polarity -> entail; high overlap with opposite polarity ->
    contradict; otherwise neutral. Metric claims need numeric evidence to entail.
    """
    ck = _keywords(claim_text)
    ek = _keywords(evidence_text)
    if not ck:
        return {"label": "neutral", "score": 0.0, "overlap": 0.0}
    inter = ck & ek
    overlap = len(inter) / len(ck)
    polarity_conflict = (_has_negation(claim_text) != _has_negation(evidence_text)) and len(inter) >= 2

    is_quant = bool(_QUANT_HINT.search(claim_text or ""))
    ev_has_num = bool(re.search(r"\d", evidence_text or ""))
    entail_th = 0.55 if is_quant else 0.4

    if overlap >= entail_th and polarity_conflict:
        return {"label": "contradict", "score": round(min(1.0, overlap), 4), "overlap": round(overlap, 4)}
    if overlap >= entail_th and (not is_quant or ev_has_num):
        return {"label": "entail", "score": round(min(1.0, overlap), 4), "overlap": round(overlap, 4)}
    return {"label": "neutral", "score": round(overlap, 4), "overlap": round(overlap, 4)}


def route_verdict(label: str) -> ClaimVerdict:
    """Map an NLI label to a ClaimVerdict."""
    return {
        "entail": ClaimVerdict.verified,
        "contradict": ClaimVerdict.rejected,
    }.get(label, ClaimVerdict.needs_evidence)


# --- L1 recall over evidence ------------------------------------------------

def _l1_retrieve(claim_text: str, index: list[dict], k: int = 5) -> list[tuple[float, dict]]:
    """Recall top-k evidence by lexical overlap; symbol-name mention is weighted."""
    ck = _keywords(claim_text)
    if not ck:
        return []
    scored: list[tuple[float, dict]] = []
    for u in index:
        inter = ck & u["kw"]
        if not inter:
            continue
        overlap = len(inter) / len(ck)
        if u["name"] and u["name"].lower() in ck:
            overlap = min(1.0, overlap + 0.2)
        scored.append((overlap, u))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]


def _build_index(evidence: list[EvidenceUnit]) -> list[dict]:
    """Precompute canonical text + keyword set per evidence unit for L1/L2 reuse."""
    index: list[dict] = []
    for u in evidence or []:
        text = canonicalize_evidence(u)
        if not text:
            continue
        index.append({
            "id": u.id,
            "symbol": u.symbol,
            "name": _symbol_name(u.symbol or ""),
            "text": text,
            "kw": _keywords(text),
        })
    return index


# --- optional LLM-judge -----------------------------------------------------

_LLM_SYS = (
    "你是严格的事实核验员(NLI 判定器)。给定一条候选人声明(claim)与若干项目代码证据片段, "
    "判断证据对该声明的支持关系。只输出 JSON, 不要多余文字。"
)


def _llm_nli(claim_text: str, evidence_texts: list[str], gateway) -> Optional[dict]:
    """Optional LLM entailment judgement. Returns None on failure (caller falls back offline)."""
    ev = "\n".join(f"- {t}" for t in evidence_texts[:5]) or "(无相关证据)"
    user = (
        f"声明:\n{claim_text}\n\n项目代码证据片段:\n{ev}\n\n"
        "请判定证据与声明的关系, 严格输出 JSON:\n"
        '{"label":"entail|neutral|contradict","score":0.0,"reason":"简述依据"}'
    )
    try:
        from coach.llm.gateway import extract_json
        raw = gateway.complete([
            {"role": "system", "content": _LLM_SYS},
            {"role": "user", "content": user},
        ])
    except Exception:
        return None
    obj = extract_json(raw or "")
    if not obj or obj.get("label") not in ("entail", "neutral", "contradict"):
        return None
    try:
        obj["score"] = round(float(obj.get("score", 0.0)), 4)
    except Exception:
        obj["score"] = 0.0
    return obj


# --- public API -------------------------------------------------------------

def check_claims(
    claims: list[str],
    evidence: list[EvidenceUnit],
    gateway=None,
    *,
    k: int = 5,
) -> list[ClaimCheck]:
    """Ground each claim against evidence, returning a ClaimCheck per claim.

    L1 recalls top-k evidence by lexical overlap; with no hit the claim is
    ``needs_evidence``. L2 picks the strongest relation among the hits offline
    (contradict > entail > best-score). If a ``gateway`` is provided it refines
    the verdict via LLM-judge, falling back to the offline label on any failure.
    The matched evidence ids and the relation score populate the ClaimCheck.
    """
    index = _build_index(evidence)
    results: list[ClaimCheck] = []
    for claim in claims:
        hits = _l1_retrieve(claim, index, k=k)
        if not hits:
            results.append(ClaimCheck(
                claim=claim, verdict=ClaimVerdict.needs_evidence, evidence_ids=[], score=0.0))
            continue

        ev_ids = [u["id"] for _, u in hits]
        ev_texts = [u["text"] for _, u in hits]

        # offline: pick the strongest relation among recalled evidence.
        best = None
        for t in ev_texts:
            s = nli_score(claim, t)
            if best is None:
                best = s
                continue
            if s["label"] == "contradict":
                best = s
                break
            if s["label"] == "entail" and best["label"] != "entail":
                best = s
            elif s["score"] > best["score"] and best["label"] != "entail":
                best = s

        label = best["label"]
        score = float(best["score"])
        if gateway is not None:
            judged = _llm_nli(claim, ev_texts, gateway)
            if judged is not None:
                label = judged["label"]
                score = float(judged.get("score", score))

        results.append(ClaimCheck(
            claim=claim,
            verdict=route_verdict(label),
            evidence_ids=ev_ids,
            score=round(score, 4),
        ))
    return results


def grounding_rate(checks: list[ClaimCheck]) -> float:
    """Fraction of checks with a 'verified' verdict (0.0 when there are none)."""
    if not checks:
        return 0.0
    verified = sum(1 for c in checks if c.verdict == ClaimVerdict.verified)
    return round(verified / len(checks), 4)
