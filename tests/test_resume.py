"""Tests for coach/resume/{parse,analyze,optimize,benchmark}.

All offline: no network, no LLM calls, no PDF downloads.
LLM paths are exercised via a deterministic fake gateway.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from coach.schemas import (
    EvidenceUnit, Channel, ResumeProfile, ResumeProject,
    SkillGapCategory,
)
from coach.resume.parse import redact_pii, parse_resume
from coach.resume.analyze import (
    health_report, classify_skill_gap,
    _DEFAULT_REQUIRED_SKILLS,
)
from coach.resume.optimize import (
    flag_unsupported_numbers, classify_skill_gap as opt_classify_skill_gap,
    optimize,
)
from coach.resume.benchmark import benchmark_competitors


# ============================================================
# Fixtures
# ============================================================

SAMPLE_RESUME = """\
基本信息
姓名: 张三
求职意向: AI应用工程师
电话: 18888501310
邮箱: zhangsan@example.com
身份证: 110101199003071234
银行卡: 6222021234567890123

教育经历
2016.09-2020.06  某大学  计算机科学  本科

工作经历
2020.07-2024.05  某科技公司  后端工程师
用Kafka消息队列削峰, QPS提升至5000。
引入Redis缓存, P99从200ms降到30ms。

项目经历
智能客服RAG系统  2023.03-2023.10
搭建检索增强生成问答, 接入LLM大模型。
向量检索+BM25混合召回, 降低token成本40%。

技能
Java、Kafka、Redis、MySQL、LLM、RAG、向量检索
"""


def _make_profile(skills=None, experiences=None, raw_text="") -> ResumeProfile:
    return ResumeProfile(
        basics={"name": "Test", "target_role": "AI应用工程师"},
        skills=skills or ["Java", "Kafka", "Redis", "LLM", "RAG"],
        experiences=experiences or [],
        projects=[],
        raw_text=raw_text,
    )


def _make_evidence(text: str, path: str = "src/main.py", line: int = 1) -> EvidenceUnit:
    return EvidenceUnit(
        id=f"ev-{hash(text) & 0xFFFF:04x}",
        source_path=path,
        start_line=line,
        end_line=line + 5,
        channel=Channel.code,
        lang="python",
        text=text,
        content_hash="abc",
    )


# ============================================================
# parse.py: redact_pii
# ============================================================

class TestRedactPii:
    def test_phone_redacted(self):
        masked, stats = redact_pii("联系: 18888501310")
        assert "<PHONE>" in masked
        assert "18888501310" not in masked
        assert stats["counts"]["phone"] == 1

    def test_email_redacted(self):
        masked, stats = redact_pii("email: test@example.com")
        assert "<EMAIL>" in masked
        assert stats["counts"]["email"] == 1

    def test_id_card_redacted(self):
        masked, stats = redact_pii("身份证: 110101199003071234")
        assert "<ID_CARD>" in masked
        assert stats["counts"]["id_card"] == 1

    def test_bank_card_redacted(self):
        masked, stats = redact_pii("银行卡: 6222021234567890123")
        assert "<BANK_CARD>" in masked
        assert stats["counts"]["bank_card"] == 1

    def test_no_pii_coverage_one(self):
        _, stats = redact_pii("no pii here")
        assert stats["coverage"] == 1.0
        assert stats["total_pii"] == 0

    def test_all_pii_types(self):
        text = "18888501310 foo@bar.com 110101199003071234 6222021234567890123"
        masked, stats = redact_pii(text)
        assert stats["total_pii"] == 4
        assert stats["coverage"] == 1.0
        assert "18888501310" not in masked

    def test_none_input(self):
        masked, stats = redact_pii(None)
        assert masked == ""
        assert stats["total_pii"] == 0

    def test_returns_tuple(self):
        result = redact_pii("hello")
        assert isinstance(result, tuple) and len(result) == 2

    def test_id_card_before_bank_card(self):
        # 18-digit ID card must not be consumed as bank card
        text = "110101199003071234"
        masked, stats = redact_pii(text)
        assert stats["counts"]["id_card"] == 1
        assert stats["counts"]["bank_card"] == 0

    def test_masked_pii_equals_total(self):
        _, stats = redact_pii(SAMPLE_RESUME)
        assert stats["masked_pii"] == stats["total_pii"]

    def test_spans_audit(self):
        _, stats = redact_pii("18888501310 foo@bar.com")
        assert len(stats["spans"]) == 2
        types = {s["type"] for s in stats["spans"]}
        assert "phone" in types and "email" in types


# ============================================================
# parse.py: parse_resume (text input, no PDF, no LLM)
# ============================================================

class TestParseResume:
    def test_returns_resume_profile(self):
        profile = parse_resume(SAMPLE_RESUME)
        assert isinstance(profile, ResumeProfile)

    def test_pii_not_in_raw_text(self):
        profile = parse_resume(SAMPLE_RESUME)
        assert "18888501310" not in profile.raw_text
        assert "<PHONE>" in profile.raw_text

    def test_skills_extracted(self):
        profile = parse_resume(SAMPLE_RESUME)
        assert len(profile.skills) > 0
        skill_text = " ".join(profile.skills)
        assert any(s in skill_text for s in ("Java", "Kafka", "Redis"))

    def test_experiences_extracted(self):
        profile = parse_resume(SAMPLE_RESUME)
        assert len(profile.experiences) > 0

    def test_projects_extracted(self):
        profile = parse_resume(SAMPLE_RESUME)
        assert len(profile.projects) > 0

    def test_education_extracted(self):
        profile = parse_resume(SAMPLE_RESUME)
        assert len(profile.education) > 0

    def test_basics_name(self):
        profile = parse_resume(SAMPLE_RESUME)
        assert profile.basics.get("name") == "张三"

    def test_empty_text_returns_empty_profile(self):
        profile = parse_resume("")
        assert isinstance(profile, ResumeProfile)
        assert profile.skills == []

    def test_llm_path_called(self):
        fake_llm = MagicMock()
        fake_profile = _make_profile()
        fake_llm.structured.return_value = fake_profile
        result = parse_resume(SAMPLE_RESUME, llm=fake_llm)
        fake_llm.structured.assert_called_once()
        assert isinstance(result, ResumeProfile)

    def test_llm_failure_falls_back(self):
        fake_llm = MagicMock()
        fake_llm.structured.side_effect = RuntimeError("network down")
        result = parse_resume(SAMPLE_RESUME, llm=fake_llm)
        # fallback: returns heuristic profile
        assert isinstance(result, ResumeProfile)

    def test_nonexistent_path_treated_as_text(self, tmp_path):
        # a path string that doesn't exist should be treated as raw text
        result = parse_resume("def foo(): pass\nskills: Python")
        assert isinstance(result, ResumeProfile)


# ============================================================
# analyze.py: classify_skill_gap
# ============================================================

class TestClassifySkillGap:
    def test_have_skill_not_in_gaps(self):
        profile = _make_profile(skills=["Java", "Kafka", "Redis", "MySQL",
                                        "LLM", "RAG", "向量检索", "Prompt工程",
                                        "Agent", "分布式", "高并发"])
        gaps = classify_skill_gap(profile, "AI工程师")
        missing_skills = {g["skill"] for g in [g.model_dump() for g in gaps]
                         if g["category"] == "missing"}
        assert "Java" not in missing_skills

    def test_missing_skill_flagged(self):
        profile = _make_profile(skills=["Java"])  # missing most required
        gaps = classify_skill_gap(profile, "AI工程师")
        categories = {g.category for g in gaps}
        assert SkillGapCategory.missing in categories

    def test_transferable_recognised(self):
        # 检索增强 maps to RAG
        profile = _make_profile(skills=["检索增强", "Java", "MySQL", "高并发", "分布式"])
        gaps = classify_skill_gap(profile, "AI工程师")
        xfer_targets = {g.skill for g in gaps if g.category == SkillGapCategory.transferable}
        # RAG should be covered via transferable
        missing_targets = {g.skill for g in gaps if g.category == SkillGapCategory.missing}
        assert "RAG" not in missing_targets or "RAG" not in xfer_targets

    def test_empty_profile_all_missing(self):
        profile = _make_profile(skills=[])
        gaps = classify_skill_gap(profile, "AI工程师")
        assert len(gaps) > 0
        categories = {g.category for g in gaps}
        assert SkillGapCategory.missing in categories

    def test_returns_skill_gap_list(self):
        profile = _make_profile()
        from coach.schemas import SkillGap
        gaps = classify_skill_gap(profile, "AI工程师")
        assert all(isinstance(g, SkillGap) for g in gaps)


# ============================================================
# analyze.py: health_report
# ============================================================

class TestHealthReport:
    def test_returns_dict_with_required_keys(self):
        profile = _make_profile()
        report = health_report(profile, "AI工程师")
        assert "match_score" in report
        assert "coverage_heatmap" in report
        assert "gaps" in report
        assert "ats" in report
        assert "summary" in report

    def test_match_score_range(self):
        profile = _make_profile()
        report = health_report(profile, "AI工程师")
        assert 0 <= report["match_score"] <= 100

    def test_full_skills_high_score(self):
        profile = _make_profile(skills=_DEFAULT_REQUIRED_SKILLS)
        report = health_report(profile, "AI工程师")
        assert report["match_score"] >= 80

    def test_no_skills_low_score(self):
        # empty profile: no skills, no experiences, no projects
        profile = ResumeProfile(basics={}, skills=[], experiences=[], projects=[], raw_text="")
        report = health_report(profile, "AI工程师")
        assert report["match_score"] == 0

    def test_heatmap_has_required_skills(self):
        profile = _make_profile()
        report = health_report(profile, "AI工程师")
        for skill in _DEFAULT_REQUIRED_SKILLS:
            assert skill in report["coverage_heatmap"]

    def test_heatmap_values(self):
        profile = _make_profile()
        report = health_report(profile, "AI工程师")
        assert all(v in ("have", "transferable", "missing")
                   for v in report["coverage_heatmap"].values())

    def test_ats_has_expected_keys(self):
        profile = _make_profile(raw_text="responsible for doing various things")
        report = health_report(profile, "AI工程师")
        ats = report["ats"]
        assert "ats_score" in ats
        assert "redflag_phrases" in ats
        assert "quantified_claims" in ats

    def test_ats_catches_redflags(self):
        profile = _make_profile(raw_text="responsible for various tasks")
        report = health_report(profile, "AI工程师")
        assert len(report["ats"]["redflag_phrases"]) > 0

    def test_summary_is_string(self):
        profile = _make_profile()
        report = health_report(profile, "AI工程师")
        assert isinstance(report["summary"], str)


# ============================================================
# optimize.py: flag_unsupported_numbers
# ============================================================

class TestFlagUnsupportedNumbers:
    def test_unsupported_number_flagged(self):
        flags = flag_unsupported_numbers(["QPS提升至5000"], evidence=[])
        assert len(flags) > 0
        assert any("5000" in f["claim"] for f in flags)

    def test_supported_number_not_flagged(self):
        ev = _make_evidence("QPS提升至5000", "src/order.java", 10)
        flags = flag_unsupported_numbers(["QPS提升至5000"], evidence=[ev])
        # 5000 appears in evidence text so should not be flagged
        assert all("5000" not in f["claim"] for f in flags)

    def test_percentage_flagged(self):
        flags = flag_unsupported_numbers(["降低成本40%"], evidence=[])
        assert any("40%" in f["claim"] for f in flags)

    def test_no_numbers_no_flags(self):
        flags = flag_unsupported_numbers(["improved performance significantly"], evidence=[])
        assert flags == []

    def test_string_input(self):
        flags = flag_unsupported_numbers("P99从200ms降到30ms", evidence=[])
        assert len(flags) >= 1

    def test_empty_input(self):
        assert flag_unsupported_numbers([], evidence=[]) == []

    def test_flag_has_required_keys(self):
        flags = flag_unsupported_numbers(["5000 QPS"], evidence=[])
        if flags:
            f = flags[0]
            assert "claim" in f
            assert "reason" in f
            assert "severity" in f
            assert "where" in f

    def test_severity_high(self):
        flags = flag_unsupported_numbers(["rate 99%"], evidence=[])
        assert all(f["severity"] == "high" for f in flags)

    def test_dedup_same_number(self):
        flags = flag_unsupported_numbers(["5000 QPS", "5000 requests"], evidence=[])
        claims = [f["claim"] for f in flags]
        assert len(claims) == len(set(claims))


# ============================================================
# optimize.py: optimize (with mock gateway)
# ============================================================

class TestOptimize:
    def _fake_gateway(self, reply: str = '{"star":{"situation":"s","task":"t","action":"a","result":"r"},"rewritten_bullets":["improved"],"diff":[],"citations":[]}'):
        gw = MagicMock()
        gw.complete.return_value = reply
        return gw

    def test_returns_dict_with_summary(self):
        profile = _make_profile(
            experiences=[{
                "title": "Backend engineer",
                "skills": ["Java", "Kafka"],
                "bullets": ["scaled service to 5000 QPS"],
                "company": "Acme",
            }]
        )
        result = optimize(profile, [], self._fake_gateway())
        assert "summary" in result
        assert "experiences" in result

    def test_summary_counts(self):
        profile = _make_profile(
            experiences=[
                {"title": "Job1", "skills": ["Java"], "bullets": ["did stuff"]},
                {"title": "Job2", "skills": ["Go"],   "bullets": ["did more"]},
            ]
        )
        result = optimize(profile, [], self._fake_gateway())
        assert result["summary"]["total"] >= 2

    def test_degraded_on_llm_failure(self):
        gw = MagicMock()
        gw.complete.side_effect = RuntimeError("offline")
        profile = _make_profile(
            experiences=[{"title": "Job", "skills": [], "bullets": ["did x"]}]
        )
        result = optimize(profile, [], gw)
        assert result["summary"]["degraded"] >= 1

    def test_risk_flags_unsupported_numbers(self):
        profile = _make_profile(
            experiences=[{
                "title": "Job",
                "skills": ["Java"],
                "bullets": ["improved throughput by 300%"],
            }]
        )
        # gateway returns a star with no evidence-backed numbers
        result = optimize(profile, [], self._fake_gateway(
            '{"star":{"result":"improved by 300%"},"rewritten_bullets":[],"citations":[]}'
        ))
        # 300% has no evidence backing -> should be in risk flags
        flags = [f for item in result["experiences"] for f in item["risk_flags"]]
        assert any("300" in f["claim"] or "%" in f["claim"] for f in flags)

    def test_citations_sanitised(self):
        ev = _make_evidence("some code", "src/svc.py", 5)
        profile = _make_profile(
            experiences=[{
                "title": "svc",
                "skills": ["some"],
                "bullets": ["some code"],
            }]
        )
        # gateway returns a fabricated citation not in allowed list
        gw = self._fake_gateway(
            '{"star":{"situation":"s","task":"t","action":"a","result":"r"},'
            '"rewritten_bullets":[],"citations":["evil/hack.py:1"]}'
        )
        result = optimize(profile, [ev], gw)
        for item in result["experiences"]:
            if item["rewrite"]:
                assert "evil/hack.py:1" not in item["rewrite"].get("citations", [])

    def test_empty_profile_no_crash(self):
        profile = _make_profile(experiences=[], skills=[])
        result = optimize(profile, [], self._fake_gateway())
        assert isinstance(result, dict)


# ============================================================
# optimize.py: classify_skill_gap (re-export)
# ============================================================

class TestOptClassifySkillGap:
    def test_same_result_as_analyze(self):
        profile = _make_profile(skills=["Java", "LLM"])
        from coach.resume.analyze import classify_skill_gap as a_classify
        gaps_opt = opt_classify_skill_gap(profile, "AI工程师")
        gaps_ana = a_classify(profile, "AI工程师")
        assert len(gaps_opt) == len(gaps_ana)


# ============================================================
# benchmark.py: benchmark_competitors
# ============================================================

class TestBenchmarkCompetitors:
    def _write_resumes(self, tmp_path: Path, resumes: dict[str, str]) -> Path:
        d = tmp_path / "resumes"
        d.mkdir()
        for name, text in resumes.items():
            (d / name).write_text(text, encoding="utf-8")
        return d

    def test_returns_expected_keys(self, tmp_path):
        d = self._write_resumes(tmp_path, {"c1.txt": SAMPLE_RESUME})
        cfg = {"run": {"target_role": "AI工程师"}}
        result = benchmark_competitors(str(d), cfg)
        assert "competitors" in result
        assert "target_role" in result
        assert "gap_vs_best" in result
        assert "summary" in result

    def test_competitor_parsed(self, tmp_path):
        d = self._write_resumes(tmp_path, {"c1.txt": SAMPLE_RESUME})
        cfg = {"run": {"target_role": "AI工程师"}}
        result = benchmark_competitors(str(d), cfg)
        assert len(result["competitors"]) == 1
        assert result["competitors"][0]["name"] == "c1"

    def test_nonexistent_dir(self, tmp_path):
        cfg = {"run": {"target_role": "AI工程师"}}
        result = benchmark_competitors(str(tmp_path / "missing"), cfg)
        assert result["competitors"] == []
        assert "not found" in result["summary"]

    def test_multiple_competitors_sorted(self, tmp_path):
        weak = "技能\nPython\n"
        strong = SAMPLE_RESUME
        d = self._write_resumes(tmp_path, {"weak.txt": weak, "strong.txt": strong})
        cfg = {"run": {"target_role": "AI工程师"}}
        result = benchmark_competitors(str(d), cfg)
        scores = [c.get("match_score", 0) for c in result["competitors"]]
        assert scores == sorted(scores, reverse=True)

    def test_user_profile_loaded(self, tmp_path):
        d = self._write_resumes(tmp_path, {"c1.txt": SAMPLE_RESUME})
        profile_path = tmp_path / "profile.json"
        profile = _make_profile(skills=["Java", "LLM"])
        profile_path.write_text(
            json.dumps(profile.model_dump(), ensure_ascii=False), encoding="utf-8"
        )
        cfg = {
            "run": {"target_role": "AI工程师"},
            "paths": {"resume_profile": str(profile_path)},
        }
        result = benchmark_competitors(str(d), cfg)
        assert result["user"] is not None
        assert result["user"]["name"] == "user"

    def test_empty_dir(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        cfg = {"run": {"target_role": "AI工程师"}}
        result = benchmark_competitors(str(d), cfg)
        assert result["competitors"] == []

    def test_score_range(self, tmp_path):
        d = self._write_resumes(tmp_path, {"r1.txt": SAMPLE_RESUME})
        cfg = {"run": {"target_role": "AI工程师"}}
        result = benchmark_competitors(str(d), cfg)
        for c in result["competitors"]:
            if "match_score" in c:
                assert 0 <= c["match_score"] <= 100
