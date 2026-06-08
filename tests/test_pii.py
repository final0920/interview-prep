"""Tests for coach.pii -- the single PII detection/redaction source of truth.

Covers the public API (redact / looks_like_pii), accurate count/coverage
reporting, and the trivial-bypass cases the old per-module regexes missed:
spaced / dashed digits, +86 / 0086 prefixes, full-width digits, spaced
email, and a spaced bank-card number.
"""
from __future__ import annotations

import pytest

from coach.pii import redact, looks_like_pii


# ============================================================
# Basic detection + reporting contract
# ============================================================

class TestRedactBasics:
    def test_phone_masked(self):
        masked, rep = redact("联系: 18888501310")
        assert "<PHONE>" in masked and "18888501310" not in masked
        assert rep["counts"]["phone"] == 1

    def test_email_masked(self):
        masked, rep = redact("email: test@example.com")
        assert "<EMAIL>" in masked
        assert rep["counts"]["email"] == 1

    def test_id_card_masked(self):
        masked, rep = redact("身份证: 110101199003071234")
        assert "<ID_CARD>" in masked
        assert rep["counts"]["id_card"] == 1

    def test_bank_card_masked(self):
        masked, rep = redact("银行卡: 6222021234567890123")
        assert "<BANK_CARD>" in masked
        assert rep["counts"]["bank_card"] == 1

    def test_id_before_bank(self):
        # an 18-digit ID must be tagged id_card, never bank_card
        _, rep = redact("110101199003071234")
        assert rep["counts"]["id_card"] == 1
        assert rep["counts"]["bank_card"] == 0

    def test_none_input(self):
        masked, rep = redact(None)
        assert masked == ""
        assert rep["total_pii"] == 0

    def test_returns_tuple(self):
        result = redact("hello")
        assert isinstance(result, tuple) and len(result) == 2


class TestReportAccuracy:
    def test_coverage_real_not_hardcoded_for_clean_text(self):
        # no PII: coverage defined as 1.0, but total/masked are 0 (not faked)
        _, rep = redact("the quick brown fox")
        assert rep["total_pii"] == 0
        assert rep["masked_pii"] == 0
        assert rep["coverage"] == 1.0
        assert rep["pii_masked"] is False

    def test_counts_reflect_actual_matches(self):
        text = "18888501310 foo@bar.com 110101199003071234 6222021234567890123"
        masked, rep = redact(text)
        assert rep["total_pii"] == 4
        assert rep["counts"] == {
            "phone": 1, "email": 1, "id_card": 1, "bank_card": 1,
        }
        # redact-on-detect => coverage is genuinely masked/total == 1.0
        assert rep["masked_pii"] == 4
        assert rep["coverage"] == 1.0
        assert rep["pii_masked"] is True
        assert "18888501310" not in masked

    def test_multiple_same_kind_counted(self):
        _, rep = redact("13800000000 and 13900000001")
        assert rep["counts"]["phone"] == 2
        assert rep["total_pii"] == 2

    def test_spans_audit(self):
        _, rep = redact("18888501310 foo@bar.com")
        assert len(rep["spans"]) == 2
        types = {s["type"] for s in rep["spans"]}
        assert types == {"phone", "email"}


# ============================================================
# Bypass cases (the whole point of consolidating into coach.pii)
# ============================================================

class TestBypassResistance:
    def test_spaced_phone(self):
        assert looks_like_pii("188 8850 1310")
        masked, rep = redact("call 188 8850 1310 now")
        assert rep["counts"]["phone"] == 1
        assert "188 8850 1310" not in masked

    def test_dashed_phone(self):
        _, rep = redact("188-8850-1310")
        assert rep["counts"]["phone"] == 1

    def test_plus86_phone(self):
        assert looks_like_pii("+8618888501310")
        assert redact("+86 188 8850 1310")[1]["counts"]["phone"] == 1

    def test_0086_phone(self):
        assert redact("008618888501310")[1]["counts"]["phone"] == 1

    def test_fullwidth_phone(self):
        # full-width digits should normalise (NFKC) to ASCII and match
        assert looks_like_pii("１８８８８５０１３１０")
        assert redact("１８８８８５０１３１０")[1]["counts"]["phone"] == 1

    def test_spaced_email(self):
        assert looks_like_pii("foo @ bar . com")
        masked, _ = redact("reach me: foo @ bar . com")
        assert "<EMAIL>" in masked

    def test_spaced_bank_card(self):
        masked, rep = redact("6222 0212 3456 7890 123")
        assert rep["counts"]["bank_card"] == 1
        assert "<BANK_CARD>" in masked

    def test_dashed_bank_card(self):
        assert redact("6222-0212-3456-7890-123")[1]["counts"]["bank_card"] == 1


# ============================================================
# No false positives on clean technical prose
# ============================================================

class TestNoFalsePositives:
    def test_bigo_not_pii(self):
        assert not looks_like_pii("The algorithm runs in O(n log n) time.")

    def test_short_numbers_not_bank(self):
        # QPS / latency style numbers are not 16-19 digit cards
        assert not looks_like_pii("scaled to 5000 QPS at 200ms p99")

    def test_year_not_phone(self):
        assert not looks_like_pii("released in 2024 with build 12345")
