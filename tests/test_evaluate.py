"""Offline tests for coach.evaluate (sql_sandbox, claim_check, calibrate).

No network: the SQL sandbox is pure sqlite3 in-memory; NLI is a deterministic
heuristic; the judge runs against a FakeGateway whose .structured() validates a
canned dict into the requested schema (mirroring the real gateway contract).
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel

from coach.evaluate import calibrate, claim_check, sql_sandbox
from coach.evaluate.claim_check import (
    check_claims,
    grounding_rate,
    nli_score,
    route_verdict,
    split_claims,
)
from coach.evaluate.sql_sandbox import normalize_mysql_to_sqlite, verify_sql
from coach.schemas import (
    AnswerEvaluation,
    ClaimVerdict,
    EvidenceUnit,
    Question,
    QuestionType,
    RetrievalHit,
    Verdict,
)


# ===========================================================================
# sql_sandbox
# ===========================================================================

MYSQL_DDL = """
CREATE TABLE `orders` (
  `id` bigint(20) unsigned NOT NULL AUTO_INCREMENT COMMENT '主键',
  `user_id` int(11) NOT NULL,
  `amount` decimal(10,2) DEFAULT NULL,
  `status` tinyint(1) DEFAULT '0',
  `created_at` datetime(0) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `idx_user` (`user_id`) USING BTREE
) ENGINE=InnoDB AUTO_INCREMENT=100 DEFAULT CHARSET=utf8mb4 COMMENT='订单表';
"""


def test_normalize_strips_mysql_specifics():
    ddl = normalize_mysql_to_sqlite(MYSQL_DDL)
    assert "`" not in ddl
    assert "ENGINE" not in ddl.upper()
    assert "AUTO_INCREMENT" not in ddl.upper()
    assert "COMMENT" not in ddl.upper()
    assert "USING BTREE" not in ddl.upper()
    # inline KEY index line dropped
    assert "idx_user" not in ddl
    # type narrowing
    assert "bigint" not in ddl.lower()
    assert "INTEGER" in ddl.upper()
    assert "NUMERIC" in ddl.upper()


def test_normalized_ddl_builds_in_sandbox():
    ddl = normalize_mysql_to_sqlite(MYSQL_DDL)
    ok, err = verify_sql(ddl, "SELECT user_id, SUM(amount) FROM orders GROUP BY user_id")
    assert ok, err


def test_verify_sql_select_via_explain_no_data():
    ddl = "CREATE TABLE t (a INTEGER, b TEXT)"
    ok, err = verify_sql(ddl, "SELECT a, b FROM t WHERE a > 1")
    assert ok and err == ""


def test_verify_sql_bad_column_fails():
    ddl = "CREATE TABLE t (a INTEGER)"
    ok, err = verify_sql(ddl, "SELECT nonexistent FROM t")
    assert not ok
    assert "nonexistent" in err or "no such column" in err.lower()


def test_verify_sql_bad_table_fails():
    ok, err = verify_sql("CREATE TABLE t (a INTEGER)", "SELECT * FROM missing_table")
    assert not ok


def test_verify_sql_multi_statement_insert_then_select():
    ddl = "CREATE TABLE t (a INTEGER, b INTEGER)"
    ok, err = verify_sql(ddl, "INSERT INTO t (a,b) VALUES (1,2); SELECT a FROM t;")
    assert ok, err


def test_verify_sql_empty_query():
    ok, err = verify_sql("CREATE TABLE t (a INTEGER)", "   ")
    assert not ok
    assert "no SQL" in err


def test_verify_sql_setup_error_reported():
    ok, err = verify_sql("CREATE TABLE (bad", "SELECT 1")
    assert not ok
    assert err.startswith("setup ")


def test_verify_sql_with_cte():
    ddl = "CREATE TABLE t (a INTEGER)"
    ok, err = verify_sql(ddl, "WITH c AS (SELECT a FROM t) SELECT * FROM c")
    assert ok, err


# ===========================================================================
# claim_check
# ===========================================================================

def test_split_claims_basic():
    text = "我熟悉 Kafka 消息队列的高可用设计。基于 Redis 实现了分布式锁；优化了慢查询。"
    claims = split_claims(text)
    assert len(claims) == 3
    assert any("Kafka" in c for c in claims)


def test_split_claims_drops_short_and_dedupes():
    text = "好。\n好。\n我使用了 Spring Boot 框架搭建后端服务。\n我使用了 Spring Boot 框架搭建后端服务。"
    claims = split_claims(text)
    assert len(claims) == 1
    assert "Spring" in claims[0]


def test_split_claims_strips_bullets():
    text = "- 使用 Docker 容器化部署应用\n* 基于 Kubernetes 编排调度"
    claims = split_claims(text)
    assert len(claims) == 2
    assert not claims[0].startswith("-")


def test_nli_entail_on_overlap():
    s = nli_score("使用 Kafka 实现异步消息解耦", "KafkaProducer 异步消息 解耦 send")
    assert s["label"] == "entail"
    assert s["overlap"] > 0.4


def test_nli_neutral_on_no_overlap():
    s = nli_score("精通 Rust 内存安全", "completely unrelated python helper function")
    assert s["label"] == "neutral"


def test_nli_contradict_on_polarity_conflict():
    # claim is negative, evidence positive, with shared keywords -> contradict
    s = nli_score("缓存 未 使用 Redis 集群", "缓存 使用 Redis 集群 部署")
    assert s["label"] == "contradict"


def test_nli_metric_needs_number():
    # quantitative claim, evidence has the words but no number -> not entailed
    s = nli_score("QPS 提升 三倍 吞吐", "提升 吞吐 优化 性能 改造")
    assert s["label"] != "entail"


def test_route_verdict_mapping():
    assert route_verdict("entail") == ClaimVerdict.verified
    assert route_verdict("contradict") == ClaimVerdict.rejected
    assert route_verdict("neutral") == ClaimVerdict.needs_evidence
    assert route_verdict("anything") == ClaimVerdict.needs_evidence


def _ev(eid: str, symbol: str, text: str) -> EvidenceUnit:
    return EvidenceUnit(id=eid, source_path="a.py", symbol=symbol, start_line=1, text=text)


def test_check_claims_verified_and_needs_evidence():
    evidence = [
        _ev("e1", "function_definition:send_kafka", "KafkaProducer 异步 消息 send 解耦 队列"),
        _ev("e2", "function_definition:noise", "totally unrelated helper"),
    ]
    claims = ["使用 Kafka 异步 消息 解耦", "精通 量子 计算 编程"]
    checks = check_claims(claims, evidence)
    assert len(checks) == 2
    by_claim = {c.claim: c for c in checks}
    assert by_claim["使用 Kafka 异步 消息 解耦"].verdict == ClaimVerdict.verified
    assert "e1" in by_claim["使用 Kafka 异步 消息 解耦"].evidence_ids
    assert by_claim["精通 量子 计算 编程"].verdict == ClaimVerdict.needs_evidence
    assert by_claim["精通 量子 计算 编程"].evidence_ids == []


def test_grounding_rate():
    evidence = [_ev("e1", "function_definition:f", "Kafka 异步 消息 解耦 队列 send")]
    checks = check_claims(["使用 Kafka 异步 消息 解耦", "毫无 关联 的 主张 内容"], evidence)
    gr = grounding_rate(checks)
    assert gr == pytest.approx(0.5)
    assert grounding_rate([]) == 0.0


def test_check_claims_with_llm_gateway_overrides_label():
    # gateway forces 'contradict' regardless of offline heuristic
    class FakeGW:
        def complete(self, messages, **kw):
            return '{"label": "contradict", "score": 0.9, "reason": "x"}'

    evidence = [_ev("e1", "function_definition:f", "Kafka 异步 消息 解耦 send 队列")]
    checks = check_claims(["使用 Kafka 异步 消息 解耦"], evidence, gateway=FakeGW())
    assert checks[0].verdict == ClaimVerdict.rejected


def test_check_claims_llm_failure_falls_back_offline():
    class BoomGW:
        def complete(self, messages, **kw):
            raise RuntimeError("network down")

    evidence = [_ev("e1", "function_definition:f", "Kafka 异步 消息 解耦 send 队列")]
    checks = check_claims(["使用 Kafka 异步 消息 解耦"], evidence, gateway=BoomGW())
    # offline heuristic still yields verified
    assert checks[0].verdict == ClaimVerdict.verified


# ===========================================================================
# calibrate.judge
# ===========================================================================

def _question() -> Question:
    return Question(
        id="q1",
        type=QuestionType.project_deep_dive,
        prompt="讲讲你的消息队列高可用设计。",
        key_points=["副本机制", "幂等消费", "限流降级"],
    )


def _hits() -> list[RetrievalHit]:
    return [
        RetrievalHit(evidence=_ev("e1", "function_definition:consume", "Kafka 幂等 消费 offset 提交")),
        RetrievalHit(evidence=_ev("e2", "class_definition:RateLimiter", "限流 降级 熔断 令牌桶")),
    ]


class FakeGateway:
    """Mimics LLMGateway.structured: validate a canned dict into the schema."""

    def __init__(self, payload: dict, *, fail: bool = False):
        self.payload = payload
        self.fail = fail
        self.calls = 0

    def structured(self, messages, schema: type[BaseModel], **kw):
        self.calls += 1
        if self.fail:
            raise RuntimeError("gateway down")
        return schema.model_validate(self.payload)


def test_judge_maps_structured_output():
    gw = FakeGateway({
        "score": 82,
        "verdict": "pass",
        "key_points_hit": ["副本机制", "幂等消费"],
        "issues": ["未提限流"],
        "fabrication_flags": [],
        "followup": "如何保证 exactly-once?",
    })
    ev = calibrate.judge(_question(), "我们用 Kafka 多副本 + 幂等消费保证不丢不重。", _hits(), gw)
    assert isinstance(ev, AnswerEvaluation)
    assert ev.question_id == "q1"
    assert ev.score == 82
    assert ev.verdict == Verdict.passed
    assert "幂等消费" in ev.key_points_hit
    assert ev.followup.startswith("如何")
    assert 0.0 <= ev.grounding_rate <= 1.0
    assert gw.calls == 1


def test_judge_pass_downgraded_when_low_score():
    gw = FakeGateway({"score": 40, "verdict": "pass", "key_points_hit": [],
                      "issues": [], "fabrication_flags": [], "followup": ""})
    ev = calibrate.judge(_question(), "随便答的。", _hits(), gw)
    assert ev.verdict == Verdict.needs_fix  # score < 60 cannot be a pass


def test_judge_pass_downgraded_when_fabrication():
    gw = FakeGateway({"score": 90, "verdict": "pass", "key_points_hit": [],
                      "issues": [], "fabrication_flags": ["宣称 QPS 百万但无证据"], "followup": ""})
    ev = calibrate.judge(_question(), "我做到百万 QPS。", _hits(), gw)
    assert ev.verdict == Verdict.needs_fix
    assert ev.fabrication_flags


def test_judge_score_clamped():
    gw = FakeGateway({"score": 250, "verdict": "pass", "key_points_hit": [],
                      "issues": [], "fabrication_flags": [], "followup": ""})
    ev = calibrate.judge(_question(), "答案。", _hits(), gw)
    assert ev.score == 100


def test_judge_degrades_on_gateway_failure():
    gw = FakeGateway({}, fail=True)
    ev = calibrate.judge(_question(), "我用 Kafka 幂等 消费。", _hits(), gw)
    assert ev.verdict == Verdict.needs_fix
    assert ev.score == 0
    assert ev.issues  # carries a diagnostic
    assert 0.0 <= ev.grounding_rate <= 1.0  # offline grounding still computed


def test_judge_empty_answer_zero_grounding():
    gw = FakeGateway({"score": 0, "verdict": "needs_fix", "key_points_hit": [],
                      "issues": ["空回答"], "fabrication_flags": [], "followup": "请展开"})
    ev = calibrate.judge(_question(), "", _hits(), gw)
    assert ev.grounding_rate == 0.0
