"""Offline tests for coach/memory/{store,placeholders,fold,tidal}.

All tests use :memory: sqlite -- no disk, no network, no LLM calls.
"""
from __future__ import annotations

import time

import pytest

from coach.schemas import MemoryEpisode, MemorySemantic, ResumeProfile, ResumeProject
from coach.memory.store import (
    MemoryStore,
    score_recency,
    tag_overlap,
)
from coach.memory.placeholders import render
from coach.memory.fold import fold_context
from coach.memory.tidal import tidal_recall


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    s = MemoryStore(":memory:")
    yield s
    s.close()


def _ep(content: str, tags: list[str], ts: float, kind: str = "event", score: int | None = None) -> MemoryEpisode:
    return MemoryEpisode(content=content, tags=tags, ts=ts, kind=kind, score=score)


def _sem(key: str, value: str, kind: str = "skill", confidence: float = 0.8) -> MemorySemantic:
    return MemorySemantic(key=key, value=value, kind=kind, confidence=confidence,
                          valid_from=0.0, updated_ts=0.0)


# ---------------------------------------------------------------------------
# score_recency pure function
# ---------------------------------------------------------------------------

class TestScoreRecency:
    def test_age_zero_is_one(self):
        assert score_recency(0.0) == pytest.approx(1.0)

    def test_age_half_life_is_half(self):
        assert score_recency(7.0, half_life_days=7.0) == pytest.approx(0.5, rel=1e-6)

    def test_age_double_half_life_is_quarter(self):
        assert score_recency(14.0, half_life_days=7.0) == pytest.approx(0.25, rel=1e-6)

    def test_negative_age_clamped_to_one(self):
        assert score_recency(-5.0) == pytest.approx(1.0)

    def test_zero_half_life_degenerate(self):
        assert score_recency(0.0, half_life_days=0.0) == pytest.approx(1.0)
        assert score_recency(1.0, half_life_days=0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# tag_overlap pure function
# ---------------------------------------------------------------------------

class TestTagOverlap:
    def test_identical_tags(self):
        assert tag_overlap(["a", "b"], ["a", "b"]) == pytest.approx(1.0)

    def test_disjoint_tags(self):
        assert tag_overlap(["a"], ["b"]) == pytest.approx(0.0)

    def test_partial_overlap(self):
        # |{a,b} & {b,c}| / |{a,b,c}| = 1/3
        assert tag_overlap(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)

    def test_empty_a_returns_zero(self):
        assert tag_overlap([], ["x"]) == pytest.approx(0.0)

    def test_empty_b_returns_zero(self):
        assert tag_overlap(["x"], []) == pytest.approx(0.0)

    def test_both_empty(self):
        assert tag_overlap([], []) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# MemoryStore.add_episode
# ---------------------------------------------------------------------------

class TestAddEpisode:
    def test_add_returns_string_id(self, store):
        ep = _ep("hello", ["python"], ts=1_000_000.0)
        eid = store.add_episode(ep)
        assert isinstance(eid, str) and len(eid) > 0

    def test_add_with_explicit_id(self, store):
        ep = MemoryEpisode(id="my-ep", content="test", tags=[], ts=1_000_000.0)
        eid = store.add_episode(ep)
        assert eid == "my-ep"

    def test_idempotent_on_same_id(self, store):
        ep = MemoryEpisode(id="ep-x", content="v1", tags=[], ts=1_000_000.0)
        store.add_episode(ep)
        ep2 = MemoryEpisode(id="ep-x", content="v2", tags=[], ts=1_000_000.0)
        store.add_episode(ep2)
        # Should still resolve to same id; content updated
        eps = store.recent_episodes(10)
        assert len(eps) == 1
        assert eps[0].content == "v2"

    def test_multiple_episodes_stored(self, store):
        now = time.time()
        store.add_episode(_ep("ep1", [], ts=now - 100))
        store.add_episode(_ep("ep2", [], ts=now - 50))
        store.add_episode(_ep("ep3", [], ts=now))
        assert len(store.recent_episodes(10)) == 3


# ---------------------------------------------------------------------------
# MemoryStore.recent_episodes -- decay ordering
# ---------------------------------------------------------------------------

class TestRecentEpisodes:
    def test_newer_episode_ranked_first_without_tags(self, store):
        now = 1_700_000_000.0
        store.add_episode(_ep("old", [], ts=now - 86400 * 10))  # 10 days ago
        store.add_episode(_ep("new", [], ts=now - 3600))        # 1 hour ago
        eps = store.recent_episodes(k=10, now=now)
        assert eps[0].content == "new"

    def test_tag_overlap_boosts_relevance(self, store):
        now = 1_700_000_000.0
        # Both same age, but one has matching tag
        store.add_episode(_ep("no-match", ["foo"], ts=now - 3600))
        store.add_episode(_ep("match", ["python", "ml"], ts=now - 3600))
        eps = store.recent_episodes(k=10, tags=["python"], now=now)
        assert eps[0].content == "match"

    def test_k_limits_results(self, store):
        now = time.time()
        for i in range(10):
            store.add_episode(_ep(f"ep{i}", [], ts=now - i * 60))
        assert len(store.recent_episodes(k=3, now=now)) == 3

    def test_returns_memoryepisode_objects(self, store):
        now = time.time()
        store.add_episode(_ep("hello", ["tag1"], ts=now))
        eps = store.recent_episodes(k=1, now=now)
        assert isinstance(eps[0], MemoryEpisode)
        assert eps[0].tags == ["tag1"]


# ---------------------------------------------------------------------------
# MemoryStore.upsert_semantic -- self-edit dedup / temporal window
# ---------------------------------------------------------------------------

class TestUpsertSemantic:
    def test_first_write_creates_record(self, store):
        store.upsert_semantic(_sem("skill:python", "intermediate"))
        results = store.get_semantic()
        assert len(results) == 1
        assert results[0].key == "skill:python"
        assert results[0].value == "intermediate"
        assert results[0].valid_to is None   # currently valid

    def test_same_value_noop_no_new_version(self, store):
        store.upsert_semantic(_sem("skill:python", "intermediate"))
        store.upsert_semantic(_sem("skill:python", "intermediate"))
        results = store.get_semantic()
        # Still only one current record
        assert len(results) == 1

    def test_confidence_monotonically_increases_on_noop(self, store):
        store.upsert_semantic(_sem("skill:python", "intermediate", confidence=0.5))
        store.upsert_semantic(_sem("skill:python", "intermediate", confidence=0.9))
        results = store.get_semantic()
        assert results[0].confidence == pytest.approx(0.9)

    def test_low_conf_noop_does_not_lower_confidence(self, store):
        store.upsert_semantic(_sem("skill:python", "intermediate", confidence=0.9))
        store.upsert_semantic(_sem("skill:python", "intermediate", confidence=0.2))
        results = store.get_semantic()
        assert results[0].confidence == pytest.approx(0.9)

    def test_changed_value_closes_old_record(self, store):
        """self-edit dedup: changed value creates new version, old gets valid_to set."""
        store.upsert_semantic(_sem("skill:python", "beginner"))
        store.upsert_semantic(_sem("skill:python", "advanced"))
        current = store.get_semantic()
        assert len(current) == 1
        assert current[0].value == "advanced"
        assert current[0].valid_to is None

    def test_old_record_has_valid_to_set_after_supersede(self, store):
        store.upsert_semantic(_sem("key:x", "v1"))
        store.upsert_semantic(_sem("key:x", "v2"))
        # Query raw sqlite to check historical record
        conn = store._conn
        rows = conn.execute(
            "SELECT * FROM memory_semantic WHERE key='key:x' ORDER BY valid_from"
        ).fetchall()
        assert len(rows) == 2
        assert rows[0]["valid_to"] is not None   # old version closed
        assert rows[1]["valid_to"] is None        # new version open

    def test_multiple_keys_independent(self, store):
        store.upsert_semantic(_sem("a", "va"))
        store.upsert_semantic(_sem("b", "vb"))
        results = store.get_semantic()
        keys = {m.key for m in results}
        assert keys == {"a", "b"}

    def test_get_semantic_filter_by_kind(self, store):
        store.upsert_semantic(_sem("wp:kafka", "Kafka reliability", kind="weakpoint"))
        store.upsert_semantic(_sem("sk:python", "Python", kind="skill"))
        wps = store.get_semantic(kind="weakpoint")
        assert len(wps) == 1
        assert wps[0].key == "wp:kafka"


# ---------------------------------------------------------------------------
# MemoryStore.weakpoints
# ---------------------------------------------------------------------------

class TestWeakpoints:
    def test_returns_weakpoint_values(self, store):
        store.upsert_semantic(_sem("wp:kafka", "Kafka HA", kind="weakpoint"))
        store.upsert_semantic(_sem("wp:redis", "Redis eviction", kind="weakpoint"))
        store.upsert_semantic(_sem("sk:python", "Python", kind="skill"))
        wps = store.weakpoints()
        assert set(wps) == {"Kafka HA", "Redis eviction"}

    def test_empty_when_no_weakpoints(self, store):
        store.upsert_semantic(_sem("sk:python", "Python", kind="skill"))
        assert store.weakpoints() == []

    def test_superseded_weakpoint_not_in_list(self, store):
        store.upsert_semantic(_sem("wp:topic", "old weakness", kind="weakpoint"))
        store.upsert_semantic(_sem("wp:topic", "updated weakness", kind="weakpoint"))
        wps = store.weakpoints()
        assert len(wps) == 1
        assert wps[0] == "updated weakness"


# ---------------------------------------------------------------------------
# placeholders.render
# ---------------------------------------------------------------------------

class TestRender:
    def test_render_weakpoints(self, store):
        store.upsert_semantic(_sem("wp:k", "Kafka HA", kind="weakpoint"))
        result = render("Focus on: {{weakpoints}}", store)
        assert "Kafka HA" in result

    def test_render_profile(self, store):
        store.upsert_semantic(_sem("skill:python", "advanced", kind="skill"))
        result = render("Profile: {{profile}}", store)
        assert "skill:python" in result
        assert "advanced" in result

    def test_render_resume_with_profile(self, store):
        rp = ResumeProfile(
            basics={"name": "Alice"},
            skills=["Python", "SQL"],
            projects=[ResumeProject(name="MyProject")],
        )
        result = render("Resume: {{resume}}", store, profile=rp)
        assert "Alice" in result
        assert "Python" in result
        assert "MyProject" in result

    def test_unknown_token_left_unchanged(self, store):
        result = render("{{unknown_token}}", store)
        assert result == "{{unknown_token}}"

    def test_no_weakpoints_placeholder(self, store):
        result = render("{{weakpoints}}", store)
        assert "no weakpoints" in result.lower()

    def test_no_resume_placeholder(self, store):
        result = render("{{resume}}", store, profile=None)
        assert "no resume" in result.lower()

    def test_multiple_tokens_in_one_template(self, store):
        store.upsert_semantic(_sem("wp:x", "weakness X", kind="weakpoint"))
        result = render("WP: {{weakpoints}}\nPROFILE: {{profile}}", store)
        assert "weakness X" in result
        assert "wp:x" in result


# ---------------------------------------------------------------------------
# fold.fold_context
# ---------------------------------------------------------------------------

class TestFoldContext:
    def _make_gateway(self, response: str = "SUMMARY"):
        """Fake gateway whose cheap_complete returns a canned string."""
        class FakeGateway:
            def cheap_complete(self, messages):
                return response
        return FakeGateway()

    def test_empty_chunks_returns_empty(self):
        gw = self._make_gateway()
        assert fold_context([], gw, keep_recent=3) == ""

    def test_fewer_than_keep_recent_no_llm_call(self):
        """When total chunks <= keep_recent, no summarisation needed."""
        called = []
        class FakeGateway:
            def cheap_complete(self, messages):
                called.append(True)
                return "SUMMARY"
        result = fold_context(["a", "b"], FakeGateway(), keep_recent=3)
        assert called == []
        assert "a" in result and "b" in result

    def test_distant_chunks_summarised(self):
        gw = self._make_gateway("COMPRESSED")
        chunks = ["old1", "old2", "old3", "recent1", "recent2"]
        result = fold_context(chunks, gw, keep_recent=2)
        assert "COMPRESSED" in result
        assert "recent1" in result
        assert "recent2" in result
        # Old chunks should NOT appear verbatim (they were summarised)
        assert "old1" not in result

    def test_gateway_error_falls_back_to_plain_join(self):
        class BrokenGateway:
            def cheap_complete(self, messages):
                raise RuntimeError("network error")
        chunks = ["a", "b", "c"]
        result = fold_context(chunks, BrokenGateway(), keep_recent=1)
        # Should not raise; should return all content
        assert "a" in result and "b" in result and "c" in result

    def test_keep_recent_zero_summarises_all(self):
        gw = self._make_gateway("ALL_SUMMARISED")
        chunks = ["x", "y"]
        result = fold_context(chunks, gw, keep_recent=0)
        assert "ALL_SUMMARISED" in result


# ---------------------------------------------------------------------------
# tidal.tidal_recall -- bucketing on toy timestamps
# ---------------------------------------------------------------------------

class TestTidalRecall:
    def _populate(self, store, now: float):
        """Add episodes in near/mid/abyss zones."""
        # near: 1 day ago
        ep_near = _ep("near-event", ["python", "ml"], ts=now - 86400 * 1)
        # mid: 30 days ago
        ep_mid = _ep("mid-event", ["kafka"], ts=now - 86400 * 30)
        # abyss: 120 days ago, with matching tag
        ep_abyss = _ep("abyss-event", ["python"], ts=now - 86400 * 120)
        # abyss: 200 days ago, no matching tag (should be filtered out)
        ep_abyss_filtered = _ep("abyss-filtered", ["unrelated"], ts=now - 86400 * 200)
        for ep in [ep_near, ep_mid, ep_abyss, ep_abyss_filtered]:
            store.add_episode(ep)

    def test_near_zone_always_included(self, store):
        now = 1_700_000_000.0
        self._populate(store, now)
        results = tidal_recall(store, None, seed_tags=["python"], now=now)
        contents = [ep.content for ep in results]
        assert "near-event" in contents

    def test_mid_zone_included(self, store):
        now = 1_700_000_000.0
        self._populate(store, now)
        results = tidal_recall(store, None, seed_tags=["kafka"], now=now)
        contents = [ep.content for ep in results]
        assert "mid-event" in contents

    def test_abyss_with_resonance_included(self, store):
        now = 1_700_000_000.0
        self._populate(store, now)
        results = tidal_recall(store, None, seed_tags=["python"], now=now)
        contents = [ep.content for ep in results]
        assert "abyss-event" in contents

    def test_abyss_without_resonance_excluded(self, store):
        now = 1_700_000_000.0
        self._populate(store, now)
        results = tidal_recall(store, None, seed_tags=["python"], now=now)
        contents = [ep.content for ep in results]
        assert "abyss-filtered" not in contents

    def test_returns_list_of_memoryepisode(self, store):
        now = 1_700_000_000.0
        self._populate(store, now)
        results = tidal_recall(store, None, seed_tags=["python"], now=now)
        assert all(isinstance(ep, MemoryEpisode) for ep in results)

    def test_empty_store_returns_empty(self, store):
        now = 1_700_000_000.0
        results = tidal_recall(store, None, seed_tags=["python"], now=now)
        assert results == []

    def test_near_ranked_above_abyss(self, store):
        now = 1_700_000_000.0
        # Both have same tags but different ages
        store.add_episode(_ep("near", ["python"], ts=now - 86400 * 1))
        store.add_episode(_ep("abyss", ["python"], ts=now - 86400 * 150))
        results = tidal_recall(store, None, seed_tags=["python"], now=now)
        contents = [ep.content for ep in results]
        # near should appear before abyss
        near_idx = contents.index("near") if "near" in contents else 999
        abyss_idx = contents.index("abyss") if "abyss" in contents else 999
        assert near_idx < abyss_idx

    def test_no_seed_tags_returns_results(self, store):
        now = 1_700_000_000.0
        store.add_episode(_ep("ep1", ["a"], ts=now - 86400))
        results = tidal_recall(store, None, seed_tags=[], now=now)
        assert len(results) >= 1

    def test_vector_store_none_does_not_crash(self, store):
        now = 1_700_000_000.0
        store.add_episode(_ep("x", ["t"], ts=now - 86400))
        # Should not raise even with vector_store=None
        results = tidal_recall(store, None, seed_tags=["t"], now=now)
        assert len(results) >= 1
