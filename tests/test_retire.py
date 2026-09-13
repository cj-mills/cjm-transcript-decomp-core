"""Spine retirement (ruling a7617bd4): the three rules as pure plans, the fact op shape,
replay registration, and the compaction driver over a fake graph."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cjm_transcript_decomp_core import retire as R
from cjm_transcript_decomp_core.pipeline import decomp_replay_handlers

H_OLD = "sha256:0ld0ld0ld0ld0ld0ld"
H_MID = "sha256:m1dm1dm1dm1dm1dm1d"
H_NEW = "sha256:n3wn3wn3wn3wn3wn3w"


def _spines():
    return [
        {"skeleton_hash": None, "split_policy": None, "segments": 900, "created_at": 10.0},
        {"skeleton_hash": H_OLD, "split_policy": "sentence-split/capability", "segments": 1000, "created_at": 20.0},
        {"skeleton_hash": H_MID, "split_policy": "sentence-split/capability", "segments": 1010, "created_at": 30.0},
        {"skeleton_hash": H_NEW, "split_policy": "sentence-split/capability", "segments": 1020, "created_at": 40.0},
    ]


def test_annotate_and_default_prefers_declared_successor_over_creation_order():
    # Rule (c): a declared successor wins even when it is OLDER than the newest spine.
    retired = {R.spine_key(H_NEW): {"reason": "worse", "successor": R.spine_key(H_MID), "ts": 100.0}}
    rows = R.annotate_spines(_spines(), retired)
    for r in rows:
        r["retired_ts"] = retired.get(R.spine_key(r["skeleton_hash"]), {}).get("ts")
    assert [r["retired"] for r in rows] == [False, False, False, True]
    assert R.default_live_spine(rows)["skeleton_hash"] == H_MID
    # no declaration: newest live spine
    rows2 = R.annotate_spines(_spines(), {})
    assert R.default_live_spine(rows2)["skeleton_hash"] == H_NEW
    # everything retired: None
    all_r = {R.spine_key(s["skeleton_hash"]): {"reason": "x"} for s in _spines()}
    assert R.default_live_spine(R.annotate_spines(_spines(), all_r)) is None
    assert R.spine_label(rows[3]).endswith("· retired")


def test_plan_retire_any_age_and_older_successor():
    # Rule (a): the NEWEST spine may retire; the successor may be an older live spine.
    new_map, plan = R.plan_retire(_spines(), {}, "n3wn3w", reason="prefer previous",
                                  successor="m1dm1d", actor="t", ts=5.0)
    assert plan["act"] == "retire" and plan["key"] == H_NEW
    assert new_map[H_NEW] == {"reason": "prefer previous", "successor": H_MID, "actor": "t", "ts": 5.0}
    # legacy spine retires by its token
    m2, p2 = R.plan_retire(_spines(), new_map, "legacy", ts=6.0)
    assert p2["key"] == R.LEGACY_KEY and set(m2) == {H_NEW, R.LEGACY_KEY}


def test_plan_retire_refuses_last_live_spine_not_latest():
    retired = {R.spine_key(H_OLD): {"reason": "a"}, R.spine_key(H_MID): {"reason": "b"},
               R.spine_key(None): {"reason": "c"}}
    with pytest.raises(ValueError, match="NO live spine"):
        R.plan_retire(_spines(), retired, "n3w")
    # but with another live spine present the newest retires fine
    del retired[R.spine_key(H_MID)]
    m, p = R.plan_retire(_spines(), retired, "n3w", successor="m1d")
    assert p["entry"]["successor"] == H_MID


def test_plan_retire_validation():
    with pytest.raises(ValueError, match="already retired"):
        R.plan_retire(_spines(), {H_OLD: {"reason": "x"}}, "0ld")
    with pytest.raises(ValueError, match="matches 0"):
        R.plan_retire(_spines(), {}, "zzz")
    with pytest.raises(ValueError, match="matches 3"):
        R.plan_retire(_spines(), {}, "sha256:")
    # a retired spine cannot be named successor
    with pytest.raises(ValueError, match="matches 0"):
        R.plan_retire(_spines(), {H_MID: {"reason": "x"}}, "0ld", successor="m1d")
    # unretire
    m, p = R.plan_retire(_spines(), {H_OLD: {"reason": "x"}}, "0ld", unretire=True)
    assert p["act"] == "unretire" and m == {}
    with pytest.raises(ValueError, match="not retired"):
        R.plan_retire(_spines(), {}, "0ld", unretire=True)
    with pytest.raises(ValueError, match="COMPACTED"):
        R.plan_retire(_spines(), {H_OLD: {"reason": "x", "compacted": {"archive": "a.jsonl"}}}, "0ld", unretire=True)


def test_replay_registry_carries_the_fact_verbs():
    h = decomp_replay_handlers()
    assert h[R.RETIRE_VERB] is R.apply_spine_fact and h[R.COMPACTION_VERB] is R.apply_spine_fact
    assert "spine-extension" in h and "derivation" in h


class _FakeGraph:
    """A minimal graph_task stand-in: Source nodes with properties, Segments keyed by
    (rendition, skeleton), CORRECTS/REVIEWED edges, delete_nodes."""

    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.calls = []

    async def __call__(self, queue, graph_id, method, **kw):
        self.calls.append((method, kw))
        if method == "get_node":
            n = self.nodes.get(kw["node_id"])
            return None if n is None else dict(n)
        if method == "update_node":
            self.nodes[kw["node_id"]]["properties"].update(kw["properties"])
            return True
        if method == "delete_nodes":
            n = 0
            for i in kw["node_ids"]:
                if i in self.nodes:
                    del self.nodes[i]
                    n += 1
            self.edges = [e for e in self.edges if e["source_id"] in self.nodes and e["target_id"] in self.nodes]
            return n
        if method == "query_nodes":
            q = kw["query"]
            rows = []
            for i, n in self.nodes.items():
                if n["label"] != q["label"]:
                    continue
                p = n["properties"]
                rel = q.get("related")
                if rel:
                    far = set(rel.get("node_ids") or ([rel["node_id"]] if rel.get("node_id") else []))
                    if not any(e["source_id"] == i and e["target_id"] in far and e["relation_type"] == rel["relation_type"]
                               for e in self.edges):
                        continue
                ok = True
                for w in q.get("where") or []:
                    v = p.get(w["prop"])
                    if w["op"] == "eq" and v != w["value"]:
                        ok = False
                    if w["op"] == "is_null" and v is not None:
                        ok = False
                if not ok:
                    continue
                rows.append({"id": i, **{k: p.get(k) for k in (q.get("project") or [])}, "created_at": n.get("created_at")})
            off = q.get("offset") or 0
            lim = q.get("limit")
            rows = rows[off:off + lim] if lim else rows[off:]
            return SimpleNamespace(rows=rows)
        if method == "query_edges":
            q = kw["query"]
            tg = set(q.get("target_ids") or [])
            rows = [{"id": e["id"], "source_id": e["source_id"], "target_id": e["target_id"]}
                    for e in self.edges if e["relation_type"] == q["relation_type"] and e["target_id"] in tg]
            return SimpleNamespace(rows=rows)
        raise AssertionError(method)


def _seed(g: _FakeGraph):
    g.nodes["src"] = {"label": "Source", "properties": {"title": "Lecture 1"}}
    g.nodes["aseg"] = {"label": "AudioSegment", "properties": {}}
    g.nodes["rend"] = {"label": "AudioRendition", "properties": {}}
    g.edges += [{"id": "e1", "source_id": "aseg", "target_id": "src", "relation_type": "PART_OF"},
                {"id": "e2", "source_id": "rend", "target_id": "aseg", "relation_type": "DERIVED_FROM"}]
    for h, born in ((H_OLD, 20.0), (H_NEW, 40.0)):
        for k in range(3):
            sid = f"seg-{h[-3:]}-{k}"
            g.nodes[sid] = {"label": "Segment", "created_at": born,
                            "properties": {"skeleton_hash": h, "split_policy": "sentence-split/capability"}}
            g.edges.append({"id": f"p-{sid}", "source_id": sid, "target_id": "rend", "relation_type": "PART_OF"})
    return g


def test_retire_gate_and_fact_op(monkeypatch, tmp_path):
    g = _seed(_FakeGraph())
    monkeypatch.setattr(R, "graph_task", g)
    jp = str(tmp_path / "wf.writes.jsonl")
    # a correction on the OLD spine blocks its retirement (rule b), listing the counts
    g.nodes["c1"] = {"label": "Correction", "properties": {}}
    g.edges.append({"id": "ec", "source_id": "c1", "target_id": f"seg-{H_OLD[-3:]}-1", "relation_type": "CORRECTS"})
    with pytest.raises(ValueError, match="1 correction"):
        asyncio.run(R.retire_spine(g, "graph", "src", "0ld", journal_path=jp, actor="t"))
    assert not Path(jp).exists()
    # the NEW spine (no dependents) retires with the OLD one as successor — older successor is legal
    r = asyncio.run(R.retire_spine(g, "graph", "src", "n3w", reason="prefer old", successor="0ld",
                                   journal_path=jp, actor="t"))
    op = r["op"]
    assert op["verb"] == R.RETIRE_VERB and op["args"]["act"] == "retire" and op["args"]["successor"] == H_OLD
    assert g.nodes["src"]["properties"][R.RETIRED_SPINES_PROP][H_NEW]["reason"] == "prefer old"
    journaled = [json.loads(l) for l in Path(jp).read_text().splitlines()]
    assert journaled[-1]["verb"] == R.RETIRE_VERB and journaled[-1]["updates"][0]["id"] == "src"
    # listing reflects it; default is the declared successor
    spines = asyncio.run(R.list_spines(g, "graph", "src"))
    assert {s["skeleton_hash"]: s["retired"] for s in spines} == {H_OLD: False, H_NEW: True}
    assert R.default_live_spine(spines)["skeleton_hash"] == H_OLD
    # the last live spine refuses (rule a)
    with pytest.raises(ValueError, match="NO live spine"):
        asyncio.run(R.retire_spine(g, "graph", "src", "0ld", journal_path=jp, actor="t", force_dependents=True))
    # replaying the fact onto a fresh Source converges
    g.nodes["src"]["properties"].pop(R.RETIRED_SPINES_PROP)
    asyncio.run(R.apply_spine_fact(g, "graph", op))
    assert H_NEW in g.nodes["src"]["properties"][R.RETIRED_SPINES_PROP]


def test_plan_superseded_and_compact(monkeypatch, tmp_path):
    g = _seed(_FakeGraph())
    monkeypatch.setattr(R, "graph_task", g)
    jp = tmp_path / "wf.writes.jsonl"
    # the journal holds the two spine ops + a backfill op mixing both spines
    def seg_wire(h, k):
        return {"id": f"seg-{h[-3:]}-{k}", "label": "Segment", "properties": {"skeleton_hash": h}, "sources": []}
    ops = [
        {"verb": "spine-extension", "actor": "p", "run": "r-old", "ts": 20.0, "args": {"source_id": "src"},
         "wires": {"nodes": [seg_wire(H_OLD, k) for k in range(3)],
                   "edges": [{"id": f"p-seg-{H_OLD[-3:]}-{k}", "source_id": f"seg-{H_OLD[-3:]}-{k}",
                              "target_id": "rend", "relation_type": "PART_OF", "properties": {}} for k in range(3)]}},
        {"verb": "spine-extension", "actor": "p", "run": "r-new", "ts": 40.0, "args": {"source_id": "src"},
         "wires": {"nodes": [seg_wire(H_NEW, k) for k in range(3)], "edges": []}},
        {"verb": "spine-extension", "actor": "cli", "ts": 50.0, "args": {"act": "provenance-backfill"},
         "wires": {"nodes": [], "edges": [
             {"id": "d-old", "source_id": f"seg-{H_OLD[-3:]}-0", "target_id": "t1", "relation_type": "DERIVED_FROM", "properties": {}},
             {"id": "d-new", "source_id": f"seg-{H_NEW[-3:]}-0", "target_id": "t1", "relation_type": "DERIVED_FROM", "properties": {}}]}},
    ]
    jp.write_text("".join(json.dumps(o, sort_keys=True) + "\n" for o in ops))
    # plan: the OLD spine is superseded by the newest (no declaration) and dependent-free
    plan = asyncio.run(R.plan_superseded(g, "graph", ["src"]))
    assert len(plan) == 1 and plan[0]["spine"]["skeleton_hash"] == H_OLD and plan[0]["eligible"]
    assert plan[0]["successor"]["skeleton_hash"] == H_NEW
    asyncio.run(R.retire_spine(g, "graph", "src", "0ld", reason="superseded", successor="n3w",
                               journal_path=str(jp), actor="t"))
    # dry run moves nothing
    dry = asyncio.run(R.compact_retired(g, "graph", ["src"], journal_path=str(jp), archive_dir=str(tmp_path / "archive"),
                                        label="L", actor="t", dry_run=True))
    assert dry["report"].nodes_archived == 3 and dry["deleted"] == 0 and len(g.nodes) == 3 + 6 + 0
    # live: archive written, segments deleted from the db, fact journaled, no dangling reference
    res = asyncio.run(R.compact_retired(g, "graph", ["src"], journal_path=str(jp), archive_dir=str(tmp_path / "archive"),
                                        label="L", actor="t"))
    rep = res["report"]
    assert rep.nodes_archived == 3 and rep.edges_archived == 4 and rep.ops_split == 1 and rep.ops_archived_whole == 1
    assert res["deleted"] == 3 and res["dangling_after"] == 0
    assert not any(i.startswith(f"seg-{H_OLD[-3:]}") for i in g.nodes)
    entry = g.nodes["src"]["properties"][R.RETIRED_SPINES_PROP][H_OLD]
    assert entry["compacted"]["archive"] == rep.archive_path and entry["compacted"]["segments"] == 3
    from cjm_context_graph_primitives.journal import read_journal
    fam = read_journal(str(jp))
    assert fam[-1]["verb"] == R.COMPACTION_VERB
    assert not any(n["id"].startswith(f"seg-{H_OLD[-3:]}") for o in fam for n in (o.get("wires") or {}).get("nodes", []))
    # a second compaction finds nothing to do
    again = asyncio.run(R.compact_retired(g, "graph", ["src"], journal_path=str(jp), archive_dir=str(tmp_path / "archive"),
                                          label="L2", actor="t"))
    assert again["targets"] == [] and again["report"].ops_scanned == 0
