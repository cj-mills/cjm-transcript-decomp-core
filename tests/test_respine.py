"""Chunk respine (work item 7a5e9c84; ruling 0b4d5cfa, amendment 4a7ec4f8): the pure
plans (manifest lineage, chunk resolution, renumber, dependents), the salted extension
payload, the fact op + its replay registration, the pseg-scoped composition, and the
parser surface — over a fake graph where a read is needed."""

import asyncio
from types import SimpleNamespace

import pytest
from cjm_transcript_graph_schema.schema import (audio_rendition_node_id, audio_segment_node_id,
                                                RESPINE_OP_VERB, RespinedChunkEntry,
                                                SEGMENT_SUPERSEDED_BY_PROP, segment_node_id,
                                                source_node_id, SOURCE_RESPINED_CHUNKS_PROP)

from cjm_transcript_decomp_core import respine as RS
from cjm_transcript_decomp_core.cli import build_parser
from cjm_transcript_decomp_core.graph import build_extension_payload
from cjm_transcript_decomp_core.models import DecompConfig, DecompManifest, DecompSegment, SegmentVariant
from cjm_transcript_decomp_core.pipeline import build_alignment_composition, decomp_replay_handlers
from cjm_transcript_decomp_core.retire import apply_spine_fact

H = "sha256:l1vel1vel1vel1vel1ve"
SRC_HASH = "sha256:src"
SID = source_node_id(SRC_HASH)
SOURCE_ENTRY = {
    "source_path": "/media/lecture.mp3", "content_hash": SRC_HASH, "chain": [],
    "segments": [
        {"index": 0, "start": 0.0, "end": 300.0, "model_input_hash": "sha256:w0", "model_input_path": "/w0.wav",
         "transcripts": {"voxtral": {"text": "alpha beta."}}},
        {"index": 1, "start": 300.0, "end": 600.0, "model_input_hash": "sha256:w1", "model_input_path": "/w1.wav",
         "transcripts": {"voxtral": {"text": "gamma delta."}}},
        {"index": 2, "start": 600.0, "end": 900.0, "model_input_hash": "sha256:w2", "model_input_path": "/w2.wav",
         "transcripts": {"voxtral": {"text": "epsilon."}}},
    ]}
CAPS = {"voxtral": {"config_hash": "sha256:cv"}}


def test_find_decomp_manifest_prefers_per_source_hash_newest_first():
    older = {"run_id": "d1", "skeleton_config_hash": H,
             "sources": [{"source_node_id": SID, "skeleton_config_hash": ""}]}
    newer = {"run_id": "d2", "skeleton_config_hash": "",
             "sources": [{"source_node_id": SID, "skeleton_config_hash": H, "title": "lecture"}]}
    other = {"run_id": "d3", "skeleton_config_hash": "sha256:other",
             "sources": [{"source_node_id": SID, "skeleton_config_hash": "sha256:other"}]}
    m, rec = RS.find_decomp_manifest([other, newer, older], SID, H)
    assert m["run_id"] == "d2" and rec["title"] == "lecture"
    # the run-level hash is the pre-0.2.6 fallback
    m2, _ = RS.find_decomp_manifest([older], SID, H)
    assert m2["run_id"] == "d1"
    with pytest.raises(ValueError, match="no decomp manifest records"):
        RS.find_decomp_manifest([other], SID, H)


def test_source_and_chunk_resolution_by_index_and_time():
    tm = {"run_id": "run-1", "sources": [{"content_hash": "sha256:zzz", "segments": []}, SOURCE_ENTRY]}
    si, entry = RS.source_entry_for(tm, SID)
    assert si == 1 and entry is SOURCE_ENTRY
    with pytest.raises(ValueError, match="not in the transcription manifest"):
        RS.source_entry_for(tm, "nope")
    pos, seg = RS.chunk_entry_for(SOURCE_ENTRY, chunk=1)
    assert pos == 1 and seg["start"] == 300.0
    pos2, seg2 = RS.chunk_entry_for(SOURCE_ENTRY, at_time=650.5)
    assert pos2 == 2 and seg2["index"] == 2
    assert RS.chunk_entry_for(SOURCE_ENTRY, at_time=0.0)[0] == 0
    with pytest.raises(ValueError, match="exactly one"):
        RS.chunk_entry_for(SOURCE_ENTRY)
    with pytest.raises(ValueError, match="exactly one"):
        RS.chunk_entry_for(SOURCE_ENTRY, chunk=1, at_time=5.0)
    with pytest.raises(ValueError, match="not in the source entry"):
        RS.chunk_entry_for(SOURCE_ENTRY, chunk=9)
    with pytest.raises(ValueError, match="after the last chunk"):
        RS.chunk_entry_for(SOURCE_ENTRY, at_time=901.0)


def test_plan_renumber_shifts_the_tail_and_guards_contiguity():
    tail = [("t1", 13), ("t2", 14), ("t3", 15)]
    first, delta, updates = RS.plan_renumber([10, 11, 12], 5, tail)
    assert (first, delta) == (10, 2) and updates == [("t1", 15), ("t2", 16), ("t3", 17)]
    # fewer segments than before shifts the tail DOWN; equal count = no updates
    assert RS.plan_renumber([10, 11, 12], 1, tail)[1:] == (-2, [("t1", 11), ("t2", 12), ("t3", 13)])
    assert RS.plan_renumber([10, 11, 12], 3, tail) == (10, 0, [])
    with pytest.raises(ValueError, match="not contiguous"):
        RS.plan_renumber([10, 12], 2, tail)
    with pytest.raises(ValueError, match="no live segments"):
        RS.plan_renumber([], 2, tail)
    with pytest.raises(ValueError, match="no segments"):
        RS.plan_renumber([10], 0, tail)
    with pytest.raises(ValueError, match="overlap"):
        RS.plan_renumber([10, 11], 2, [("x", 11)])


def test_classify_dependents_two_transferable_classes_rest_strands():
    old = {"s1", "s2"}
    rows = [
        {"id": "ev", "correction_type": "insertion", "status": "applied",
         "payload": {"operation": "chunk_insert", "after_segment_id": "s1", "label": "inhale", "text": ""}},
        {"id": "ev-proposed", "correction_type": "insertion", "status": "proposed",
         "payload": {"operation": "chunk_insert", "after_segment_id": "s1", "label": "inhale", "text": ""}},
        {"id": "ev-words", "correction_type": "insertion", "status": "applied",
         "payload": {"operation": "chunk_insert", "before_segment_id": "s2", "label": "empty", "text": "missed words"}},
        {"id": "ev-elsewhere", "correction_type": "insertion", "status": "applied",
         "payload": {"operation": "chunk_insert", "after_segment_id": "far", "label": "inhale", "text": ""}},
        {"id": "split", "correction_type": "insertion", "status": "applied", "rationale": "chunk-split",
         "payload": {"operation": "chunk_insert", "after_segment_id": "s1", "label": None, "text": ""}},
        {"id": "spk", "correction_type": "speaker",
         "payload": {"operation": "speaker_assign", "segment_ids": ["s0", "s1"], "entity_id": "e1"}},
        {"id": "spk-old", "correction_type": "speaker", "superseded": True,
         "payload": {"operation": "speaker_assign", "segment_ids": ["s1"], "entity_id": "e0"}},
        {"id": "mark", "correction_type": "mark", "payload": {"segment_id": "s1"}},
        {"id": "nudge", "correction_type": "timing", "payload": {"operation": "time_nudge"}},
        {"id": "edit", "correction_type": "text_content", "payload": {"segment_id": "s2"}},
    ]
    dep = RS.classify_dependents(rows, old)
    assert [c["id"] for c in dep["events"]] == ["ev"]
    assert [c["id"] for c in dep["speakers"]] == ["spk"]
    assert sorted(c["id"] for c in dep["stranded"]) == ["edit", "ev-proposed", "ev-words", "mark", "nudge", "split"]
    text = RS.describe_dependents(dep, reviews=3)
    assert "1 event insert(s) + 1 speaker assignment(s)" in text and "6 would STRAND" in text and "3 review" in text
    assert "STRAND" not in RS.describe_dependents({"events": [], "speakers": [], "stranded": []})


def test_build_respine_op_and_replay():
    entry = RespinedChunkEntry(audio_segment="aseg", transcript="t1", run="decomp_x", op_id="op-1", ts=1.0,
                               old_segments=["o1", "o2"], new_segments=["n1", "n2", "n3"],
                               stranded=["c9"], carried=["c1"], straddles=["n2"])
    prior = [{"op_id": "op-0", "audio_segment": "other"}, {"op_id": "op-1", "audio_segment": "aseg", "stale": True}]
    op = RS.build_respine_op(source_id="src", entry=entry, skeleton_hash=H, renumber=[("t1", 6), ("t2", 7)],
                             existing_entries=prior, actor="human:t")
    assert op["verb"] == RESPINE_OP_VERB and op["args"]["act"] == "chunk-respine"
    assert op["args"]["old_segments"] == 2 and op["args"]["new_segments"] == 3 and op["args"]["renumbered"] == 2
    ups = {u["id"]: u["properties"] for u in op["updates"]}
    assert ups["o1"] == {SEGMENT_SUPERSEDED_BY_PROP: "op-1"} and ups["o2"] == {SEGMENT_SUPERSEDED_BY_PROP: "op-1"}
    assert ups["t1"] == {"index": 6} and ups["t2"] == {"index": 7}
    entries = ups["src"][SOURCE_RESPINED_CHUNKS_PROP]
    assert [e["op_id"] for e in entries] == ["op-0", "op-1"], "the same op id replaces its stale entry"
    assert RespinedChunkEntry.from_dict(entries[-1]) == entry
    # replay: property merges, the registered handler
    assert decomp_replay_handlers()[RESPINE_OP_VERB] is apply_spine_fact
    nodes = {"o1": {"properties": {"index": 3, "text": "keep"}}, "o2": {"properties": {"index": 4}},
             "t1": {"properties": {"index": 5}}, "t2": {"properties": {"index": 6}}, "src": {"properties": {}}}

    async def fake_graph_task(queue, gid, method, **kw):
        assert method == "update_node"
        nodes[kw["node_id"]]["properties"].update(kw["properties"])

    import cjm_transcript_decomp_core.retire as R
    saved = R.graph_task
    R.graph_task = fake_graph_task
    try:
        asyncio.run(apply_spine_fact(None, "g", op))
    finally:
        R.graph_task = saved
    assert nodes["o1"]["properties"] == {"index": 3, "text": "keep", SEGMENT_SUPERSEDED_BY_PROP: "op-1"}
    assert nodes["t1"]["properties"]["index"] == 6 and nodes["t2"]["properties"]["index"] == 7
    assert [e["op_id"] for e in nodes["src"]["properties"][SOURCE_RESPINED_CHUNKS_PROP]] == ["op-0", "op-1"]
    # the op id is deterministic from (source, transcript)
    assert RS.respine_op_id("src", "t1") == RS.respine_op_id("src", "t1") != RS.respine_op_id("src", "t2")


def test_extension_payload_salt_forks_ids_keeps_hash_and_bridges():
    segs = [DecompSegment(index=7, text="Gamma.", start_time=300.0, end_time=302.0, chunk_start=0.0,
                          chunk_end=2.0, vad_chunk_index=0, pseg_index=1,
                          variants=[SegmentVariant("voxtral", "Gamma.", 0, 6)], text_from="voxtral"),
            DecompSegment(index=8, text="Delta.", start_time=302.0, end_time=304.0, chunk_start=2.0,
                          chunk_end=4.0, vad_chunk_index=1, pseg_index=1,
                          variants=[SegmentVariant("voxtral", "Delta.", 7, 13)], text_from="voxtral")]
    plain, _, ids0 = build_extension_payload(SOURCE_ENTRY, CAPS, H, "voxtral", segs)
    salted, edges, ids1 = build_extension_payload(SOURCE_ENTRY, CAPS, H, "voxtral", segs, identity_salt="t-landed")
    again, _, ids2 = build_extension_payload(SOURCE_ENTRY, CAPS, H, "voxtral", segs, identity_salt="t-landed")
    assert ids0["segments"] != ids1["segments"] and ids1["segments"] == ids2["segments"]
    rend = audio_rendition_node_id(audio_segment_node_id(SID, 300.0, 600.0), [])
    assert ids1["segments"][0] == segment_node_id(rend, H, 0.0, 2.0, identity_salt="t-landed")
    for n in salted:
        assert n["properties"]["skeleton_hash"] == H and n["properties"]["identity_salt"] == "t-landed"
        assert n["properties"]["index"] in (7, 8)
    for n in plain:
        assert "identity_salt" not in n["properties"]
    # the chunk's own NEXT chain + the bridges to the live neighbours
    nxt = [(e["source_id"], e["target_id"]) for e in edges if e["relation_type"] == "NEXT"]
    assert nxt == [(ids1["segments"][0], ids1["segments"][1])]
    br = RS.bridge_edges("prev", ids1["segments"], "after")
    assert [(e["source_id"], e["target_id"], e["relation_type"]) for e in br] == [
        ("prev", ids1["segments"][0], "NEXT"), (ids1["segments"][1], "after", "NEXT")]
    assert RS.bridge_edges(None, ids1["segments"], None) == [] and RS.bridge_edges("p", [], "n") == []


def test_alignment_composition_pseg_filter_keeps_original_positions():
    comp, metas = build_alignment_composition(SOURCE_ENTRY["segments"], "silero", "qwen3", ["voxtral"],
                                              pseg_indices={1})
    assert len(metas) == 1 and metas[0]["pseg_index"] == 1 and metas[0]["seg_start"] == 300.0
    assert [n.id for n in comp.nodes] == ["vad_0001", "fa_t0_0001"]
    full, fmetas = build_alignment_composition(SOURCE_ENTRY["segments"], "silero", "qwen3", ["voxtral"])
    assert len(fmetas) == 3 and len(full.nodes) == 6


def test_decomp_config_from_snapshot_is_headless_and_tolerant():
    cfg = RS.decomp_config_from({"text_from": "voxtral", "event_split": True, "respine": True,
                                 "event_classes": ["inhale"], "assume_yes": False, "force": True,
                                 "unknown_future_key": 1})
    assert isinstance(cfg, DecompConfig) and cfg.text_from == "voxtral" and cfg.event_split and cfg.respine
    assert cfg.assume_yes is True and cfg.force is False


def test_manifest_027_carries_the_respine_lineage():
    m = DecompManifest(run_id="r", created_at=1.0, config={}, source_manifest="x", parent_run_id="parent",
                       respined_chunk={"audio_segment": "a", "transcript": "t"})
    d = m.to_dict()
    assert d["version"] == "0.2.7" and d["parent_run_id"] == "parent" and d["respined_chunk"]["transcript"] == "t"
    assert DecompManifest(run_id="r", created_at=1.0, config={}, source_manifest="x").to_dict()["respined_chunk"] == {}


def test_transfer_handler_discovery_is_optional():
    # No registration in a bare env resolves to None (the verb then refuses to strand silently);
    # a registered one loads. Both shapes go through the same entry-point group.
    assert RS.TRANSFER_ENTRY_POINT_GROUP == "cjm_transcript_decomp_core.chunk_transfer"
    h = RS.transfer_handler()
    assert h is None or callable(h)


def test_respine_chunk_parses():
    p = build_parser()
    a = p.parse_args(["respine-chunk", "--source", "Bonus", "--chunk", "3", "--prompt", "--graph-db-path", "/g.db"])
    assert a.command == "respine-chunk" and a.chunk == 3 and a.prompt and a.text_file is None and not a.strand
    b = p.parse_args(["respine-chunk", "--source", "0b877597", "--at-time", "660.5", "--text-file", "/p.txt",
                      "--model-id", "gemini-3.8-flash", "--prompt-hash", "sha256:abc", "--strand", "--dry-run"])
    assert b.at_time == 660.5 and b.text_file == "/p.txt" and b.model_id == "gemini-3.8-flash" and b.strand and b.dry_run
    assert b.reason == "escalation" and b.text_source == "paste" and b.runs_dir is None


class _Graph:
    """query_nodes / query_edges over dict rows — enough for the chunk-context reads."""

    def __init__(self, segments, edges):
        self.segments = segments  # id -> props (+ "rend")
        self.edges = edges        # (relation, source, target)

    async def __call__(self, queue, gid, method, **kw):
        q = kw["query"]
        if method == "query_nodes":
            rows = []
            for sid, p in self.segments.items():
                rel = q.get("related") or {}
                far = set(rel.get("node_ids") or ([rel["node_id"]] if rel.get("node_id") else []))
                if far and p["rend"] not in far:
                    continue
                ok = True
                for w in q.get("where") or []:
                    v = p.get(w["prop"])
                    if w["op"] == "eq" and v != w["value"]:
                        ok = False
                    if w["op"] == "is_null" and v is not None:
                        ok = False
                if ok:
                    rows.append({"id": sid, **{k: p.get(k) for k in (q.get("project") or [])}})
            off, lim = q.get("offset") or 0, q.get("limit")
            return SimpleNamespace(rows=rows[off:off + lim] if lim else rows[off:])
        if method == "query_edges":
            tg = set(q.get("target_ids") or [])
            return SimpleNamespace(rows=[{"id": f"{r}-{s}-{t}", "source_id": s, "target_id": t}
                                         for r, s, t in self.edges if r == q["relation_type"] and t in tg])
        raise AssertionError(method)


def test_live_reads_apply_the_one_predicate(monkeypatch):
    segs = {"a0": {"rend": "r0", "skeleton_hash": H, "index": 0, "start_time": 0.0, "end_time": 1.0},
            "a1": {"rend": "r1", "skeleton_hash": H, "index": 1, "start_time": 300.0, "end_time": 301.0},
            "a1-old": {"rend": "r1", "skeleton_hash": H, "index": 1, "start_time": 300.0, "end_time": 301.5,
                       SEGMENT_SUPERSEDED_BY_PROP: "op-x"},
            "a2": {"rend": "r1", "skeleton_hash": H, "index": 2, "start_time": 301.0, "end_time": 302.0},
            "b0": {"rend": "r1", "skeleton_hash": "sha256:sibling", "index": 0, "start_time": 300.0, "end_time": 302.0},
            "a3": {"rend": "r2", "skeleton_hash": H, "index": 3, "start_time": 600.0, "end_time": 601.0}}
    g = _Graph(segs, [("CORRECTS", "c1", "a1"), ("CORRECTS", "c2", "a2"), ("REVIEWED", "sess", "a1"),
                      ("SUPERSEDES", "c3", "c2")])
    monkeypatch.setattr(RS, "graph_task", g)
    chunk = asyncio.run(RS.live_chunk_segments(None, "g", "r1", H))
    assert [r["id"] for r in chunk] == ["a1", "a2"], "superseded + sibling-spine rows are out"
    spine = asyncio.run(RS.live_spine_index(None, "g", ["r0", "r1", "r2"], H, page=2))
    assert spine == [("a0", 0), ("a1", 1), ("a2", 2), ("a3", 3)]

    async def nodes_by_id(queue, gid, method, **kw):
        q = kw["query"]
        if method == "query_edges":
            return await g(queue, gid, method, **kw)
        props = {"c1": {"correction_type": "mark", "payload": {"segment_id": "a1"}},
                 "c2": {"correction_type": "speaker", "payload": {"operation": "speaker_assign", "segment_ids": ["a2"]}}}
        return SimpleNamespace(rows=[{"id": i, **props[i]} for i in q["ids"] if i in props])
    monkeypatch.setattr(RS, "graph_task", nodes_by_id)
    rows, reviews = asyncio.run(RS.read_dependents(None, "g", ["a1", "a2"]))
    assert reviews == 1
    by = {r["id"]: r for r in rows}
    assert by["c1"]["superseded"] is False and by["c2"]["superseded"] is True
    dep = RS.classify_dependents(rows, {"a1", "a2"})
    assert [c["id"] for c in dep["stranded"]] == ["c1"] and dep["speakers"] == []
