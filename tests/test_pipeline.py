"""Tests for cjm_transcript_decomp_core.pipeline — pure-logic checks (no capabilities involved).

Projected from the pipeline notebook's import-smoke and composition-builder cells at
the golden-reference flip."""
from cjm_capability_primitives.forced_alignment import ForcedAlignItem, ForcedAlignResult
from cjm_capability_primitives.vad import TimeRange, VADResult
from cjm_substrate.core.ports import new_composition_run

from cjm_transcript_decomp_core.pipeline import (
    build_alignment_composition,
    decompose_source,
    fa_words_from_result,
    load_source_manifest,
    run_decomp,
    vad_chunks_from_result,
)

SEGS = [
    {"model_input_path": "/s0.wav", "start": 0.0,
     "transcripts": {"whisper": {"text": "alpha"}, "voxtral": {"text": "Alpha."}}},
    {"model_input_path": "/s1.wav", "start": 300.0,
     "transcripts": {"whisper": {"text": "  "}, "voxtral": {"text": ""}}},
    {"model_input_path": "/s2.wav", "start": 600.0,
     "transcripts": {"whisper": {"text": "beta"}, "voxtral": {"text": ""}}},
]


def test_pipeline_symbols_importable():
    assert callable(run_decomp)
    assert callable(decompose_source)
    assert callable(load_source_manifest)


def test_normalizers_fold_typed_results():
    chunks = vad_chunks_from_result(VADResult(
        ranges=[TimeRange(start=5.0, end=9.0), TimeRange(start=0.5, end=2.0)]))
    assert [(c.index, c.start_time) for c in chunks] == [(0, 0.5), (1, 5.0)]
    words = fa_words_from_result(ForcedAlignResult(
        items=[ForcedAlignItem(text="hi", start_time=0.1, end_time=0.4)]))
    assert words[0].text == "hi" and words[0].end_time == 0.4


def test_alignment_composition_shape():
    # Whole-source M×(VAD ∥ T×FA) shape (stage 5: per-transcriber FA off the
    # shared skeleton): one VAD node per pseg; one FA node per transcriber with
    # non-empty text; psegs where ALL transcribers are empty skip entirely.
    comp, metas = build_alignment_composition(SEGS, "silero", "qwen3", ["whisper", "voxtral"])
    # pseg0: vad + 2 FA; pseg1: skipped; pseg2: vad + 1 FA (voxtral empty there)
    assert len(comp.nodes) == 5 and len(metas) == 3
    assert metas[1]["skipped"] is True and metas[1]["pseg_index"] == 1
    assert metas[0]["fa_nodes"] == {"whisper": "fa_t0_0000", "voxtral": "fa_t1_0000"}
    assert metas[2]["fa_nodes"] == {"whisper": "fa_t0_0002"}


def test_alignment_nodes_ride_the_task_channel():
    comp, metas = build_alignment_composition(SEGS, "silero", "qwen3", ["whisper", "voxtral"])
    # stage 8: the VAD node rides the task channel (vad/detect_speech); force -> control
    assert comp.nodes[0].kwargs == {"audio": "/s0.wav"}
    assert comp.nodes[0].task_name == "vad" and comp.nodes[0].method == "detect_speech"
    assert comp.nodes[0].control == {"force": False}
    # stage 8: the FA node ALSO rides the task channel (forced_alignment/align); force -> control
    assert comp.nodes[2].kwargs == {"audio": "/s0.wav", "text": "Alpha."}
    assert comp.nodes[2].task_name == "forced_alignment" and comp.nodes[2].method == "align"
    assert comp.nodes[2].control == {"force": False}
    run = new_composition_run(comp, "r")
    assert set(run.ready_nodes()) == {"vad_0000", "fa_t0_0000", "fa_t1_0000", "vad_0002", "fa_t0_0002"}


def test_alignment_composition_seg_nodes():
    # B.5: with a segmentation capability + authoritative transcriber, each
    # non-skipped pseg where the authoritative transcriber has text also gets
    # one sentence_segmentation/segment_text node over that text — riding the
    # SAME composition (the text comes from the manifest, not from FA).
    comp, metas = build_alignment_composition(
        SEGS, "silero", "qwen3", ["whisper", "voxtral"],
        seg_id="pysbd", seg_text_from="voxtral")
    by_name = {n.id: n for n in comp.nodes}
    # pseg0: voxtral has text -> seg node; pseg1: skipped; pseg2: voxtral empty -> none.
    assert metas[0]["seg_node"] == "seg_0000"
    seg = by_name["seg_0000"]
    assert seg.kwargs == {"text": "Alpha."}
    assert seg.task_name == "sentence_segmentation" and seg.method == "segment_text"
    assert "seg_node" not in metas[2]
    assert len(comp.nodes) == 6
    # Without a seg capability the composition is unchanged (no seg nodes).
    comp2, metas2 = build_alignment_composition(SEGS, "silero", "qwen3",
                                                ["whisper", "voxtral"])
    assert len(comp2.nodes) == 5 and all("seg_node" not in m for m in metas2)


def test_compute_skeleton_hash_identity_and_respine():
    """DEC f1024568 + 9241564f: no-split = the raw VAD hash (legacy-identical
    ids); split = the B.5 composite; --respine widens EITHER identity with the
    run token — distinct from the config-identical spine, deterministic per
    token, and never equal across tokens."""
    from cjm_transcript_decomp_core.pipeline import compute_skeleton_hash
    vad = "vadhash123"
    assert compute_skeleton_hash(vad) == vad
    split = compute_skeleton_hash(vad, split_policy="pysbd-sentence-v2",
                                  split_min_chunk_s=0.5,
                                  seg_capability="cjm-capability-pysbd",
                                  seg_config_hash="seghash")
    assert split != vad
    assert split == compute_skeleton_hash(vad, split_policy="pysbd-sentence-v2",
                                          split_min_chunk_s=0.5,
                                          seg_capability="cjm-capability-pysbd",
                                          seg_config_hash="seghash")
    r1 = compute_skeleton_hash(vad, respine_token="decomp_run_1")
    r2 = compute_skeleton_hash(vad, respine_token="decomp_run_2")
    assert r1 != vad and r2 != vad and r1 != r2
    assert r1 == compute_skeleton_hash(vad, respine_token="decomp_run_1")
    rs = compute_skeleton_hash(vad, split_policy="pysbd-sentence-v2",
                               split_min_chunk_s=0.5,
                               seg_capability="cjm-capability-pysbd",
                               seg_config_hash="seghash",
                               respine_token="decomp_run_1")
    assert rs not in (split, r1, vad)
    # DEC a6e4c040: the TEXT AUTHORITY joins the identity — a transcriber
    # config change (the verbatim-prompt case) mints a sibling spine instead
    # of colliding; same transcriber config converges; respine still stacks
    # OUTSIDE it (the deliberate widening stays outermost).
    t1 = compute_skeleton_hash(vad, text_from_capability="voxtral",
                               text_from_config_hash="cfg_A")
    t2 = compute_skeleton_hash(vad, text_from_capability="voxtral",
                               text_from_config_hash="cfg_B")
    assert t1 != vad and t2 != vad and t1 != t2
    assert t1 == compute_skeleton_hash(vad, text_from_capability="voxtral",
                                       text_from_config_hash="cfg_A")
    tr = compute_skeleton_hash(vad, text_from_capability="voxtral",
                               text_from_config_hash="cfg_A",
                               respine_token="decomp_run_1")
    assert tr not in (t1, r1, vad)


def test_event_carve_identity_and_propset_loading(tmp_path):
    """Respine trial DEC 6cc10fb7: the event composite widens the skeleton
    identity (propset id + classes are inputs), and event_spans_from_propset
    consumes a set by pointer (dir or json), filtering to the carve classes."""
    import json
    from cjm_transcript_decomp_core.pipeline import compute_skeleton_hash, event_spans_from_propset

    vad = "vadhash123"
    ev = compute_skeleton_hash(vad, event_policy="event-carve/v1",
                               event_propset_id="propset_a", event_classes=["inhale"])
    assert ev != vad
    assert ev == compute_skeleton_hash(vad, event_policy="event-carve/v1",
                                       event_propset_id="propset_a", event_classes=["inhale"])
    # Different set, different classes: different spines by construction.
    assert ev != compute_skeleton_hash(vad, event_policy="event-carve/v1",
                                       event_propset_id="propset_b", event_classes=["inhale"])
    assert ev != compute_skeleton_hash(vad, event_policy="event-carve/v1",
                                       event_propset_id="propset_a", event_classes=["inhale", "dead-air"])
    # The event composite stacks on the sentence composite, and respine widens both.
    split = compute_skeleton_hash(vad, split_policy="sentence-split/capability",
                                  seg_capability="cjm-capability-pysbd", seg_config_hash="seghash")
    both = compute_skeleton_hash(vad, split_policy="sentence-split/capability",
                                 seg_capability="cjm-capability-pysbd", seg_config_hash="seghash",
                                 event_policy="event-carve/v1", event_propset_id="propset_a",
                                 event_classes=["inhale"])
    assert both not in (vad, split, ev)
    assert compute_skeleton_hash(vad, event_policy="event-carve/v1",
                                 event_propset_id="propset_a", event_classes=["inhale"],
                                 respine_token="run1") != ev

    set_dir = tmp_path / "propset_test"
    set_dir.mkdir()
    (set_dir / "manifest.json").write_text(json.dumps({
        "format": "cjm-capability-pyannote/proposal-set-manifest",
        "proposal_set_id": "propset_test",
        "files": {"proposals": "proposals.jsonl"},
    }))
    rows = [
        {"proposal_id": "p1", "label": "inhale", "start_time": 5.0, "end_time": 5.4, "score": 0.9},
        {"proposal_id": "p2", "label": "hesitation-marker", "start_time": 1.0, "end_time": 1.5, "score": 0.8},
        {"proposal_id": "p3", "label": "inhale", "start_time": 2.0, "end_time": 2.3, "score": 0.7},
        {"proposal_id": "p4", "label": "inhale", "start_time": 9.0, "end_time": 9.0, "score": 0.5},
        # Dual-tier set (propset manifest 0.2.0): the audition tier must NEVER
        # carve; an explicit tier-1 tag carves like the legacy tierless rows.
        {"proposal_id": "p5", "label": "inhale", "start_time": 7.0, "end_time": 7.3,
         "score": 0.42, "tier": 2},
        {"proposal_id": "p6", "label": "inhale", "start_time": 3.0, "end_time": 3.2,
         "score": 0.8, "tier": 1},
    ]
    (set_dir / "proposals.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    # Dir pointer: classes filter, tier filter, ordering, empty-span drop all hold.
    manifest, spans = event_spans_from_propset(set_dir, ["inhale"])
    assert manifest["proposal_set_id"] == "propset_test"
    assert spans == [(2.0, 2.3), (3.0, 3.2), (5.0, 5.4)]
    # Json pointer resolves the same set.
    _, spans2 = event_spans_from_propset(set_dir / "manifest.json", ["inhale"])
    assert spans2 == spans

    import pytest
    with pytest.raises(RuntimeError):
        event_spans_from_propset(tmp_path / "missing", ["inhale"])
