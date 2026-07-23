"""Tests for cjm_transcript_decomp_core.graph — fine-spine extension payload.

Projected from the graph notebook's smoke-check cell at the golden-reference flip
(pure; no capabilities involved)."""
from cjm_transcript_graph_schema.schema import (
    audio_rendition_node_id,
    audio_segment_node_id,
    source_node_id,
    transcript_node_id,
)

from cjm_transcript_decomp_core.graph import build_extension_payload, resolve_root_ids
from cjm_transcript_decomp_core.models import DecompSegment, SegmentVariant

SOURCE_ENTRY = {
    "source_path": "/media/ep1.mp3",
    "content_hash": "sha256:src",
    "segments": [
        {"start": 0.0, "end": 300.0, "model_input_hash": "sha256:wav0",
         "transcripts": {"whisper": {}, "voxtral": {}}},
        {"start": 300.0, "end": 600.0, "model_input_hash": "sha256:wav1",
         "transcripts": {"whisper": {}, "voxtral": {}}},
    ],
}
CAPABILITIES = {"whisper": {"config_hash": "sha256:cw"}, "voxtral": {"config_hash": "sha256:cv"}}


def _segments():
    return [
        DecompSegment(index=0, text="Alpha.", start_time=0.0, end_time=2.0,
                      chunk_start=0.0, chunk_end=2.0, vad_chunk_index=0, pseg_index=0,
                      variants=[SegmentVariant("voxtral", "Alpha.", 0, 6),
                                SegmentVariant("whisper", "alfa", 0, 4)]),
        DecompSegment(index=1, text="Beta.", start_time=2.0, end_time=4.0,
                      chunk_start=2.0, chunk_end=4.0, vad_chunk_index=1, pseg_index=0,
                      variants=[SegmentVariant("voxtral", "Beta.", 7, 12)]),
        DecompSegment(index=2, text="", start_time=300.5, end_time=301.0,
                      chunk_start=0.5, chunk_end=1.0, vad_chunk_index=0, pseg_index=1,
                      variants=[]),  # D14-class empty segment: audio ref only
    ]


def test_resolve_root_ids_recomputation():
    roots = resolve_root_ids(SOURCE_ENTRY, CAPABILITIES)
    assert roots["source"] == source_node_id("sha256:src") and roots["chain"] == []
    a0 = audio_segment_node_id(roots["source"], 0.0, 300.0)
    assert roots["audio_segments"][0]["audio_segment"] == a0
    # raw rendition (empty chain) + transcript keyed on the RENDITION
    r0 = audio_rendition_node_id(a0, [])
    assert roots["audio_segments"][0]["rendition"] == r0
    assert roots["audio_segments"][1]["transcripts"]["voxtral"] == transcript_node_id(
        roots["audio_segments"][1]["rendition"], "voxtral", "sha256:cv")


def test_preprocessed_manifest_distinct_renditions_same_boundaries():
    # a preprocessed manifest (chain set) recomputes DISTINCT rendition/transcript ids,
    # SAME audio-segment ids — raw + vocals fine spines coexist under one boundary.
    roots = resolve_root_ids(SOURCE_ENTRY, CAPABILITIES)
    a0 = audio_segment_node_id(roots["source"], 0.0, 300.0)
    r0 = audio_rendition_node_id(a0, [])
    roots_vox = resolve_root_ids({**SOURCE_ENTRY, "chain": ["source_separation:demucs@cfg"]},
                                 CAPABILITIES)
    assert roots_vox["audio_segments"][0]["audio_segment"] == a0
    assert roots_vox["audio_segments"][0]["rendition"] != r0
    assert (roots_vox["audio_segments"][0]["transcripts"]["voxtral"]
            != roots["audio_segments"][0]["transcripts"]["voxtral"])


def test_extension_payload_deterministic_ids():
    roots = resolve_root_ids(SOURCE_ENTRY, CAPABILITIES)
    segs = _segments()
    nodes, edges, ids = build_extension_payload(SOURCE_ENTRY, CAPABILITIES, "sha256:vad", "voxtral", segs)
    assert len(nodes) == 3 and ids["source"] == roots["source"]
    assert ids["renditions"] == [a["rendition"] for a in roots["audio_segments"]]
    # deterministic + audio-side identity: rebuild reproduces ids byte-identically
    nodes2, edges2, ids2 = build_extension_payload(SOURCE_ENTRY, CAPABILITIES, "sha256:vad", "voxtral", segs)
    assert ids2["segments"] == ids["segments"]
    assert [e["id"] for e in edges2] == [e["id"] for e in edges]


def test_edge_topology_spans_coarse_boundary():
    # edge topology: STARTS_WITH per OWNING rendition (2), PART_OF per segment (3),
    # NEXT source-wide (2 — crosses the pseg boundary); PART_OF targets the rendition
    nodes, edges, ids = build_extension_payload(SOURCE_ENTRY, CAPABILITIES, "sha256:vad", "voxtral", _segments())
    rels = [e["relation_type"] for e in edges]
    assert rels.count("STARTS_WITH") == 2 and rels.count("PART_OF") == 3 and rels.count("NEXT") == 2
    part_of_targets = {e["target_id"] for e in edges if e["relation_type"] == "PART_OF"}
    assert part_of_targets <= set(ids["renditions"]), "fine spine PART_OF the rendition"
    nexts = [(e["source_id"], e["target_id"]) for e in edges if e["relation_type"] == "NEXT"]
    assert nexts == [(ids["segments"][0], ids["segments"][1]), (ids["segments"][1], ids["segments"][2])]


def test_provenance_refs_and_text_from():
    # provenance: audio TimeSlice ref + per-transcriber CharSlice refs; text_from recorded
    roots = resolve_root_ids(SOURCE_ENTRY, CAPABILITIES)
    a0 = audio_segment_node_id(roots["source"], 0.0, 300.0)
    r0 = audio_rendition_node_id(a0, [])
    nodes, edges, ids = build_extension_payload(SOURCE_ENTRY, CAPABILITIES, "sha256:vad", "voxtral", _segments())
    n0 = nodes[0]
    assert n0["properties"]["text_from"] == roots["audio_segments"][0]["transcripts"]["voxtral"]
    assert n0["properties"]["rendition_id"] == r0
    assert len(n0["sources"]) == 3  # audio + 2 variants
    assert n0["sources"][0]["slice"]["kind"] == "time" and n0["sources"][1]["slice"]["kind"] == "char"
    assert n0["sources"][0]["locator"]["node_id"] == r0, "audio ref points at the rendition"
    # empty segment: audio ref only, no text_from
    n2 = nodes[2]
    assert len(n2["sources"]) == 1 and "text_from" not in n2["properties"]
    assert n2["properties"]["text"] == ""


def test_parallel_spines_disjoint_ids_and_split_metadata():
    # DEC f1024568: a split run passes a DIFFERENT skeleton hash — every node id
    # forks, so the split spine coexists with the original by construction.
    base_nodes, _, base_ids = build_extension_payload(
        SOURCE_ENTRY, CAPABILITIES, "sha256:vad", "voxtral", _segments())
    split_nodes, _, split_ids = build_extension_payload(
        SOURCE_ENTRY, CAPABILITIES, "sha256:skel-split", "voxtral", _segments(),
        split_policy="sentence-split/v1")
    assert set(base_ids["segments"]).isdisjoint(split_ids["segments"])
    # The identity input rides every node as the queryable skeleton_hash prop;
    # split runs also record their policy tag (base runs carry no split_policy).
    assert all(n["properties"]["skeleton_hash"] == "sha256:vad" for n in base_nodes)
    assert all("split_policy" not in n["properties"] for n in base_nodes)
    assert all(n["properties"]["skeleton_hash"] == "sha256:skel-split" for n in split_nodes)
    assert all(n["properties"]["split_policy"] == "sentence-split/v1" for n in split_nodes)
