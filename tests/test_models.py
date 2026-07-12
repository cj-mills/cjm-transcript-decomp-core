"""Tests for cjm_transcript_decomp_core.models — decomposition data shapes.

Projected from the models notebook's smoke-check cell at the golden-reference flip."""
from cjm_transcript_decomp_core.models import (
    DecompManifest,
    DecompSegment,
    DecompSourceRecord,
    FAWord,
    SegmentVariant,
    VADChunk,
    new_run_id,
)


def test_faword_from_wire_tolerates_extra_keys():
    w = FAWord.from_wire({"text": "Hello", "start_time": 0.1, "end_time": 0.4, "extra": 1})
    assert w.text == "Hello" and w.end_time == 0.4


def test_vad_chunk_duration():
    c = VADChunk(index=0, start_time=1.0, end_time=2.5)
    assert abs(c.duration - 1.5) < 1e-9


def test_stage5_segment_shape():
    # stage-5 segment shape: audio-side identity fields + per-transcriber variants
    seg = DecompSegment(index=0, text="hi there", start_time=10.0, end_time=12.0,
                        chunk_start=1.0, chunk_end=3.0, vad_chunk_index=0, pseg_index=0,
                        variants=[SegmentVariant("whisper", "hi there", 0, 8),
                                  SegmentVariant("voxtral", "Hi, there", 0, 9)])
    d = seg.to_dict()
    assert d["chunk_start"] == 1.0 and d["pseg_index"] == 0
    assert d["variants"][1]["transcriber"] == "voxtral"


def test_manifest_shape_and_run_id():
    assert new_run_id().startswith("decomp_")
    m = DecompManifest(run_id="r", created_at=0.0, config={}, source_manifest="/tmp/s.json")
    md = m.to_dict()
    assert md["format"] == "cjm-transcript-decomp-core/run-manifest"
    assert md["version"] == "0.2.1" and md["sources"] == []
    rec = DecompSourceRecord(source_node_id="sid", source_path="/a.mp3", title="a",
                             segment_count=2, segment_ids=["s1", "s2"])
    assert rec.to_dict()["source_node_id"] == "sid"


def test_manifest_save_round_trip(tmp_path):
    import json
    m = DecompManifest(run_id="r", created_at=0.0, config={}, source_manifest="/tmp/s.json")
    out = m.save(tmp_path / "runs" / "m.json")
    assert out.exists()
    assert json.loads(out.read_text())["run_id"] == "r"
