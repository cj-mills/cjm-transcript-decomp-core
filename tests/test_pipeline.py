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
