"""Pure-logic tests for the segments drill (segments.py below the stack seam):
spine-order re-imposition, gap derivation (a LEADING gap is the de994164
headline case, not an edge case), display interleave, timestamp formatting,
and the recorded-VAD-config summary."""

import json
from cjm_transcript_decomp_core.segments import (aseg_index_for, build_display,
                                                capability_config_schema,
                                                find_gaps, fmt_ts, locate_span,
                                                ordered_rows, predicted_rows,
                                                probe_compare, realign_rows,
                                                split_predicted, vad_summary)


def _row(i, s, e, text="t"):
    """A minimal projected segment row (the read_segments projection shape)."""
    return {"id": f"s{i}", "index": i, "start_time": s, "end_time": e, "text": text}


def test_ordered_rows_reimposes_manifest_order():
    by_id = {"b": {"id": "b"}, "a": {"id": "a"}}
    assert ordered_rows(by_id, ["a", "missing", "b"]) == [{"id": "a"}, {"id": "b"}]


def test_find_gaps_leading_and_mid():
    rows = [_row(0, 12.0, 15.0), _row(1, 15.4, 20.0), _row(2, 26.0, 30.0)]
    assert find_gaps(rows, 2.0) == [(0, 12.0), (2, 6.0)]


def test_find_gaps_below_threshold_and_overlap():
    # Overlapping timings must not mint a negative gap (coverage is monotonic).
    rows = [_row(0, 0.0, 10.0), _row(1, 8.0, 9.0), _row(2, 10.5, 12.0)]
    assert find_gaps(rows, 2.0) == []


def test_find_gaps_skips_untimed_rows():
    rows = [{"id": "x", "start_time": None, "end_time": None}, _row(1, 5.0, 6.0)]
    assert find_gaps(rows, 2.0) == [(1, 5.0)]


def test_build_display_interleaves_focusable_gaps():
    rows = [_row(0, 3.0, 5.0), _row(1, 5.1, 7.0)]
    assert build_display(rows, 2.0) == [("gap", 0, 3.0), ("seg", 0, 0.0),
                                        ("seg", 1, 0.0)]


def test_fmt_ts():
    assert fmt_ts(65.3) == "1:05.3"
    assert fmt_ts(0.0) == "0:00.0"
    assert fmt_ts(3671.25) in ("1:01:11.2", "1:01:11.3")  # float-repr rounding


def test_vad_summary_reads_recorded_config():
    m = {"config": {"vad_capability": "cjm-capability-silero-vad"},
         "capabilities": {"cjm-capability-silero-vad": {"config": {
             "threshold": 0.5, "min_speech_duration_ms": 250,
             "min_silence_duration_ms": 100, "speech_pad_ms": 30}}}}
    assert vad_summary(m) == ("silero-vad: thr 0.5 · min-speech 250ms · "
                              "min-sil 100ms · pad 30ms")


def test_vad_summary_absent():
    assert vad_summary({"config": {}, "capabilities": {}}) is None


def test_locate_span_resolves_and_clamps():
    asegs = [{"start": 0.0, "end": 60.0, "wav": "a.wav"},
             {"start": 60.0, "end": 120.0, "wav": "b.wav"}]
    assert locate_span(asegs, 44.1, 50.8) == ("a.wav", 44.1, 50.8)
    # A span crossing the coarse boundary clamps to its head's WAV (locate_span contract).
    assert locate_span(asegs, 55.0, 70.0) == ("a.wav", 55.0, 60.0)
    assert locate_span(asegs, 61.0, 65.0) == ("b.wav", 1.0, 5.0)
    assert locate_span([], 0.0, 1.0) is None
    assert locate_span([{"start": 0.0, "end": 60.0, "wav": None}], 5.0, 6.0) is None


def test_aseg_index_for_boundaries():
    # aseg_index_for: bisect over ordered starts, boundary lands rightward.
    asegs = [{"start": 0.0, "end": 60.0}, {"start": 60.0, "end": 120.0}]
    assert aseg_index_for(asegs, 0.0) == 0
    assert aseg_index_for(asegs, 59.9) == 0
    assert aseg_index_for(asegs, 60.0) == 1
    assert aseg_index_for([], 5.0) is None


def test_probe_compare_flags_recovered():
    committed = [(0.0, 44.1), (50.8, 60.0)]
    predicted = [(0.0, 44.0), (45.0, 50.0), (51.0, 60.0)]
    out = probe_compare(committed, predicted, [(44.1, 50.8)])
    assert out["committed"] == 2 and out["predicted"] == 3
    assert out["recovered"] == [(45.0, 50.0)]
    # No uncovered spans -> probe_compare recovers nothing, whatever the probe cut.
    assert probe_compare(committed, predicted, [])["recovered"] == []


def test_capability_config_schema_matches_code_name(tmp_path):
    # capability_config_schema matches code.name via json reads; bad files skip.
    (tmp_path / "x.json").write_text(json.dumps(
        {"code": {"name": "cap-x", "config_schema": {"properties": {}}}}))
    (tmp_path / "bad.json").write_text("{not json")
    assert capability_config_schema(str(tmp_path), "cap-x") == {"properties": {}}
    assert capability_config_schema(str(tmp_path), "cap-y") is None


def test_predicted_rows_borrow_text_and_flag_recovery():
    committed = [_row(0, 0.0, 10.0, "alpha"), _row(1, 20.0, 30.0, "beta")]
    rows = predicted_rows([(0.0, 9.0), (12.0, 18.0), (19.0, 31.0)],
                          committed, [(10.0, 20.0)])
    assert [r["text"] for r in rows] == ["alpha", "", "beta"]
    assert [r["recovered"] for r in rows] == [False, True, True]
    # predicted_rows output walks like committed rows — display interleave works.
    assert build_display(rows, 2.0)[0] == ("seg", 0, 0.0)


def test_realign_rows_reslices_pipeline_fold():
    # realign_rows = the pipeline's own fold, probe-side (words -> chunks -> text).
    text = "first segment text second segment text extra tail words"
    words = [("first", 0.6, 1.0), ("segment", 1.1, 1.5), ("text", 1.6, 2.0),
             ("second", 12.0, 12.4), ("segment", 12.5, 13.0), ("text", 13.1, 13.5),
             ("extra", 15.4, 15.8), ("tail", 16.0, 16.4), ("words", 16.5, 16.9)]
    chunks = [(0.5, 10.0), (11.9, 15.1), (15.3, 20.1)]
    assert realign_rows(text, words, chunks) == [
        "first segment text", "second segment text", "extra tail words"]


def test_predicted_rows_realigned_text_wins():
    committed = [_row(0, 0.0, 10.0, "alpha")]
    rows = predicted_rows([(0.0, 5.0), (5.5, 9.0)], committed, [],
                          realigned=["ay", "bee"])
    assert [r["text"] for r in rows] == ["ay", "bee"]
    assert all(r["realigned"] for r in rows)


def test_split_predicted_refines_probe_skeleton():
    # The probe-side sentence-split preview runs the pipeline's own stage with
    # CAPABILITY-delivered sentence spans (B.5): a predicted chunk holding two
    # sentences splits at the FA word gap, and the realign fold over the
    # refined spans yields <=1 sentence per chunk.
    text = "Hello world. Foo bar."
    words = [("hello", 0.0, 0.5), ("world", 0.5, 1.0),
             ("foo", 1.4, 2.5), ("bar", 2.5, 3.0)]
    sent_spans = [(0, 12), (13, 21)]  # "Hello world." / "Foo bar."
    predicted = [(0.0, 3.1)]
    refined = split_predicted(text, words, predicted, sent_spans)
    assert refined == [(0.0, 1.2), (1.2, 3.1)]
    assert realign_rows(text, words, refined) == ["Hello world.", "Foo bar."]
    # No sentence crossing -> the skeleton passes through unchanged.
    assert split_predicted(text, words, [(0.0, 1.2), (1.9, 3.1)], sent_spans) \
        == [(0.0, 1.2), (1.9, 3.1)]
    # One span covering everything (the abbreviation class, now the
    # segmenter's job) -> nothing cuts.
    assert split_predicted(text, words, predicted, [(0, len(text))]) == predicted
