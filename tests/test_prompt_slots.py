"""render_chunk_prompt passes a LIVE slot-text provider through to the transcription
core's renderer (finding c63cd2e3): the correction app's corrected spine fills the
slots; the CLI path (no provider) keeps the manifest source."""
from cjm_transcript_decomp_core.respine import render_chunk_prompt


def _ctx():
    def seg(i, start, end, text):
        return {"index": i, "start": start, "end": end, "model_input_path": f"/audio/{i}.wav",
                "transcripts": {"whisper": {"text": text}}}
    segs = [seg(0, 0, 100, "raw zero"), seg(1, 100, 200, "raw one")]
    tm = {"config": {"transcriber_capabilities": ["whisper"]}, "collections": [],
          "sources": [{"source_path": "x/Lecture.mp4", "segments": segs}]}
    return {"transcription_manifest": tm, "source_index": 0, "chunk_entry": segs[1],
            "decomp_manifest": {"config": {"text_from": "whisper"}}, "old_segments": [1, 2, 3]}


def test_render_chunk_prompt_passes_the_provider_through():
    r = render_chunk_prompt(_ctx(), slot_text=lambda s, e: f"LIVE {s:.0f}-{e:.0f}")
    assert r["slots"]["prev_text"] == "LIVE 0-100" and r["slots"]["draft_text"] == "LIVE 100-200"
    assert r["slot_sources"]["draft_text"] == "spine"
    assert r["chunk"] == 1 and r["chunk_range"] == (100.0, 200.0) and r["live_segments"] == 3
    plain = render_chunk_prompt(_ctx())
    assert plain["slots"]["draft_text"] == "raw one" and plain["slot_sources"]["draft_text"] == "manifest"
    assert plain["prompt_hash"] == r["prompt_hash"]
