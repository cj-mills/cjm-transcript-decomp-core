"""Tests for cjm_transcript_decomp_core.alignment — pure forced-alignment logic.

Projected from the alignment notebook's smoke-check cell at the golden-reference flip
(pure logic; deterministic)."""
from cjm_transcript_decomp_core.alignment import (
    assign_words_to_chunks,
    build_segments_from_alignment,
    map_fa_words_to_text,
    tier1_alignment_checks,
)
from cjm_transcript_decomp_core.models import FAWord, VADChunk

TEXT = "Hello world. Foo bar."
FA = [FAWord("hello", 0.0, 0.5), FAWord("world", 0.5, 1.0),
      FAWord("foo", 2.0, 2.5), FAWord("bar", 2.5, 3.0)]
CHUNKS = [VADChunk(0, 0.0, 1.2), VADChunk(1, 1.9, 3.1)]


def test_map_fa_words_to_text_spans():
    spans = map_fa_words_to_text(TEXT, FA)
    assert len(spans) == 4
    assert TEXT[spans[0][0]:spans[0][1]] == "Hello"
    assert TEXT[spans[1][0]:spans[1][1]] == "world."


def test_assign_words_to_chunks_by_timestamp():
    assert assign_words_to_chunks(FA, CHUNKS) == [0, 0, 1, 1]


def test_build_segments_per_chunk():
    spans = map_fa_words_to_text(TEXT, FA)
    assign = assign_words_to_chunks(FA, CHUNKS)
    segs = build_segments_from_alignment(TEXT, spans, assign, num_chunks=2)
    assert len(segs) == 2
    assert segs[0].text == "Hello world."
    assert segs[1].text == "Foo bar."
    assert tier1_alignment_checks(segs, CHUNKS) == []


def test_empty_chunk_yields_empty_segment_warning():
    # empty-chunk case -> empty-segment warning
    spans = map_fa_words_to_text(TEXT, FA)
    segs2 = build_segments_from_alignment(TEXT, spans, [0, 0, 0, 0], num_chunks=2)
    assert segs2[1].text == ""
    assert any("EMPTY" in w for w in tier1_alignment_checks(segs2, CHUNKS))
