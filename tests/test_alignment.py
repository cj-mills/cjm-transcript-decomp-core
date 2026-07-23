"""Tests for cjm_transcript_decomp_core.alignment — pure forced-alignment logic.

Projected from the alignment notebook's smoke-check cell at the golden-reference flip
(pure logic; deterministic)."""
from cjm_transcript_decomp_core.alignment import (
    assign_words_to_chunks,
    build_segments_from_alignment,
    map_fa_words_to_text,
    split_chunks_at_sentence_gaps,
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


def test_sentence_split_cuts_at_fa_word_gap():
    # One fluent chunk holding two sentences (the bc69e3e6 shape): the cut lands
    # at the midpoint of the FA gap after the sentence-ending word.
    text = "Hello world. Foo bar."
    fa = [FAWord("hello", 0.0, 0.5), FAWord("world", 0.5, 1.0),
          FAWord("foo", 1.4, 2.5), FAWord("bar", 2.5, 3.0)]
    chunks = [VADChunk(0, 0.0, 3.1)]
    spans = map_fa_words_to_text(text, fa)
    refined = split_chunks_at_sentence_gaps(chunks, fa, spans, text)
    assert [(c.start_time, c.end_time) for c in refined] == [(0.0, 1.2), (1.2, 3.1)]
    assert [c.index for c in refined] == [0, 1]
    # ...and the standard fold over the refined skeleton yields <=1 sentence per chunk.
    segs = build_segments_from_alignment(
        text, spans, assign_words_to_chunks(fa, refined), num_chunks=2)
    assert [s.text for s in segs] == ["Hello world.", "Foo bar."]


def test_sentence_split_no_op_cases():
    text = "Hello world. Foo bar."
    fa = [FAWord("hello", 0.0, 0.5), FAWord("world", 0.5, 1.0),
          FAWord("foo", 2.0, 2.5), FAWord("bar", 2.5, 3.0)]
    chunks = [VADChunk(0, 0.0, 1.2), VADChunk(1, 1.9, 3.1)]
    spans = map_fa_words_to_text(text, fa)
    # Already one sentence per chunk: the sentence ends fall on chunk-final
    # words, so nothing splits and the skeleton passes through re-indexed.
    refined = split_chunks_at_sentence_gaps(chunks, fa, spans, text)
    assert [(c.start_time, c.end_time) for c in refined] == [(0.0, 1.2), (1.9, 3.1)]
    # Textless/montage chunks (no words assigned anywhere) pass through whole.
    assert [(c.start_time, c.end_time)
            for c in split_chunks_at_sentence_gaps(chunks, [], [], "")] \
        == [(0.0, 1.2), (1.9, 3.1)]


def test_sentence_split_min_duration_guard():
    # The sentence gap sits 0.25s from the chunk end — a cut would mint a
    # sliver, so the guard refuses it and the chunk survives whole.
    text = "Hello world. Hi."
    fa = [FAWord("hello", 0.0, 0.5), FAWord("world", 0.5, 2.6),
          FAWord("hi", 2.7, 2.9)]
    chunks = [VADChunk(0, 0.0, 2.9)]
    spans = map_fa_words_to_text(text, fa)
    refined = split_chunks_at_sentence_gaps(chunks, fa, spans, text, min_chunk_s=0.5)
    assert [(c.start_time, c.end_time) for c in refined] == [(0.0, 2.9)]
    # A permissive guard lets the same cut through.
    loose = split_chunks_at_sentence_gaps(chunks, fa, spans, text, min_chunk_s=0.1)
    assert [(round(c.start_time, 2), round(c.end_time, 2)) for c in loose] \
        == [(0.0, 2.65), (2.65, 2.9)]


def test_sentence_split_trailing_closer_and_question():
    # 'dispatch."' and 'Why?' both end sentences under the v1 rule (trailing
    # closers stripped before the punctuation check).
    text = 'He said "dispatch." Why? Because.'
    fa = [FAWord("he", 0.0, 0.3), FAWord("said", 0.3, 0.6),
          FAWord("dispatch", 0.6, 1.5), FAWord("why", 2.1, 2.8),
          FAWord("because", 3.5, 4.2)]
    chunks = [VADChunk(0, 0.0, 4.4)]
    spans = map_fa_words_to_text(text, fa)
    refined = split_chunks_at_sentence_gaps(chunks, fa, spans, text)
    assert [(round(c.start_time, 2), round(c.end_time, 2)) for c in refined] \
        == [(0.0, 1.8), (1.8, 3.15), (3.15, 4.4)]


def test_sentence_split_v2_probe_drive_findings():
    # (1) Dotted abbreviations must not end sentences (2026-07-22 drive: 'U.S.'
    # split mid-sentence): no cut anywhere in this chunk.
    text = "Sitting down with the U.S. Secretary of Energy."
    fa = [FAWord("sitting", 0.0, 0.4), FAWord("down", 0.4, 0.8),
          FAWord("with", 0.8, 1.1), FAWord("the", 1.1, 1.3),
          FAWord("us", 1.3, 2.0), FAWord("secretary", 2.0, 2.9),
          FAWord("of", 2.9, 3.1), FAWord("energy", 3.1, 3.8)]
    chunks = [VADChunk(0, 0.0, 4.0)]
    spans = map_fa_words_to_text(text, fa)
    assert [(c.start_time, c.end_time)
            for c in split_chunks_at_sentence_gaps(chunks, fa, spans, text)] \
        == [(0.0, 4.0)]

    # (2) Contiguous FA words (zero gap at the cut): the boundary word must land
    # RIGHT of the cut — text and audio agree (the off-by-one drive finding).
    text2 = "Stagnant on energy growth. Any fundamental reason?"
    fa2 = [FAWord("stagnant", 0.0, 0.6), FAWord("on", 0.6, 0.8),
           FAWord("energy", 0.8, 1.2), FAWord("growth", 1.2, 1.8),
           FAWord("any", 1.8, 2.1), FAWord("fundamental", 2.1, 2.8),
           FAWord("reason", 2.8, 3.3)]
    chunks2 = [VADChunk(0, 0.0, 3.5)]
    spans2 = map_fa_words_to_text(text2, fa2)
    refined = split_chunks_at_sentence_gaps(chunks2, fa2, spans2, text2)
    assert [(c.start_time, c.end_time) for c in refined] == [(0.0, 1.8), (1.8, 3.5)]
    segs = build_segments_from_alignment(
        text2, spans2, assign_words_to_chunks(fa2, refined), num_chunks=2)
    assert [s.text for s in segs] == ["Stagnant on energy growth.",
                                      "Any fundamental reason?"]


def test_sentence_split_v3_title_stub_guard():
    # 'Mr. Gorbachev, tear down this wall.' (the SN1 probe find): the title
    # stub must not cut, even with a wide FA gap after 'Mr.'.
    text = "Mr. Gorbachev, tear down this wall."
    fa = [FAWord("mr", 0.0, 0.7), FAWord("gorbachev", 0.9, 1.7),
          FAWord("tear", 1.8, 2.2), FAWord("down", 2.2, 2.5),
          FAWord("this", 2.5, 2.7), FAWord("wall", 2.7, 3.2)]
    chunks = [VADChunk(0, 0.0, 3.4)]
    spans = map_fa_words_to_text(text, fa)
    assert [(c.start_time, c.end_time)
            for c in split_chunks_at_sentence_gaps(chunks, fa, spans, text)] \
        == [(0.0, 3.4)]
    # ...and a real sentence end right after a guarded stub still cuts there.
    text2 = "He met Dr. Smith. Then he left."
    fa2 = [FAWord("he", 0.0, 0.3), FAWord("met", 0.3, 0.6),
           FAWord("dr", 0.6, 0.9), FAWord("smith", 0.9, 1.6),
           FAWord("then", 2.4, 2.7), FAWord("he", 2.7, 2.9),
           FAWord("left", 2.9, 3.3)]
    chunks2 = [VADChunk(0, 0.0, 3.5)]
    spans2 = map_fa_words_to_text(text2, fa2)
    assert [(c.start_time, c.end_time)
            for c in split_chunks_at_sentence_gaps(chunks2, fa2, spans2, text2)] \
        == [(0.0, 2.0), (2.0, 3.5)]
