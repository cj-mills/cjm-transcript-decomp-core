"""Tests for cjm_transcript_decomp_core.alignment — pure forced-alignment logic.

Projected from the alignment notebook's smoke-check cell at the golden-reference flip
(pure logic; deterministic)."""
from cjm_transcript_decomp_core.alignment import (
    assign_words_to_chunks,
    build_segments_from_alignment,
    map_fa_words_to_text,
    sentence_end_word_indices,
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
    end_words = sentence_end_word_indices(
        spans, sent_spans(text, "Hello world.", "Foo bar."))
    assert end_words == {1, 3}
    refined = split_chunks_at_sentence_gaps(chunks, fa, end_words)
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
    end_words = sentence_end_word_indices(
        spans, sent_spans(text, "Hello world.", "Foo bar."))
    # Already one sentence per chunk: the sentence ends fall on chunk-final
    # words, so nothing splits and the skeleton passes through re-indexed.
    refined = split_chunks_at_sentence_gaps(chunks, fa, end_words)
    assert [(c.start_time, c.end_time) for c in refined] == [(0.0, 1.2), (1.9, 3.1)]
    # Textless/montage chunks (no words assigned anywhere) pass through whole.
    assert [(c.start_time, c.end_time)
            for c in split_chunks_at_sentence_gaps(chunks, [], set())] \
        == [(0.0, 1.2), (1.9, 3.1)]


def test_sentence_split_min_duration_guard():
    # The sentence gap sits 0.25s from the chunk end — a cut would mint a
    # sliver, so the guard refuses it and the chunk survives whole.
    text = "Hello world. Hi."
    fa = [FAWord("hello", 0.0, 0.5), FAWord("world", 0.5, 2.6),
          FAWord("hi", 2.7, 2.9)]
    chunks = [VADChunk(0, 0.0, 2.9)]
    spans = map_fa_words_to_text(text, fa)
    end_words = sentence_end_word_indices(
        spans, sent_spans(text, "Hello world.", "Hi."))
    refined = split_chunks_at_sentence_gaps(chunks, fa, end_words, min_chunk_s=0.5)
    assert [(c.start_time, c.end_time) for c in refined] == [(0.0, 2.9)]
    # A permissive guard lets the same cut through.
    loose = split_chunks_at_sentence_gaps(chunks, fa, end_words, min_chunk_s=0.1)
    assert [(round(c.start_time, 2), round(c.end_time, 2)) for c in loose] \
        == [(0.0, 2.65), (2.65, 2.9)]


def test_sentence_split_three_way_cut():
    # Three capability-delivered sentences in one chunk ('dispatch."' / 'Why?' /
    # 'Because.') -> two cuts at the FA gaps after each sentence-ending word.
    text = 'He said "dispatch." Why? Because.'
    fa = [FAWord("he", 0.0, 0.3), FAWord("said", 0.3, 0.6),
          FAWord("dispatch", 0.6, 1.5), FAWord("why", 2.1, 2.8),
          FAWord("because", 3.5, 4.2)]
    chunks = [VADChunk(0, 0.0, 4.4)]
    spans = map_fa_words_to_text(text, fa)
    end_words = sentence_end_word_indices(
        spans, sent_spans(text, 'He said "dispatch."', "Why?", "Because."))
    assert end_words == {2, 3, 4}
    refined = split_chunks_at_sentence_gaps(chunks, fa, end_words)
    assert [(round(c.start_time, 2), round(c.end_time, 2)) for c in refined] \
        == [(0.0, 1.8), (1.8, 3.15), (3.15, 4.4)]


def test_sentence_split_capability_spans_drive_the_decision():
    # (1) The v2-era abbreviation class ('U.S.' mid-sentence) is now the
    # SEGMENTER's problem: one capability span covering the whole text means
    # no end-word lands mid-chunk, so nothing cuts (segmenter correctness for
    # this class is pinned in the cjm-capability-pysbd suite).
    text = "Sitting down with the U.S. Secretary of Energy."
    fa = [FAWord("sitting", 0.0, 0.4), FAWord("down", 0.4, 0.8),
          FAWord("with", 0.8, 1.1), FAWord("the", 1.1, 1.3),
          FAWord("us", 1.3, 2.0), FAWord("secretary", 2.0, 2.9),
          FAWord("of", 2.9, 3.1), FAWord("energy", 3.1, 3.8)]
    chunks = [VADChunk(0, 0.0, 4.0)]
    spans = map_fa_words_to_text(text, fa)
    end_words = sentence_end_word_indices(spans, sent_spans(text, text))
    assert [(c.start_time, c.end_time)
            for c in split_chunks_at_sentence_gaps(chunks, fa, end_words)] \
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
    end_words2 = sentence_end_word_indices(
        spans2, sent_spans(text2, "Stagnant on energy growth.",
                           "Any fundamental reason?"))
    refined = split_chunks_at_sentence_gaps(chunks2, fa2, end_words2)
    assert [(c.start_time, c.end_time) for c in refined] == [(0.0, 1.8), (1.8, 3.5)]
    segs = build_segments_from_alignment(
        text2, spans2, assign_words_to_chunks(fa2, refined), num_chunks=2)
    assert [s.text for s in segs] == ["Stagnant on energy growth.",
                                      "Any fundamental reason?"]


def test_sentence_end_word_indices_merge_walk():
    # The word<->sentence merge walk directly: the 'Mr. Gorbachev' class is one
    # capability span (no mid-sentence end word); a real sentence end right
    # after a title stub still marks its last word.
    text = "He met Dr. Smith. Then he left."
    fa = [FAWord("he", 0.0, 0.3), FAWord("met", 0.3, 0.6),
          FAWord("dr", 0.6, 0.9), FAWord("smith", 0.9, 1.6),
          FAWord("then", 2.4, 2.7), FAWord("he", 2.7, 2.9),
          FAWord("left", 2.9, 3.3)]
    spans = map_fa_words_to_text(text, fa)
    end_words = sentence_end_word_indices(
        spans, sent_spans(text, "He met Dr. Smith.", "Then he left."))
    assert end_words == {3, 6}  # 'Smith.' and 'left.' — never 'Dr.'
    chunks = [VADChunk(0, 0.0, 3.5)]
    assert [(c.start_time, c.end_time)
            for c in split_chunks_at_sentence_gaps(chunks, fa, end_words)] \
        == [(0.0, 2.0), (2.0, 3.5)]
    # A sentence no word starts in (e.g. segmenter output beyond the FA tail)
    # contributes nothing; empty inputs stay empty.
    assert sentence_end_word_indices(spans, [(900, 950)]) == set()
    assert sentence_end_word_indices([], [(0, 10)]) == set()
    assert sentence_end_word_indices(spans, []) == set()


def sent_spans(text, *sentences):
    """Char spans for the given sentence strings within `text` — what the
    segmentation capability delivers for this text (B.5: the tests feed the
    capability's OUTPUT SHAPE; segmenter correctness lives in the
    cjm-capability-pysbd suite)."""
    spans, pos = [], 0
    for s in sentences:
        start = text.index(s, pos)
        spans.append((start, start + len(s)))
        pos = start + len(s)
    return spans
