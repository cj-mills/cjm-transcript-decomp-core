"""Pure forced-alignment logic (no capability calls): map FA words back to character spans in the original text, assign words to VAD chunks by timestamp, and build one text segment per VAD chunk. Extracted from the page-centric ForcedAlignmentService (Tier-1 logic)."""

import re
from typing import Dict, List, Optional, Tuple

from cjm_transcript_decomp_core.models import FAWord, TextSegment, VADChunk

# Strip punctuation for comparison (matches what FA models strip).
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

# Sentence-split policy tag (DEC f1024568): versioned because the policy is a
# SKELETON IDENTITY input — any change to the split rule below must bump it,
# or re-runs would silently collide with spines a different rule produced.
# v2 (2026-07-22 probe-drive findings): dotted abbreviations no longer end
# sentences, and the boundary word lands RIGHT of the cut (half-open fold).
# v3 (same drive, 'Mr. Gorbachev'): honorific/title stubs guarded. The stub
# list is a BRIDGE — the ratified endgame is a sentence-segmentation
# capability replacing this whole token heuristic (see the B.5 work item).
SENTENCE_SPLIT_POLICY = "sentence-split/v3"

# Closing quotes/brackets a sentence-ending token may trail with ('dispatch."').
_SENTENCE_CLOSERS = "\"'”’)]}»"

# Dotted-abbreviation shapes that must NOT end a sentence: letter-dot sequences
# ('U.S.', 'p.m.', 'i.e.') and single initials ('J.').
_ABBREV_RE = re.compile(r"(?:[A-Za-z]\.)+")

# Honorific/title stubs that must NOT end a sentence ('Mr. Gorbachev').
_ABBREV_STUBS = {"mr", "mrs", "ms", "dr", "prof", "rev", "fr", "st", "gen",
                 "col", "capt", "lt", "sgt", "maj", "jr", "sr", "vs"}


def _strip_punct(
    text: str,  # Text to normalize
) -> str:  # Text with punctuation removed
    """Strip punctuation from text for comparison with FA output."""
    return _PUNCT_RE.sub("", text)


def map_fa_words_to_text(
    text: str,             # Original text with punctuation
    fa_items: List[FAWord],  # FA word-level alignment results
) -> List[Tuple[int, int]]:  # (start_char, end_char) spans into the original text
    """Map forced-alignment words back to character spans in the original text.

    Walks the original text, matching each FA word (punctuation-stripped) against
    original-text tokens; returns character offset pairs for each FA word.
    """
    spans = []
    pos = 0  # Current position in original text

    for item in fa_items:
        fa_word = item.text.lower()

        # Skip whitespace to find next token start
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            break

        # Find the end of the current token (run of non-space characters)
        token_start = pos
        token_end = pos
        while token_end < len(text) and not text[token_end].isspace():
            token_end += 1

        orig_token = text[token_start:token_end]
        stripped_token = _strip_punct(orig_token).lower()

        if stripped_token == fa_word:
            spans.append((token_start, token_end))
            pos = token_end
        else:
            # Multi-token FA words (e.g. "p.m." -> "pm") or punctuation-split tokens:
            # consume up to 3 extra tokens until the stripped concatenation matches.
            concat = stripped_token
            scan_end = token_end
            matched = False
            for _ in range(3):
                if concat.lower() == fa_word:
                    spans.append((token_start, scan_end))
                    pos = scan_end
                    matched = True
                    break
                while scan_end < len(text) and text[scan_end].isspace():
                    scan_end += 1
                if scan_end >= len(text):
                    break
                next_start = scan_end
                while scan_end < len(text) and not text[scan_end].isspace():
                    scan_end += 1
                concat += _strip_punct(text[next_start:scan_end])

            if not matched:
                if concat.lower() == fa_word:
                    spans.append((token_start, scan_end))
                    pos = scan_end
                else:
                    # Fallback: take the single token and move on (handles
                    # insertions/deletions between transcript and FA output).
                    spans.append((token_start, token_end))
                    pos = token_end

    return spans


def assign_words_to_chunks(
    fa_items: List[FAWord],     # FA word-level alignment results
    vad_chunks: List[VADChunk],  # VAD chunks with start/end times
) -> List[int]:  # Chunk index for each FA word
    """Assign each FA word to a VAD chunk by timestamp overlap.

    Words whose start_time falls within a chunk's [start, end) are assigned to
    that chunk — HALF-OPEN, so a word starting exactly on a shared boundary
    belongs to the chunk that STARTS there (sentence-split cuts land exactly on
    the next word's start when FA words are contiguous; the old inclusive end
    pulled that word one chunk LEFT — the off-by-one the 2026-07-22 probe drive
    caught). Words in silence gaps (incl. exactly at the last chunk's end) go
    to the nearest chunk by time proximity.
    """
    if not vad_chunks:
        return [0] * len(fa_items)

    assignments = []
    for item in fa_items:
        t = item.start_time
        best_idx = 0
        best_dist = float("inf")
        for i, chunk in enumerate(vad_chunks):
            if chunk.start_time <= t < chunk.end_time:
                best_idx = i
                break
            dist = min(abs(t - chunk.start_time), abs(t - chunk.end_time))
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        assignments.append(best_idx)
    return assignments


def build_segments_from_alignment(
    text: str,                      # Original text with punctuation
    spans: List[Tuple[int, int]],   # Character spans from map_fa_words_to_text
    assignments: List[int],         # Chunk index per word from assign_words_to_chunks
    num_chunks: int,                # Total number of VAD chunks
    source_id: Optional[str] = None,           # Source row id for traceability
    source_provider_id: Optional[str] = None,  # Source provider identifier
) -> List[TextSegment]:  # One segment per VAD chunk
    """Build a TextSegment per VAD chunk by grouping words by chunk assignment.

    Each chunk's text is the original (punctuated) slice from the first to the
    last word assigned to it; chunks with no words become empty segments.
    """
    chunk_spans: Dict[int, List[Tuple[int, int]]] = {}
    for span, chunk_idx in zip(spans, assignments):
        chunk_spans.setdefault(chunk_idx, []).append(span)

    segments = []
    for chunk_idx in range(num_chunks):
        word_spans = chunk_spans.get(chunk_idx, [])
        if word_spans:
            seg_start = word_spans[0][0]
            seg_end = word_spans[-1][1]
            seg_text = text[seg_start:seg_end].strip()
        else:
            seg_text = ""
            seg_start = None
            seg_end = None
        segments.append(TextSegment(
            index=chunk_idx, text=seg_text,
            source_id=source_id, source_provider_id=source_provider_id,
            start_char=seg_start, end_char=seg_end,
        ))
    return segments


def tier1_alignment_checks(
    segments: List[TextSegment],  # Segments produced by build_segments_from_alignment
    vad_chunks: List[VADChunk],   # The VAD chunks they were aligned to
) -> List[str]:  # Human-readable warnings (empty = all clear)
    """Tier-1 deterministic pre-filters for the alignment-review seam (no AI)."""
    warnings: List[str] = []
    if len(segments) != len(vad_chunks):
        warnings.append(
            f"segment/chunk count mismatch: {len(segments)} vs {len(vad_chunks)}"
        )
    empty = sum(1 for s in segments if not s.text.strip())
    if empty:
        warnings.append(
            f"{empty}/{len(segments)} segment(s) have EMPTY text (VAD chunk with no aligned words)"
        )
    return warnings


def _ends_sentence(
    token: str,  # One original-text token (an FA word's char-span slice)
) -> bool:  # True when the token closes a sentence
    """Does a token end a sentence? (v3 rule: trailing closers stripped, then a
    sentence-ending mark, EXCEPT dotted abbreviations — 'U.S.', 'p.m.',
    initials — and honorific/title stubs — 'Mr.', 'Dr.' — both caught splitting
    mid-sentence by the 2026-07-22 probe drives. Remaining false positives
    ('etc.', 'Inc.') stay accepted; refinements version the policy tag rather
    than silently changing committed identity — and the whole token heuristic
    retires when the sentence-segmentation capability lands.)"""
    t = token.rstrip(_SENTENCE_CLOSERS)
    if not t or t[-1] not in ".?!…":
        return False
    core = t.lstrip("([{\"'“‘«")
    if _ABBREV_RE.fullmatch(core):
        return False
    return core[:-1].lower() not in _ABBREV_STUBS


def split_chunks_at_sentence_gaps(
    vad_chunks: List[VADChunk],    # The VAD skeleton (segment-local times)
    fa_items: List[FAWord],        # The AUTHORITATIVE transcriber's FA words (segment-local times)
    spans: List[Tuple[int, int]],  # Char spans from map_fa_words_to_text (parallel to fa_items)
    text: str,                     # The authoritative original (punctuated) text
    min_chunk_s: float = 0.5,      # Min sub-chunk duration — a split never mints a sliver
) -> List[VADChunk]:  # The refined skeleton, re-indexed (identical content when nothing splits)
    """The sentence-split stage (SENTENCE_SPLIT_POLICY, DEC f1024568): refine the
    VAD skeleton by cutting any chunk whose assigned text crosses a sentence end.

    Runs POST-FA, PRE-fold: a chunk holding a sentence-ending word that is not
    its last word splits at the corresponding FA word gap (midpoint between the
    ending word's end and the next word's start — the pause the VAD's min-sil
    threshold failed to cut, finding bc69e3e6). The split decision reads ONLY
    the authoritative transcriber (montage/textless chunks have no words here
    and pass through untouched); every transcriber then re-folds over the
    refined skeleton, so variants stay per-chunk consistent by construction.
    Both sides of an accepted cut must be >= `min_chunk_s` at accept time
    (greedy left-to-right), so FA jitter cannot mint unplayable slivers.
    """
    assignments = assign_words_to_chunks(fa_items, vad_chunks)
    by_chunk: Dict[int, List[int]] = {}
    for wi, ci in enumerate(assignments):
        by_chunk.setdefault(ci, []).append(wi)

    refined: List[VADChunk] = []
    for chunk in vad_chunks:
        words = by_chunk.get(chunk.index, [])
        cuts: List[float] = []
        cur_start = chunk.start_time
        for p in range(len(words) - 1):
            wi, wj = words[p], words[p + 1]
            if wi >= len(spans):
                break  # map_fa_words_to_text ran out of text — no span, no verdict
            token = text[spans[wi][0]:spans[wi][1]]
            if not _ends_sentence(token):
                continue
            cut = (fa_items[wi].end_time + fa_items[wj].start_time) / 2.0
            if not (chunk.start_time < cut < chunk.end_time):
                continue
            if cut - cur_start < min_chunk_s or chunk.end_time - cut < min_chunk_s:
                continue
            cuts.append(cut)
            cur_start = cut
        bounds = [chunk.start_time] + cuts + [chunk.end_time]
        for k in range(len(bounds) - 1):
            refined.append(VADChunk(index=0, start_time=bounds[k], end_time=bounds[k + 1]))
    for i, c in enumerate(refined):
        c.index = i
    return refined
