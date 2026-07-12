"""Pure forced-alignment logic (no capability calls): map FA words back to character spans in the original text, assign words to VAD chunks by timestamp, and build one text segment per VAD chunk. Extracted from the page-centric ForcedAlignmentService (Tier-1 logic)."""

import re
from typing import Dict, List, Optional, Tuple

from cjm_transcript_decomp_core.models import FAWord, TextSegment, VADChunk

# Strip punctuation for comparison (matches what FA models strip).
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


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

    Words whose start_time falls within a chunk's [start, end] are assigned to
    that chunk; words in silence gaps go to the nearest chunk by time proximity.
    """
    if not vad_chunks:
        return [0] * len(fa_items)

    assignments = []
    for item in fa_items:
        t = item.start_time
        best_idx = 0
        best_dist = float("inf")
        for i, chunk in enumerate(vad_chunks):
            if chunk.start_time <= t <= chunk.end_time:
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
