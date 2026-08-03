"""Pure forced-alignment logic (no capability calls): map FA words back to character spans in the original text, assign words to VAD chunks by timestamp, and build one text segment per VAD chunk. Extracted from the page-centric ForcedAlignmentService (Tier-1 logic)."""

import re
from typing import Dict, List, Optional, Set, Tuple

from cjm_transcript_decomp_core.models import FAWord, TextSegment, VADChunk

# Strip punctuation for comparison (matches what FA models strip).
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

# Sentence-split policy tag (DEC f1024568): versioned because the policy is a
# SKELETON IDENTITY input — any change to the split rule must bump it, or
# re-runs would silently collide with spines a different rule produced.
# 'capability' (B.5, DEC cc904eee): sentence boundaries now come from a
# sentence-segmentation CAPABILITY (pySBD first) as char spans over the
# authoritative text — the v1-v3 token heuristic (closer/abbreviation/stub
# lists) is RETIRED. The segmenter's identity (capability name + config hash)
# joins the skeleton-identity composite beside this tag.
SENTENCE_SPLIT_POLICY = "sentence-split/capability"

# Event-carve policy tag (respine trial DEC 6cc10fb7): versioned for the same
# reason — it is a SKELETON IDENTITY input. 'v1' = cut-don't-label: model event
# spans (a ProposalSetManifest consumed by pointer) become GAPS between chunks;
# sliver pieces under the min-chunk guard are absorbed into the gap.
EVENT_SPLIT_POLICY = "event-carve/v1"

# Word-rescue policy tag (96edc646 verdict bc7ece7b): a SKELETON IDENTITY
# input like the other stage tags. 'v2' = FA-word authority over chunk
# coverage: an authoritative-transcriber word whose START no chunk contains
# (the fold's own assignment test) seeds rescue from its UNCOVERED PIECES —
# the word interval minus chunks ∪ event spans — so an edge-straddling word
# rescues its poke-out while a word FULLY inside a verified event span stays
# unrescued (true FA drift). Pieces group per free interval, padded, clipped;
# the min-chunk guard does NOT apply — rescue exists precisely to keep speech
# the guard or VAD dropped. ('v1' seeded on word midpoints and skipped any
# word whose midpoint fell inside an event span — it stranded edge-straddlers;
# one v1 spine exists and is superseded.) Pad/join/min-piece are
# version-bound: changing them bumps the tag.
# 'v4' makes the FOLD pure argmax-overlap (assign_words_to_chunks): a word
# homes into the chunk holding the most of its audio; start-containment and
# nearest-edge survive only as the zero-overlap fallback. Overlap subsumes
# half-open containment for the boundary case, and kills the last mis-homing
# channel: a word start-captured by a whisker of the WRONG chunk while its
# body sits in a rescued poke-out. (v3 preferred overlap only for uncontained
# words; v2 seeded rescue from uncovered pieces but kept nearest-edge
# assignment; one spine of each exists, both superseded.)
# 'v5' narrows the SEED rule to HOMELESS words only (max overlap with every
# existing chunk < min-piece): a boundary-clipped word is already homed by
# the argmax-overlap fold, and minting its poke-out littered v4 spines with
# hundreds of tiny empty sliver segments. Fold unchanged from v4.
WORD_RESCUE_POLICY = "word-rescue/v5"
WORD_RESCUE_PAD_S = 0.05        # Minted-chunk padding around each rescued piece group (seconds)
WORD_RESCUE_JOIN_GAP_S = 0.5    # Pieces closer than this join one rescued chunk (seconds)
WORD_RESCUE_MIN_PIECE_S = 0.03  # Uncovered piece shorter than this = FA jitter, not speech


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

    A word homes into the chunk holding the MOST of its audio (word-rescue/v4
    fold rule, 96edc646 verdict bc7ece7b): argmax overlap over [start, end] ∩
    chunk. Overlap subsumes the old half-open start-containment — a word
    starting exactly on a shared boundary lies fully in the chunk that STARTS
    there (the sentence-split contiguity case the 2026-07-22 probe drive
    pinned), and a word whose FA start clips the END of the wrong chunk by a
    whisker no longer gets start-captured there while its body sits in the
    next chunk or a rescued poke-out. Ties keep the earlier chunk. Only a
    word overlapping nothing at all (fully inside a carved event span, or
    zero-duration) falls back — containment of its start first, then
    nearest-edge proximity.
    """
    if not vad_chunks:
        return [0] * len(fa_items)

    assignments = []
    for item in fa_items:
        t = item.start_time
        best_idx = 0
        best_overlap = 0.0
        for i, chunk in enumerate(vad_chunks):
            overlap = min(item.end_time, chunk.end_time) - max(t, chunk.start_time)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i
        if best_overlap <= 0.0:
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


def sentence_end_word_indices(
    spans: List[Tuple[int, int]],           # FA-word char spans from map_fa_words_to_text (ordered)
    sentence_spans: List[Tuple[int, int]],  # Sentence char spans over the SAME text (ordered, non-overlapping)
) -> Set[int]:  # Indices of FA words that END a sentence
    """Map capability-delivered sentence boundaries onto FA words (B.5: the
    successor of the v1-v3 `_ends_sentence` token heuristic).

    A word ends a sentence when it is the LAST word whose span STARTS before
    that sentence's end_char. Both lists are ordered over the same text, so a
    single forward merge walk suffices; a sentence no word starts in
    contributes nothing, and the trailing word of the text marks its sentence
    but can never produce a cut (cuts live BETWEEN consecutive words)."""
    out: Set[int] = set()
    wi = 0
    for ss, se in sentence_spans:
        last = None
        while wi < len(spans) and spans[wi][0] < se:
            if spans[wi][0] >= ss:  # the word must START inside the sentence
                last = wi
            wi += 1
        if last is not None:
            out.add(last)
    return out


def split_chunks_at_sentence_gaps(
    vad_chunks: List[VADChunk],  # The VAD skeleton (segment-local times)
    fa_items: List[FAWord],      # The AUTHORITATIVE transcriber's FA words (segment-local times)
    end_words: Set[int],         # FA-word indices that END a sentence (sentence_end_word_indices)
    min_chunk_s: float = 0.5,    # Min sub-chunk duration — a split never mints a sliver
) -> List[VADChunk]:  # The refined skeleton, re-indexed (identical content when nothing splits)
    """The sentence-split stage (SENTENCE_SPLIT_POLICY, DEC f1024568): refine the
    VAD skeleton by cutting any chunk whose assigned text crosses a sentence end.

    Runs POST-FA, PRE-fold: a chunk holding a sentence-ending word that is not
    its last word splits at the corresponding FA word gap (midpoint between the
    ending word's end and the next word's start — the pause the VAD's min-sil
    threshold failed to cut, finding bc69e3e6). Sentence ends arrive
    PRECOMPUTED (B.5): the segmentation capability's char spans over the
    authoritative text, mapped onto FA words by `sentence_end_word_indices` —
    this function no longer inspects text. The split decision reads ONLY the
    authoritative transcriber (montage/textless chunks have no words here and
    pass through untouched); every transcriber then re-folds over the refined
    skeleton, so variants stay per-chunk consistent by construction. Both
    sides of an accepted cut must be >= `min_chunk_s` at accept time (greedy
    left-to-right), so FA jitter cannot mint unplayable slivers.
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
            if wi not in end_words:
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


def carve_chunks_at_event_spans(
    vad_chunks: List[VADChunk],            # The VAD skeleton (segment-local times)
    event_spans: List[Tuple[float, float]],  # Model event spans (segment-local times, ordered)
    min_chunk_s: float = 0.5,              # Sliver guard — a piece shorter than this absorbs into the gap
) -> List[VADChunk]:  # The carved skeleton, re-indexed (identical content when nothing overlaps)
    """The event-carve stage (EVENT_SPLIT_POLICY, respine trial DEC 6cc10fb7):
    cut-don't-label — every model event span becomes a GAP between chunks.

    Runs on the VAD skeleton post-FA pre-fold, the same seat as
    `split_chunks_at_sentence_gaps` but with ACOUSTIC authority: each event
    span clipped to a chunk removes that interval from the chunk, so the
    propose lane's later accepts stay pure gap-inserts into exactly these
    gaps. A resulting piece shorter than `min_chunk_s` is ABSORBED into the
    adjacent gap rather than kept (a rejected cut would re-embed the event
    mid-chunk and resurrect the nudge tax; a VAD-edge sliver beside a detected
    event is padding). A chunk fully covered by an event drops whole. Word
    fidelity is safe by construction: `assign_words_to_chunks` sends words in
    gaps to the nearest surviving chunk, so FA times only apportion words
    across a cut — no text is lost."""
    carved: List[VADChunk] = []
    for chunk in vad_chunks:
        overlapping = [(max(chunk.start_time, s), min(chunk.end_time, e))
                       for s, e in event_spans
                       if s < chunk.end_time and e > chunk.start_time]
        pieces: List[Tuple[float, float]] = []
        cursor = chunk.start_time
        for s, e in sorted(overlapping):
            if s > cursor:
                pieces.append((cursor, s))
            cursor = max(cursor, e)
        if cursor < chunk.end_time:
            pieces.append((cursor, chunk.end_time))
        for ps, pe in pieces:
            if pe - ps >= min_chunk_s or not overlapping:
                carved.append(VADChunk(index=0, start_time=ps, end_time=pe))
    for i, c in enumerate(carved):
        c.index = i
    return carved


def rescue_gap_words(
    vad_chunks: List[VADChunk],              # The chunk skeleton post split/carve (segment-local times)
    fa_words: List[FAWord],                  # AUTHORITATIVE transcriber's FA words (segment-local times)
    event_spans: Optional[List[Tuple[float, float]]] = None,  # Carved event spans (segment-local; rescued chunks never overlap them)
    pad_s: float = WORD_RESCUE_PAD_S,        # Padding around each rescued word group
    join_gap_s: float = WORD_RESCUE_JOIN_GAP_S,  # Words closer than this share one rescued chunk
) -> List[VADChunk]:  # The skeleton + minted rescue chunks, merged, re-indexed
    """The word-rescue stage (WORD_RESCUE_POLICY, 96edc646 verdict bc7ece7b):
    chunks derive from VAD ∪ FA-word-coverage, not VAD alone.

    Two mechanisms strand real speech outside every chunk — the carve's sliver
    guard absorbs sub-min_chunk_s pieces (M1), and VAD misses short/soft
    speech in inter-chunk gaps entirely (M2). `assign_words_to_chunks` then
    sends those words to the NEAREST chunk: text survives but mis-homed into a
    chunk whose audio does not contain it. This stage closes both at one seat
    (v2 semantics — see WORD_RESCUE_POLICY): a word the fold cannot home (no
    chunk contains its start) seeds rescue from its UNCOVERED PIECES — the
    word interval minus chunks ∪ event spans — so an edge-straddling word
    rescues its poke-out while a word FULLY inside a human-verified event span
    contributes nothing (true FA drift). Piece groups become minted chunks —
    padded, clipped so rescued chunks never overlap existing chunks or event
    spans. The min-chunk guard deliberately does NOT apply: rescue exists to
    keep exactly the speech the guard or VAD dropped. Runs post-carve pre-fold
    on the authoritative transcriber's words only (words = text authority;
    every transcriber re-folds over the same rescued skeleton)."""
    # Free intervals = the timeline complement of chunks ∪ event spans.
    blocks = sorted([(c.start_time, c.end_time) for c in vad_chunks]
                    + [(s, e) for s, e in (event_spans or [])])
    merged: List[Tuple[float, float]] = []
    for s, e in blocks:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    def free_interval(t: float) -> Optional[Tuple[float, float]]:
        lo, hi = float("-inf"), float("inf")
        for s, e in merged:
            if s <= t <= e:
                return None
            if e < t:
                lo = max(lo, e)
            elif s > t:
                hi = min(hi, s)
                break
        return (lo, hi)

    # v2 seed rule: a word the fold cannot home (no chunk contains its START,
    # the fold's own half-open test) contributes its UNCOVERED PIECES — the
    # word interval minus every block. An edge-straddling word rescues its
    # poke-out; a word fully inside a verified event span contributes nothing
    # (true FA drift). Sub-jitter pieces are noise, not speech.
    pieces: List[Tuple[float, float]] = []
    for w in sorted(fa_words, key=lambda w: w.start_time):
        if w.end_time <= w.start_time:
            continue
        # v5 seed rule: only HOMELESS words rescue — a word with real overlap
        # in some existing chunk is already homed by the argmax-overlap fold;
        # minting its poke-out would litter the spine with empty slivers
        # (v4 minted 360 tiny empty segments on source-1 doing exactly that).
        if max((min(w.end_time, c.end_time) - max(w.start_time, c.start_time)
                for c in vad_chunks), default=0.0) >= WORD_RESCUE_MIN_PIECE_S:
            continue
        cursor = w.start_time
        for s, e in merged:
            if e <= cursor:
                continue
            if s >= w.end_time:
                break
            if s > cursor:
                pieces.append((cursor, min(s, w.end_time)))
            cursor = max(cursor, e)
        if cursor < w.end_time:
            pieces.append((cursor, w.end_time))
    pieces = sorted(p for p in pieces if p[1] - p[0] >= WORD_RESCUE_MIN_PIECE_S)

    rescued: List[VADChunk] = []
    last_fi: Optional[Tuple[float, float]] = None
    for ps, pe in pieces:
        fi = free_interval((ps + pe) / 2)
        if fi is None:  # Defensive: a piece is inside a free interval by construction
            continue
        if (rescued and fi == last_fi
                and ps - rescued[-1].end_time <= join_gap_s + pad_s):
            rescued[-1].end_time = max(rescued[-1].end_time, min(pe + pad_s, fi[1]))
        else:
            # Clamp at 0: a piece at the pipeline-segment head minus pad must
            # not mint a negative-start chunk (TimeSlice refuses it).
            rescued.append(VADChunk(index=0,
                                    start_time=max(ps - pad_s, fi[0], 0.0),
                                    end_time=min(pe + pad_s, fi[1])))
        last_fi = fi
    out = list(vad_chunks) + [c for c in rescued if c.end_time > c.start_time]
    out.sort(key=lambda c: c.start_time)
    for i, c in enumerate(out):
        c.index = i
    return out
