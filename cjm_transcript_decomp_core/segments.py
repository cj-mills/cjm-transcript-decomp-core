"""Fine-segment inspection for the decomp TUI (work item 166dd2b8, half a):
read one decomp source's committed Segment nodes back from the graph the run
RECORDED writing to (e087d059 provenance reused, read-only seat), and derive
the timestamp GAPS the listing must surface — a chunk VAD never cut has no row
anywhere downstream (finding de994164), so the gap between adjacent committed
segments is the only place a miss is visible. Pure logic below the stack seam
(runs.py precedent: everything under the paint path tests directly)."""

import json
from bisect import bisect_right
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cjm_context_graph_layer.grammar import OverlayRelations, SpineRelations
from cjm_context_graph_layer.ops import graph_task
from cjm_context_graph_primitives.query import NodeQuery, OrderBy, RelationPredicate
from cjm_substrate.core.manager import CapabilityManager
from cjm_substrate.core.queue import JobQueue
from cjm_transcript_decomp_core.alignment import (assign_words_to_chunks,
                                                  build_segments_from_alignment,
                                                  map_fa_words_to_text, sentence_end_word_indices,
                                                  split_chunks_at_sentence_gaps)
from cjm_transcript_decomp_core.cli import load_capabilities
from cjm_transcript_decomp_core.models import FAWord, VADChunk
from cjm_transcript_decomp_core.pipeline import submit_and_wait
from cjm_transcript_graph_schema.schema import TranscriptGraphLabels


class SegmentStack:
    """One lazily-opened READ-ONLY graph seat, keyed by db path.

    The batch app deliberately loads no capability; the segments drill is the
    exception. The seat opens the graph capability with readonly config — the
    inspection view must not be able to write the spine it inspects — against
    the db the decomp run RECORDED (e087d059: never the stack's own default).
    One seat at a time: drilling a run recorded against a DIFFERENT db tears
    the old seat down first (the runs dir is shared ground and can span dbs)."""

    BATCH = 500  # ids per query — bounded reads; 950-segment sources are real

    def __init__(self, manifests_dir: str,   # Capability manifests directory
                 graph_capability: str):     # Graph capability instance id
        self.manifests_dir = manifests_dir
        self.graph_capability = graph_capability
        self.db_path: Optional[str] = None
        self._manager: Optional[CapabilityManager] = None
        self._queue: Optional[JobQueue] = None
        self._probe_loaded: Dict[str, Any] = {}  # probe capability -> loaded config key (VAD/FA)

    async def open(self, db_path: str) -> None:
        """Open (or re-point) the read-only seat at a recorded db path."""
        if self._queue is not None and self.db_path == db_path:
            return
        await self.close()
        manager = CapabilityManager(search_paths=[Path(self.manifests_dir)])
        load_capabilities(manager, [self.graph_capability],
                          configs={self.graph_capability:
                                   {"db_path": db_path, "readonly": True}})
        queue = JobQueue(deps=manager)
        await queue.start()
        self._manager, self._queue, self.db_path = manager, queue, db_path

    async def read_segments(self, segment_ids: List[str]) -> List[Dict[str, Any]]:
        """Projected rows for the manifest's segment_ids, in ID-LIST order.

        The decomp manifest records segment_ids in spine order; the query
        backend does not promise row order, so the join re-imposes it
        (ordered_rows). Ids the graph no longer holds are dropped — the caller
        counts the shortfall and says so, never invents rows."""
        if self._queue is None:
            raise RuntimeError("segment seat not open — call open(db_path) first")
        by_id: Dict[str, Dict[str, Any]] = {}
        for i in range(0, len(segment_ids), self.BATCH):
            q = NodeQuery(ids=list(segment_ids[i:i + self.BATCH]),
                          project=["index", "start_time", "end_time", "text",
                                   "rendition_id"])
            res = await graph_task(self._queue, self.graph_capability,
                                   "query_nodes", query=q.to_dict())
            for r in (res.rows or []):
                by_id[str(r["id"])] = r
        return ordered_rows(by_id, segment_ids)

    async def read_audio_join(self, source_id: str,
                              rendition_ids: List[str]) -> List[Dict[str, Any]]:
        """Coarse-WAV join for playback + probes: ordered AudioSegment spans,
        each with its model-input WAV under THIS spine's renditions.

        The fine rows' own rendition_id set disambiguates raw vs vocals — no
        chain resolution needed (the spine names what it hangs under). Mirrors
        the correction TUI's ChunkRef join (more 7751243f kit pressure)."""
        if self._queue is None:
            raise RuntimeError("segment seat not open — call open(db_path) first")
        aq = NodeQuery(label=TranscriptGraphLabels.AUDIO_SEGMENT,
                       related=RelationPredicate(SpineRelations.PART_OF,
                                                 node_id=source_id),
                       order_by=OrderBy(prop="start"), project=["start", "end"])
        ares = await graph_task(self._queue, self.graph_capability,
                                "query_nodes", query=aq.to_dict())
        asegs = [(str(r["id"]), float(r.get("start") or 0.0),
                  float(r.get("end") or 0.0)) for r in (ares.rows or [])]
        if not asegs:
            return []
        rq = NodeQuery(label=TranscriptGraphLabels.AUDIO_RENDITION,
                       related=RelationPredicate(OverlayRelations.DERIVED_FROM,
                                                 node_ids=[a[0] for a in asegs]),
                       project=["model_input_path", "audio_segment_id"])
        rres = await graph_task(self._queue, self.graph_capability,
                                "query_nodes", query=rq.to_dict())
        rend_ids = set(rendition_ids)
        wav_by_aseg: Dict[str, str] = {}
        rend_by_aseg: Dict[str, str] = {}
        for r in (rres.rows or []):
            if r["id"] in rend_ids:
                aid = str(r.get("audio_segment_id"))
                wav_by_aseg[aid] = str(r.get("model_input_path") or "")
                rend_by_aseg[aid] = str(r["id"])
        return [{"start": s, "end": e, "wav": wav_by_aseg.get(aid) or None,
                 "rendition": rend_by_aseg.get(aid)}
                for aid, s, e in asegs]

    async def probe_vad(self, vad_capability: str, config: Dict[str, Any],
                        wav_path: str) -> List[Tuple[float, float]]:
        """Run a VAD config over one coarse WAV (PROBE-ONLY — nothing commits,
        per ee4a4b9c; a promising sweep argues an explicit re-decomposition).

        Reloads the VAD instance when the config changed (silero is light);
        the adapter's cache keys on (audio, config), so re-probing a config
        already run — including the committed run's own — returns from cache."""
        if self._manager is None or self._queue is None:
            raise RuntimeError("segment seat not open — call open(db_path) first")
        self._ensure_probe(vad_capability, config)
        result = await submit_and_wait(self._queue, vad_capability,
                                       audio=wav_path, task="vad",
                                       method="detect_speech",
                                       control={"force": False})
        return sorted((float(r.start), float(r.end)) for r in result.ranges)

    def _ensure_probe(self, capability: str, config: Dict[str, Any]) -> None:
        """Load (or config-swap) one probe capability instance (VAD/FA share this)."""
        key = tuple(sorted((k, str(v)) for k, v in config.items()))
        if self._probe_loaded.get(capability) == key:
            return
        if capability in self._probe_loaded:
            try:
                self._manager.unload_capability(capability)
            except Exception:
                pass
        load_capabilities(self._manager, [capability],
                          configs={capability: dict(config)})
        self._probe_loaded[capability] = key

    async def probe_fa(self, fa_capability: str, config: Dict[str, Any],
                       wav_path: str, text: str) -> List[Tuple[str, float, float]]:
        """Force-align a transcript over one coarse WAV (PROBE-ONLY).

        Returns (word, start_s, end_s), WAV-local — realign_rows' feed. Same
        cache economics as probe_vad: the adapter keys on (audio, text,
        config), so re-probing an already-aligned pair returns from cache."""
        if self._manager is None or self._queue is None:
            raise RuntimeError("segment seat not open — call open(db_path) first")
        self._ensure_probe(fa_capability, config)
        result = await submit_and_wait(self._queue, fa_capability,
                                       audio=wav_path, text=text,
                                       task="forced_alignment", method="align",
                                       control={"force": False})
        return [(str(it.text), float(it.start_time), float(it.end_time))
                for it in result.items]

    async def probe_segment(self, seg_capability: str, config: Dict[str, Any],
                            text: str) -> List[Tuple[int, int]]:
        """Sentence-segment a transcript text (PROBE-ONLY; B.5 probe parity).

        Loads the segmentation capability exactly like VAD/FA (_ensure_probe),
        so the s split-preview shows CAPABILITY-segmented cuts pre-commit —
        the same spans a --sentence-split re-decomposition would use. Cheap
        rule-based CPU (no adapter cache to warm)."""
        if self._manager is None or self._queue is None:
            raise RuntimeError("segment seat not open — call open(db_path) first")
        self._ensure_probe(seg_capability, config)
        result = await submit_and_wait(self._queue, seg_capability, text=text,
                                       task="sentence_segmentation",
                                       method="segment_text")
        return [(int(s.start_char), int(s.end_char)) for s in result.spans]

    async def read_transcript_text(self, rendition_id: str,
                                   transcriber: str) -> Optional[str]:
        """The coarse Transcript text for one rendition + transcriber.

        The text is stored ONCE at the coarse layer (fine Segments slice into
        it) — the probe realigns THIS, exactly what a re-decomposition would
        consume, never the fine rows' already-sliced approximations."""
        if self._queue is None:
            raise RuntimeError("segment seat not open — call open(db_path) first")
        tq = NodeQuery(label=TranscriptGraphLabels.TRANSCRIPT,
                       related=RelationPredicate(OverlayRelations.DERIVED_FROM,
                                                 node_id=rendition_id),
                       project=["transcriber", "text"])
        res = await graph_task(self._queue, self.graph_capability,
                               "query_nodes", query=tq.to_dict())
        for r in (res.rows or []):
            if str(r.get("transcriber")) == transcriber:
                return str(r.get("text") or "")
        return None

    async def close(self) -> None:
        """Tear down the queue + capability (idempotent)."""
        if self._queue is not None:
            await self._queue.stop()
            self._queue = None
        if self._manager is not None:
            for cap in tuple(self._probe_loaded) + (self.graph_capability,):
                try:
                    self._manager.unload_capability(cap)
                except Exception:
                    pass
            self._manager = None
        self._probe_loaded = {}
        self.db_path = None


def ordered_rows(
    by_id: Dict[str, Dict[str, Any]],  # Fetched rows keyed by node id
    segment_ids: List[str],            # The manifest's spine-order id list
) -> List[Dict[str, Any]]:  # Rows re-imposed into spine order (missing ids dropped)
    """Re-impose the manifest's spine order on fetched rows (pure)."""
    return [by_id[sid] for sid in segment_ids if sid in by_id]


def find_gaps(
    rows: List[Dict[str, Any]],  # Spine-ordered segment rows (start_time/end_time)
    threshold_s: float,          # Minimum uncovered span worth surfacing, seconds
) -> List[Tuple[int, float]]:  # (row index the gap PRECEDES, gap seconds)
    """Uncovered timestamp spans between committed segments (pure).

    Index 0 = a LEADING gap from t=0 — the de994164 miss opened the episode,
    so the leading case is the headline, not an edge case. Coverage advances
    monotonically (max of end times seen) so overlapping chunk timings never
    mint a negative gap. Trailing coverage is unknowable here: the manifest
    does not carry source duration, so a tail miss stays invisible until a
    duration lands on the graph."""
    out: List[Tuple[int, float]] = []
    covered = 0.0
    for i, r in enumerate(rows):
        start, end = r.get("start_time"), r.get("end_time")
        if start is None:
            continue
        gap = float(start) - covered
        if gap >= threshold_s:
            out.append((i, gap))
        if end is not None:
            covered = max(covered, float(end))
    return out


def build_display(
    rows: List[Dict[str, Any]],  # Spine-ordered segment rows
    threshold_s: float,          # Gap threshold, seconds
) -> List[Tuple[str, int, float]]:  # ("gap"|"seg", row index, gap seconds)
    """Interleave gap markers into the paintable entry list (pure).

    A ("gap", i, secs) entry precedes ("seg", i, 0.0). Gap rows are FOCUSABLE
    entries, not decoration — the gap IS the inspection target (de994164): its
    detail names the uncovered span to check against the source audio."""
    gaps = dict(find_gaps(rows, threshold_s))
    out: List[Tuple[str, int, float]] = []
    for i in range(len(rows)):
        if i in gaps:
            out.append(("gap", i, gaps[i]))
        out.append(("seg", i, 0.0))
    return out


def fmt_ts(seconds: float) -> str:  # m:ss.s, h:mm:ss.s from the hour mark
    """Source-coordinate timestamp for listing rows (pure)."""
    s = max(0.0, float(seconds))
    h, rem = divmod(s, 3600.0)
    m, sec = divmod(rem, 60.0)
    if h >= 1:
        return f"{int(h)}:{int(m):02d}:{sec:04.1f}"
    return f"{int(m)}:{sec:04.1f}"


def vad_summary(manifest: Dict[str, Any]) -> Optional[str]:  # One-line VAD knob summary, or None
    """The decomp run's recorded VAD config, one status line (pure).

    Read from the manifest's capabilities block (the same provenance the db
    path rides) — the 166dd2b8 config-axis rung: WHICH knobs cut this spine
    must be visible where the spine is inspected, because a miss (de994164)
    is only interpretable against the threshold that produced it."""
    vad_id = str((manifest.get("config") or {}).get("vad_capability") or "")
    cap = (manifest.get("capabilities") or {}).get(vad_id) or {}
    cfg = cap.get("config") or {}
    if not cfg:
        return None
    parts: List[str] = []
    if cfg.get("threshold") is not None:
        parts.append(f"thr {cfg['threshold']}")
    for key, tag in (("min_speech_duration_ms", "min-speech"),
                     ("min_silence_duration_ms", "min-sil"),
                     ("speech_pad_ms", "pad")):
        if cfg.get(key) is not None:
            parts.append(f"{tag} {cfg[key]}ms")
    name = vad_id.removeprefix("cjm-capability-") or "vad"
    return (name + ": " + " · ".join(parts)) if parts else name


def aseg_index_for(
    asegs: List[Dict[str, Any]],  # read_audio_join rows (start-ordered)
    t: float,                     # A source-coordinate time (seconds)
) -> Optional[int]:  # Index of the AudioSegment whose span holds t (clamped); None = empty
    """Which coarse AudioSegment a source-coordinate time falls in (pure;
    bisect_right over the ordered starts — the correction TUI convention)."""
    if not asegs:
        return None
    starts = [a["start"] for a in asegs]
    return max(0, bisect_right(starts, t) - 1)


def locate_span(
    asegs: List[Dict[str, Any]],  # read_audio_join rows ({"start","end","wav"}, ordered)
    span_start: float,            # Source-coordinate span start (seconds)
    span_end: float,              # Source-coordinate span end (seconds)
) -> Optional[Tuple[str, float, float]]:  # (wav_path, local_start, local_end); None = no WAV covers it
    """Resolve a source-coordinate span onto its owning coarse WAV (pure).

    The span clamps to the WAV containing its START — a gap crossing a coarse
    boundary auditions its head, the tail is one j-press away on the next
    entry. Times go LOCAL by subtracting the AudioSegment start (the
    correction TUI's ChunkRef convention). Gap audio lives ONLY here: the
    fine chunks never cut it, but the coarse model-input WAV spans it."""
    i = aseg_index_for(asegs, span_start)
    if i is None:
        return None
    a = asegs[i]
    if not a.get("wav") or span_start >= a["end"]:
        return None
    return (a["wav"], span_start - a["start"],
            min(span_end, a["end"]) - a["start"])


def probe_compare(
    committed: List[Tuple[float, float]],  # In-aseg committed fine spans (source coords)
    predicted: List[Tuple[float, float]],  # Probe-predicted chunk spans (source coords)
    gap_spans: List[Tuple[float, float]],  # Currently-uncovered spans in the aseg (source coords)
) -> Dict[str, Any]:  # {"committed": n, "predicted": m, "recovered": [(s, e), ...]}
    """Compare a VAD probe against the committed skeleton (pure).

    'Recovered' = predicted chunks intersecting a currently-uncovered span —
    the de994164 question made computable: WOULD this config have cut the
    missed audio? Even an empty answer is a data point (166dd2b8)."""
    recovered = [(s, e) for s, e in predicted
                 if any(s < ge and e > gs for gs, ge in gap_spans)]
    return {"committed": len(committed), "predicted": len(predicted),
            "recovered": recovered}


def capability_config_schema(
    manifests_dir: str,  # Capability manifests directory
    capability: str,     # Capability instance id (the manifest code name)
) -> Optional[Dict[str, Any]]:  # Its config_schema section, or None
    """A capability's config_schema off its installed manifest (json read, pure).

    Matched by code.name, not file stem (the discovery.py convention);
    unreadable/foreign files skip rather than fail. Feeds ConfigForm
    .from_schema — the f4a9d253 keystone applied to the VAD probe axis."""
    for f in sorted(Path(manifests_dir).glob("*.json")):
        try:
            manifest = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        code = manifest.get("code") if isinstance(manifest, dict) else None
        if isinstance(code, dict) and code.get("name") == capability:
            return code.get("config_schema")
    return None


def predicted_rows(
    predicted: List[Tuple[float, float]],  # Predicted chunk spans (source coords)
    committed_rows: List[Dict[str, Any]],  # Committed segment rows (the borrow-text carriers)
    gap_spans: List[Tuple[float, float]],  # Currently-uncovered spans (source coords)
    realigned: Optional[List[str]] = None,  # FA-realigned text per chunk (realign_rows); None = borrow
) -> List[Dict[str, Any]]:  # Segment-shaped rows for the predicted skeleton
    """Synthesize segment-shaped rows for a probe's predicted skeleton (pure).

    With `realigned`, each chunk carries the text a re-decomposition would
    ACTUALLY commit (the pipeline's own FA fold, run probe-side — drive
    feedback 2026-07-22: borrowed text repeated a split chunk's whole parent,
    unjudgeable). Without it, each chunk borrows the text of the committed
    segments it overlaps — orientation only, the probe transcribed nothing.
    Chunks intersecting a currently-uncovered span are flagged "recovered"
    (the de994164 signal made walkable, not just countable). The rows feed
    build_display, so the predicted skeleton paints EXACTLY like the
    committed one — including the gap rows it would still leave."""
    out: List[Dict[str, Any]] = []
    for i, (s, e) in enumerate(predicted):
        if realigned is not None:
            text = realigned[i] if i < len(realigned) else ""
        else:
            texts = [str(r.get("text") or "") for r in committed_rows
                     if r.get("start_time") is not None
                     and r.get("end_time") is not None
                     and float(r["start_time"]) < e and float(r["end_time"]) > s]
            text = " ".join(t for t in texts if t)
        out.append({"index": i, "start_time": s, "end_time": e, "text": text,
                    "realigned": realigned is not None,
                    "recovered": any(s < ge and e > gs for gs, ge in gap_spans)})
    return out


def realign_rows(
    coarse_text: str,                          # The transcriber's full text for the coarse chunk's rendition
    fa_words: List[Tuple[str, float, float]],  # (word, start_s, end_s) — WAV-local FA output (probe_fa)
    predicted_local: List[Tuple[float, float]],  # Predicted chunk spans, WAV-local
) -> List[str]:  # Realigned text per predicted chunk ("" = no words assigned)
    """Re-run the decomp pipeline's own text fold over a PROBE skeleton (pure).

    Reuses the core alignment verbatim — map FA words to char spans in the
    punctuated text, assign words to chunks by timestamp, slice one text per
    chunk — so the probe view shows the TEXT a re-decomposition would actually
    commit, not the committed rows' borrowed approximation (drive feedback
    2026-07-22: a split chunk repeated its whole parent text, unjudgeable)."""
    words = [FAWord(text=w, start_time=s, end_time=e) for w, s, e in fa_words]
    chunks = [VADChunk(index=i, start_time=s, end_time=e)
              for i, (s, e) in enumerate(predicted_local)]
    spans = map_fa_words_to_text(coarse_text, words)
    assignments = assign_words_to_chunks(words, chunks)
    segs = build_segments_from_alignment(coarse_text, spans, assignments, len(chunks))
    return [s.text for s in segs]


def split_predicted(
    coarse_text: str,                          # The authoritative transcriber's full coarse text
    fa_words: List[Tuple[str, float, float]],  # (word, start_s, end_s) — WAV-local FA output (probe_fa)
    predicted_local: List[Tuple[float, float]],  # Predicted chunk spans, WAV-local
    sentence_spans: List[Tuple[int, int]],     # Capability sentence spans over coarse_text (probe_segment)
    min_chunk_s: float = 0.5,                  # Split min sub-chunk duration guard (the pipeline default)
) -> List[Tuple[float, float]]:  # The sentence-split refined spans (WAV-local)
    """Run the decomp pipeline's own SENTENCE-SPLIT stage over a probe skeleton
    (pure; DEC f1024568 deliverable d — preview before any commit).

    Reuses the core stage verbatim (same policy, same guard) with the
    CAPABILITY-delivered sentence spans (B.5), so the probe view shows exactly
    the skeleton a `--sentence-split` re-decomposition would commit. Feed the
    result to realign_rows for the per-chunk text."""
    words = [FAWord(text=w, start_time=s, end_time=e) for w, s, e in fa_words]
    chunks = [VADChunk(index=i, start_time=s, end_time=e)
              for i, (s, e) in enumerate(predicted_local)]
    spans = map_fa_words_to_text(coarse_text, words)
    end_words = sentence_end_word_indices(spans, sentence_spans)
    refined = split_chunks_at_sentence_gaps(chunks, words, end_words,
                                            min_chunk_s=min_chunk_s)
    return [(c.start_time, c.end_time) for c in refined]
