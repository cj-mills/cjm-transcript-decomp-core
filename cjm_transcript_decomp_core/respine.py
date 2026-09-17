"""Chunk RESPINE — re-derive ONE coarse chunk of a LIVE spine from a landed external transcript (work item 7a5e9c84; design ruling 0b4d5cfa, amendment 4a7ec4f8).

The transcription workflow's escalation gesture lands a pasted transcript as the
`<model id>/manual` third transcriber with provenance (cf0b91d6 / 60eea30d); before
this verb the only way to fold that better text into a spine was a WHOLE-SOURCE
respine — a fresh skeleton, the old one retired (a7617bd4), every correction on it
stranded (7c0af775). `respine-chunk` keeps the spine: ONE verb, two modes.

  --prompt      render the chunk's escalation prompt with context (the transcription
                core's render_escalation_prompt) and print it with its prompt hash and
                the chunk's audio path — no writes.
  --text-file   land the paste (the same land_chunk_transcript the transcription
                app's import uses), save the derived transcription manifest, then
                re-derive THAT chunk under the live spine's OWN decomp policy (its
                manifest's DecompConfig, recorded capability configs, the same event
                proposal set) and replace the chunk's segments IN the live spine.

Identity (hard problem one, 0b4d5cfa (3)): the new segments keep the live spine's
`skeleton_hash` (every picker groups by it) and fork their node ids through
`identity_salt` = the landed Transcript id — so the same landing twice collides into
a verified no-op, and a different landing mints fresh segments.

Ordering (hard problem two, 0b4d5cfa (4)): `index` is source-wide and contiguous, so
the tail is RENUMBERED in the same journaled fact op (`chunk-respine`, replayed by
`apply_spine_fact`) that stamps `superseded_by` on the chunk's old segments and
appends a `respined_chunks` entry on the Source. Readers add ONE predicate
(`superseded_by is null`). NEXT edges bridge the neighbours to the new run; the stale
edges stay — nothing is deleted.

Dependents (0b4d5cfa (5) + 4a7ec4f8): corrections anchored on the old segments stop
projecting the moment those segments leave the live view. The verb LISTS them and
refuses unless told to strand — EXCEPT the two source-truth classes the chunk-scoped
transfer re-homes by time: accepted wordless event inserts and speaker assignments.
The transfer itself is the correction core's (it owns the overlay vocabulary); this
core discovers it through the `cjm_transcript_decomp_core.chunk_transfer` entry-point
group, the same registry shape the replay handlers use — no dependency inversion.
"""

import importlib.metadata
import json
import logging
import time
from bisect import bisect_right
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from cjm_context_graph_layer.declare import Derivation, derivation_to_graph
from cjm_context_graph_layer.grammar import make_edge, SpineRelations
from cjm_context_graph_layer.identity import derive_node_id
from cjm_context_graph_layer.journal import journal_extend
from cjm_context_graph_layer.ops import graph_task
from cjm_context_graph_primitives.journal import append_op
from cjm_context_graph_primitives.query import EdgeQuery, NodeQuery, PropertyPredicate, RelationPredicate
from cjm_substrate.core.journal_store import SubstrateEventType
from cjm_transcript_graph_schema.schema import (external_config_hash, external_transcriber_name,
                                                RESPINE_OP_VERB, RespinedChunkEntry,
                                                SEGMENT_SUPERSEDED_BY_PROP, source_node_id,
                                                SOURCE_RESPINED_CHUNKS_PROP, TranscriptGraphLabels)
from cjm_transcription_core.chunk import (apply_chunk_update, derive_manifest, land_chunk_transcript,
                                          load_run_manifest, prior_config_hash, PRODUCER_EXTERNAL,
                                          render_escalation_prompt, save_manifest, text_shape,
                                          wordwrap_warning)
from cjm_transcription_core.models import new_run_id as new_landing_run_id
from cjm_transcript_decomp_core.graph import (build_extension_payload, resolve_root_ids,
                                              verify_source)
from cjm_transcript_decomp_core.models import (DecompConfig, DecompManifest, DecompSourceRecord,
                                               new_run_id)
from cjm_transcript_decomp_core.pipeline import (_journal_run_event, collect_capability_info,
                                                 decompose_source, resolve_event_propsets)
from cjm_transcript_decomp_core.retire import (_rows, apply_spine_fact, default_live_spine,
                                               get_source, list_spines, resolve_spine,
                                               source_rendition_ids, spine_label)
from cjm_transcript_decomp_core.runs import DecompIndex

logger = logging.getLogger(__name__)

TRANSFER_ENTRY_POINT_GROUP = "cjm_transcript_decomp_core.chunk_transfer"  # The correction core registers its chunk-scoped transfer here
_CORRECTS = "CORRECTS"    # Correction-overlay relations, literal on purpose (retire.py precedent: no correction-core dependency)
_REVIEWED = "REVIEWED"
_SUPERSEDES = "SUPERSEDES"
_ID_BATCH = 500


# ---- pure: manifest lineage, chunk resolution, plans ------------------------------------------

def find_decomp_manifest(
    manifests: List[Dict[str, Any]],  # DecompIndex.runs (newest first, "_path" set)
    source_id: str,                   # The Source node id
    skeleton_hash: str,               # The LIVE spine's skeleton hash
) -> Tuple[Dict[str, Any], Dict[str, Any]]:  # (the decomp manifest, its source record for this source)
    """The decomp run that minted the live spine (pure): the newest manifest whose
    source record for THIS source carries the spine's skeleton hash — per-source
    first (0.2.6 multi-source event carve), the run-level hash as the pre-0.2.6
    fallback. Loud on a miss: the policy the chunk must be cut by is exactly what
    that manifest records, so no manifest = no respine."""
    for m in manifests:
        for rec in m.get("sources") or []:
            if str(rec.get("source_node_id") or "") != source_id:
                continue
            h = str(rec.get("skeleton_config_hash") or m.get("skeleton_config_hash") or "")
            if h == skeleton_hash:
                return m, rec
    raise ValueError(f"no decomp manifest records skeleton {skeleton_hash[:23]}… for source "
                     f"{source_id[:8]} — the live spine's run manifest is the policy authority "
                     f"(scanned {len(manifests)} decomp manifest(s))")


def source_entry_for(
    manifest: Dict[str, Any],  # A loaded transcription run manifest
    source_id: str,            # The Source node id
) -> Tuple[int, Dict[str, Any]]:  # (source index in the manifest, the source entry)
    """The transcription manifest's entry for a Source, by recomputed identity (pure)."""
    for i, s in enumerate(manifest.get("sources") or []):
        h = str(s.get("content_hash") or "")
        if h and source_node_id(h) == source_id:
            return i, s
    raise ValueError(f"source {source_id[:8]} is not in the transcription manifest "
                     f"{manifest.get('run_id')} ({len(manifest.get('sources') or [])} sources)")


def chunk_entry_for(
    source_entry: Dict[str, Any],     # The transcription manifest's source entry
    chunk: Optional[int] = None,      # The coarse chunk's manifest `index` (= the AudioSegment index)
    at_time: Optional[float] = None,  # A source-coordinate time inside the chunk (bisect over chunk starts)
) -> Tuple[int, Dict[str, Any]]:  # (position in the entry's segments list, the segment entry)
    """Resolve ONE coarse chunk of a source entry (pure): by manifest index, or by
    the source time a correction cursor stands at (bisect_right over the ordered
    chunk starts — the correction app's ChunkRef convention). Exactly one of the
    two selectors; both absent or both present refuse."""
    segs = list(source_entry.get("segments") or [])
    if not segs:
        raise ValueError("the source entry has no segments")
    if (chunk is None) == (at_time is None):
        raise ValueError("give exactly one of --chunk (manifest index) or --at-time (source seconds)")
    if chunk is not None:
        for pos, s in enumerate(segs):
            if int(s.get("index", -1)) == int(chunk):
                return pos, s
        raise ValueError(f"chunk {chunk} not in the source entry (indices "
                         f"{[int(s.get('index', -1)) for s in segs][:8]}…)")
    starts = [float(s.get("start", 0.0)) for s in segs]
    pos = max(0, bisect_right(starts, float(at_time)) - 1)
    s = segs[pos]
    if float(at_time) > float(s.get("end", 0.0)) + 1e-6:
        raise ValueError(f"time {at_time:.2f}s falls after the last chunk's end "
                         f"({float(s.get('end', 0.0)):.2f}s)")
    return pos, s


def plan_renumber(
    old_indices: List[int],           # The chunk's LIVE segment indices, ascending
    new_count: int,                   # How many segments replace them
    tail: List[Tuple[str, int]],      # (segment id, index) of every live segment AFTER the chunk
) -> Tuple[int, int, List[Tuple[str, int]]]:  # (first index, delta, [(id, new index)] for the tail)
    """The renumber plan (pure; 0b4d5cfa (4), the 'do it properly' ruling): the new
    segments take the chunk's first index onward; every later live segment shifts by
    delta = new - old so the source-wide index stays contiguous. Refuses a chunk whose
    live indices are not contiguous (a spine invariant already broken) and a tail
    that overlaps the chunk's own range."""
    if not old_indices:
        raise ValueError("the chunk has no live segments to replace")
    lo, hi = min(old_indices), max(old_indices)
    if sorted(old_indices) != list(range(lo, hi + 1)):
        raise ValueError(f"the chunk's live segment indices are not contiguous ({old_indices[:10]}…): "
                         "the spine invariant is already broken — inspect before respining")
    if new_count < 1:
        raise ValueError("the re-derivation produced no segments — nothing to land")
    bad = [i for _, i in tail if i <= hi]
    if bad:
        raise ValueError(f"tail segments overlap the chunk's index range ({bad[:5]}…)")
    delta = new_count - (hi - lo + 1)
    updates = [(sid, i + delta) for sid, i in tail] if delta else []
    return lo, delta, updates


def classify_dependents(
    corrections: List[Dict[str, Any]],  # Correction rows: {"id", "correction_type", "payload", "status", "superseded": bool, "rationale"}
    old_ids: Set[str],                  # The chunk's old segment ids
) -> Dict[str, List[Dict[str, Any]]]:  # {"events": [...], "speakers": [...], "stranded": [...]} (active rows only)
    """Sort the chunk's dependents into the two transferable classes and the rest
    (pure; 0b4d5cfa (5) + 4a7ec4f8). Transferable BY TIME: accepted, wordless,
    labeled chunk inserts anchored on the chunk (the propose lane's event layer —
    source-truth) and speaker assignments (who speaks at t is source-truth).
    Everything else active that anchors an old segment strands: marks, reviews,
    time nudges, text edits, prunes, splits, word-bearing inserts (spine-truth: the
    better text is exactly what replaces them). Superseded rows are ignored."""
    events: List[Dict[str, Any]] = []
    speakers: List[Dict[str, Any]] = []
    stranded: List[Dict[str, Any]] = []
    for c in corrections:
        if c.get("superseded"):
            continue
        p = c.get("payload") or {}
        ctype = str(c.get("correction_type") or "")
        op = str(p.get("operation") or "")
        if ctype == "insertion" and op == "chunk_insert":
            anchored = (str(p.get("after_segment_id") or "") in old_ids
                        or str(p.get("before_segment_id") or "") in old_ids)
            if not anchored:
                continue
            if (p.get("label") and not str(p.get("text") or "").strip()
                    and str(c.get("status") or "applied") != "proposed"
                    and str(c.get("rationale") or "") != "chunk-split"):
                events.append(c)
            else:
                stranded.append(c)
            continue
        if ctype == "speaker" and op == "speaker_assign":
            if any(str(s) in old_ids for s in (p.get("segment_ids") or [])):
                speakers.append(c)
            continue
        stranded.append(c)
    return {"events": events, "speakers": speakers, "stranded": stranded}


def describe_dependents(
    dep: Dict[str, List[Dict[str, Any]]],  # classify_dependents output
    reviews: int = 0,                      # REVIEWED edges (session walk markers) on the old segments
) -> str:  # One readable listing
    """The refusal / readout listing (pure)."""
    by_type: Dict[str, int] = {}
    for c in dep["stranded"]:
        k = str(c.get("correction_type") or "?")
        by_type[k] = by_type.get(k, 0) + 1
    parts = [f"{len(dep['events'])} event insert(s) + {len(dep['speakers'])} speaker assignment(s) transferable by time"]
    if dep["stranded"] or reviews:
        detail = ", ".join(f"{n} {k}" for k, n in sorted(by_type.items()))
        parts.append(f"{len(dep['stranded'])} would STRAND ({detail}{'; ' if detail and reviews else ''}"
                     f"{f'{reviews} review marker(s)' if reviews else ''})")
    return "; ".join(parts)


def build_respine_op(
    *,
    source_id: str,
    entry: RespinedChunkEntry,          # The Source-side record (op_id set)
    skeleton_hash: str,
    renumber: List[Tuple[str, int]],    # plan_renumber's tail updates
    existing_entries: List[Dict[str, Any]],  # The Source's current respined_chunks list
    actor: str,
) -> Dict[str, Any]:  # The journaled fact op (apply_spine_fact's shape)
    """ONE replayed property-update op (pure): superseded_by on the old segments, the
    new index on every later live segment, the appended respined_chunks entry on
    the Source. Deterministic op id = derive(source, transcript) so a replay
    converges and a re-land of the same transcript names the same op."""
    updates: List[Dict[str, Any]] = [
        {"id": sid, "properties": {SEGMENT_SUPERSEDED_BY_PROP: entry.op_id}} for sid in entry.old_segments]
    updates += [{"id": sid, "properties": {"index": int(i)}} for sid, i in renumber]
    kept = [dict(e) for e in existing_entries if str(e.get("op_id") or "") != entry.op_id]
    updates.append({"id": source_id, "properties": {SOURCE_RESPINED_CHUNKS_PROP: kept + [entry.to_dict()]}})
    return {"verb": RESPINE_OP_VERB, "actor": actor,
            "args": {"act": "chunk-respine", "source_id": source_id, "skeleton_hash": skeleton_hash,
                     "audio_segment": entry.audio_segment, "transcript": entry.transcript,
                     "run": entry.run, "op_id": entry.op_id,
                     "old_segments": len(entry.old_segments), "new_segments": len(entry.new_segments),
                     "renumbered": len(renumber), "stranded": len(entry.stranded),
                     "carried": len(entry.carried), "straddles": len(entry.straddles)},
            "updates": updates}


def respine_op_id(source_id: str, transcript_id: str) -> str:  # Deterministic chunk-respine op id
    """The op id `superseded_by` names: a function of (source, landed transcript)."""
    return derive_node_id("chunk-respine", source_id, transcript_id)


def decomp_config_from(
    snapshot: Dict[str, Any],  # A decomp manifest's `config` block (DecompConfig.to_dict of the live run)
) -> DecompConfig:  # The live spine's policy, rebuilt (unknown keys ignored; headless)
    """Rebuild the live spine's DecompConfig from its manifest snapshot (pure): the
    SAME policy cuts the chunk as cut its neighbours (0b4d5cfa (2)). Headless by
    construction — the seams were confirmed when the spine was born."""
    fields = set(DecompConfig.__dataclass_fields__)
    kw = {k: v for k, v in (snapshot or {}).items() if k in fields}
    cfg = DecompConfig(**kw)
    cfg.assume_yes = True
    cfg.force = False
    return cfg


def bridge_edges(
    prev_id: Optional[str],  # The live segment right BEFORE the chunk (None at the spine head)
    new_ids: List[str],      # The replacement segments, index order
    next_id: Optional[str],  # The live segment right AFTER the chunk (None at the spine tail)
) -> List[Dict[str, Any]]:  # NEXT edge wires bridging the neighbours to the new run
    """Bridge NEXT from the previous live segment to the new first and from the new
    last onward (pure). The stale NEXT edges through the old segments stay — nothing
    deletes; no live reader walks NEXT (0b4d5cfa (4))."""
    out: List[Dict[str, Any]] = []
    if not new_ids:
        return out
    if prev_id:
        out.append(make_edge(prev_id, new_ids[0], SpineRelations.NEXT))
    if next_id:
        out.append(make_edge(new_ids[-1], next_id, SpineRelations.NEXT))
    return out


def transfer_handler() -> Optional[Callable[..., Any]]:  # The registered chunk-scoped transfer, or None
    """Discover the correction core's chunk-scoped transfer through the entry-point
    group (the replay-registry pattern, DEC 426658f1): None when no core in this env
    registers one — the verb then refuses to strand transferable dependents silently."""
    try:
        eps = importlib.metadata.entry_points(group=TRANSFER_ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover — pre-3.10 selection API
        eps = importlib.metadata.entry_points().get(TRANSFER_ENTRY_POINT_GROUP, [])
    for ep in eps:
        try:
            return ep.load()
        except Exception as e:  # a broken registration is loud, never silent
            logger.warning(f"chunk_transfer entry point {ep.name!r} failed to load: {e}")
    return None


# ---- graph reads -----------------------------------------------------------------------------

async def live_chunk_segments(
    queue: Any, graph_id: str,
    rendition_id: str,      # The chunk's AudioRendition (the chunk's fine segments hang under it)
    skeleton_hash: str,     # The live spine
) -> List[Dict[str, Any]]:  # {"id","index","start_time","end_time"} rows, index order
    """The chunk's LIVE segments: PART_OF its rendition, on the spine, not superseded."""
    q = NodeQuery(label=TranscriptGraphLabels.SEGMENT, project=["index", "start_time", "end_time"],
                  where=[PropertyPredicate("skeleton_hash", "eq", skeleton_hash),
                         PropertyPredicate(SEGMENT_SUPERSEDED_BY_PROP, "is_null")],
                  related=RelationPredicate(SpineRelations.PART_OF, node_id=rendition_id))
    rows = _rows(await graph_task(queue, graph_id, "query_nodes", query=q.to_dict()))
    return sorted(({"id": r["id"], "index": int(r.get("index") or 0),
                    "start_time": r.get("start_time"), "end_time": r.get("end_time")} for r in rows),
                  key=lambda r: r["index"])


async def live_spine_index(
    queue: Any, graph_id: str,
    rendition_ids: List[str],  # Every rendition of the source
    skeleton_hash: str,        # The live spine
    page: int = 5000,
) -> List[Tuple[str, int]]:  # (id, index) of every live segment, index order
    """The whole live spine's (id, index) — the renumber plan's input."""
    where = [PropertyPredicate("skeleton_hash", "eq", skeleton_hash),
             PropertyPredicate(SEGMENT_SUPERSEDED_BY_PROP, "is_null")]
    out: List[Tuple[str, int]] = []
    offset = 0
    while True:
        q = NodeQuery(label=TranscriptGraphLabels.SEGMENT, project=["index"], where=where,
                      related=RelationPredicate(SpineRelations.PART_OF, node_ids=list(rendition_ids)),
                      limit=page, offset=offset)
        rows = _rows(await graph_task(queue, graph_id, "query_nodes", query=q.to_dict()))
        out.extend((r["id"], int(r.get("index") or 0)) for r in rows)
        if len(rows) < page:
            break
        offset += len(rows)
    return sorted(out, key=lambda x: x[1])


async def read_dependents(
    queue: Any, graph_id: str,
    segment_ids: List[str],  # The chunk's old segments
) -> Tuple[List[Dict[str, Any]], int]:  # (correction rows for classify_dependents, REVIEWED edge count)
    """What points at the old segments: every Correction with a CORRECTS edge into
    them (properties + whether a SUPERSEDES edge retired it) and the REVIEWED count."""
    corr_ids: List[str] = []
    reviews = 0
    for i in range(0, len(segment_ids), _ID_BATCH):
        batch = segment_ids[i:i + _ID_BATCH]
        cq = EdgeQuery(relation_type=_CORRECTS, target_ids=batch, project=[])
        for r in _rows(await graph_task(queue, graph_id, "query_edges", query=cq.to_dict())):
            if r.get("source_id") and r["source_id"] not in corr_ids:
                corr_ids.append(r["source_id"])
        rq = EdgeQuery(relation_type=_REVIEWED, target_ids=batch, project=[])
        reviews += len(_rows(await graph_task(queue, graph_id, "query_edges", query=rq.to_dict())))
    if not corr_ids:
        return [], reviews
    superseded: Set[str] = set()
    for i in range(0, len(corr_ids), _ID_BATCH):
        sq = EdgeQuery(relation_type=_SUPERSEDES, target_ids=corr_ids[i:i + _ID_BATCH], project=[])
        superseded.update(r.get("target_id") for r in _rows(await graph_task(queue, graph_id, "query_edges", query=sq.to_dict())))
    rows: List[Dict[str, Any]] = []
    for i in range(0, len(corr_ids), _ID_BATCH):
        nq = NodeQuery(ids=corr_ids[i:i + _ID_BATCH], project=["correction_type", "payload", "status", "rationale"])
        for r in _rows(await graph_task(queue, graph_id, "query_nodes", query=nq.to_dict())):
            payload = r.get("payload") or {}
            if isinstance(payload, str):  # a projected JSON column comes back serialized
                try:
                    payload = json.loads(payload)
                except ValueError:
                    payload = {}
            rows.append({"id": r["id"], "correction_type": r.get("correction_type"),
                         "payload": payload if isinstance(payload, dict) else {}, "status": r.get("status"),
                         "rationale": r.get("rationale"), "superseded": r["id"] in superseded})
    return rows, reviews


# ---- the verb ---------------------------------------------------------------------------------

async def resolve_chunk_context(
    queue: Any, graph_id: str,
    *,
    source_id: str,
    skeleton_selector: Optional[str],  # None = the default live spine; else resolve_spine semantics
    runs_dir: Path,                    # Where both cores' run manifests live
    chunk: Optional[int] = None,
    at_time: Optional[float] = None,
) -> Dict[str, Any]:  # Everything both modes share (spine, manifests, the chunk, its live segments)
    """Resolve the live spine -> its decomp manifest -> the transcription manifest it
    consumed -> the chunk entry -> the chunk's rendition + live segments."""
    rends = await source_rendition_ids(queue, graph_id, source_id)
    spines = await list_spines(queue, graph_id, source_id, rends)
    if not spines:
        raise ValueError(f"source {source_id[:8]} has no decomposed spine")
    sp = resolve_spine(spines, skeleton_selector) if skeleton_selector else default_live_spine(spines)
    if sp is None:
        raise ValueError("every spine of this source is retired — nothing live to respine into")
    if sp.get("retired"):
        raise ValueError(f"spine {spine_label(sp)} is RETIRED — a chunk respines into a LIVE spine only")
    skeleton_hash = sp.get("skeleton_hash")
    if not skeleton_hash:
        raise ValueError("the legacy (pre-split) spine has no decomp-manifest lineage — respine a skeleton spine")
    index = DecompIndex(str(runs_dir))
    index.load()
    dm, rec = find_decomp_manifest(index.runs, source_id, skeleton_hash)
    tm_path = Path(str(dm.get("source_manifest") or ""))
    if not tm_path.is_file():
        raise ValueError(f"the live spine's transcription manifest is missing: {tm_path}")
    tm = load_run_manifest(tm_path)
    si, src_entry = source_entry_for(tm, source_id)
    pos, seg = chunk_entry_for(src_entry, chunk=chunk, at_time=at_time)
    roots = resolve_root_ids(src_entry, tm.get("capabilities") or {})
    aseg = roots["audio_segments"][pos]
    old = await live_chunk_segments(queue, graph_id, aseg["rendition"], skeleton_hash)
    return {"spine": sp, "skeleton_hash": skeleton_hash, "renditions": rends,
            "decomp_manifest": dm, "decomp_record": rec, "transcription_manifest": tm,
            "transcription_manifest_path": tm_path, "source_index": si, "source_entry": src_entry,
            "chunk_pos": pos, "chunk_entry": seg, "audio_segment": aseg["audio_segment"],
            "rendition": aseg["rendition"], "old_segments": old}


def render_chunk_prompt(
    ctx: Dict[str, Any],  # resolve_chunk_context output
    template: Optional[str] = None,  # Prompt template override (None = the transcription core's default)
    slot_text: Optional[Callable[[float, float], Optional[str]]] = None,  # LIVE slot-text provider (chunk start, end) -> the corrected spine's text over that span (finding c63cd2e3); None = manifest text
) -> Dict[str, Any]:  # render_escalation_prompt's dict + "audio" (the chunk's model-input WAV) + "chunk"
    """Mode one (--prompt): the chunk's escalation prompt WITH CONTEXT — no writes.
    Slot text follows per-chunk authority (an escalated neighbour lends its landed
    text), else the live run's text_from (the accuracy model), never the first
    transcriber in manifest order (the lightweight one). A caller holding the
    corrected spine passes `slot_text` and the slots read THAT (manual fidelity
    edits included); the CLI has no live spine and keeps the manifest source."""
    text_from = str((ctx["decomp_manifest"].get("config") or {}).get("text_from") or "") or None
    r = render_escalation_prompt(ctx["transcription_manifest"], ctx["source_index"],
                                 int(ctx["chunk_entry"].get("index", -1)), template=template,
                                 transcriber=text_from, slot_text=slot_text)
    r["audio"] = str(ctx["chunk_entry"].get("model_input_path") or "")
    r["chunk"] = int(ctx["chunk_entry"].get("index", -1))
    r["chunk_range"] = (float(ctx["chunk_entry"].get("start", 0.0)), float(ctx["chunk_entry"].get("end", 0.0)))
    r["live_segments"] = len(ctx["old_segments"])
    return r


async def respine_chunk(
    manager: Any,              # CapabilityManager holding the graph capability (the verb loads VAD/FA/SEG itself)
    queue: Any,                # Started job queue
    load_capabilities: Callable[..., None],  # cli.load_capabilities (manager, ids, configs) — injected so the seat stays testable
    *,
    graph_id: str,
    journal_path: Optional[str],
    ctx: Dict[str, Any],       # resolve_chunk_context output
    text: str,                 # The pasted transcript
    model_id: str,             # The external model id (files as `<model id>/manual`)
    prompt_hash: str,          # The prompt TEMPLATE's hash ("" = none recorded)
    text_source: str,          # Provenance ("paste", "gemini web ui", ...)
    reason: str,               # Why (journaled)
    actor: str,
    runs_dir: Path,
    workspace: Any = None,     # Resolved Workspace (manifests record ${WS}/ paths) or None
    strand: bool = False,      # Strand the non-transferable dependents (the explicit say-so)
    dry_run: bool = False,     # Resolve + plan everything up to the landing; write nothing
    sysmon_capability: Optional[str] = None,  # Loaded first when given (GPU attribution)
) -> Dict[str, Any]:  # The readout: counts, ids, paths
    """Mode two (--text-file): land -> re-derive the chunk under the live policy ->
    extend the live spine (salted ids) -> transfer the two source-truth classes ->
    ONE chunk-respine fact (stamps + renumber + Source entry)."""
    source_id = source_node_id(str(ctx["source_entry"].get("content_hash") or ""))
    skeleton_hash: str = ctx["skeleton_hash"]
    old = ctx["old_segments"]
    if not old:
        raise ValueError("the chunk has no live segments on this spine — nothing to replace "
                         "(an un-decomposed or empty chunk is a whole-source decomp's job)")
    old_ids = [r["id"] for r in old]
    text = (text or "").strip()
    if not text:
        raise ValueError("the pasted text is empty")
    shape = text_shape(text)
    warning = wordwrap_warning(shape)
    tname = external_transcriber_name(model_id)
    chash = external_config_hash(model_id, prompt_hash)
    tm = ctx["transcription_manifest"]
    seg = ctx["chunk_entry"]
    src_entry = ctx["source_entry"]
    si = int(ctx["source_index"])
    pos = int(ctx["chunk_pos"])

    # Dependents FIRST (0b4d5cfa (5)): the refusal must fire before any write.
    corr_rows, reviews = await read_dependents(queue, graph_id, old_ids)
    dep = classify_dependents(corr_rows, set(old_ids))
    handler = transfer_handler()
    transferable = dep["events"] or dep["speakers"]
    if dep["stranded"] and not strand:
        raise ValueError(f"the chunk's old segments carry dependents — {describe_dependents(dep, reviews)}. "
                         f"Pass --strand to strand the non-transferable ones (they need re-evaluation on "
                         f"the escalated text anyway); the transferable classes carry by time")
    if transferable and handler is None and not strand:
        raise ValueError(f"{describe_dependents(dep, reviews)} — but no chunk-scoped transfer is registered "
                         f"in this env (install cjm-transcript-correction-core beside decomp-core, or pass "
                         f"--strand to strand them)")

    # Idempotency: the same transcript already respined this chunk -> a verified no-op.
    src_node = await get_source(queue, graph_id, source_id)
    existing_entries = [e for e in ((src_node or {}).get("properties") or {}).get(SOURCE_RESPINED_CHUNKS_PROP) or []
                        if isinstance(e, dict)]
    # A prior landing under the same (model, prompt) collides into the same Transcript id
    # at the landing (extend-verify); the respine of it is caught below by op id.
    prior_hash = prior_config_hash(tm, seg, tname)

    readout: Dict[str, Any] = {"chunk": int(seg.get("index", -1)), "audio_segment": ctx["audio_segment"],
                               "old_segments": len(old_ids), "skeleton_hash": skeleton_hash,
                               "transcriber": tname, "config_hash": chash, "wordwrap_warning": warning,
                               "dependents": {"events": len(dep["events"]), "speakers": len(dep["speakers"]),
                                              "stranded": len(dep["stranded"]), "reviews": reviews},
                               "dry_run": dry_run}
    if dry_run:
        readout["plan"] = describe_dependents(dep, reviews)
        return readout

    run_id = new_run_id()
    land_run = new_landing_run_id()
    queue.set_run_context(run_id=run_id, actor=actor)
    _journal_run_event(manager, SubstrateEventType.RUN_STARTED.value, run_id, actor, {
        "core": "cjm-transcript-decomp-core", "kind": "respine-chunk", "source_id": source_id,
        "audio_segment": ctx["audio_segment"], "skeleton_hash": skeleton_hash,
        "parent_run_id": ctx["decomp_manifest"].get("run_id"), "graph_capability": graph_id})
    try:
        # 0. PROVE the policy stack BEFORE any write (the 2026-09-16 field failure: a
        # forced-aligner load that failed after the landing left a transcript + derived
        # manifest behind with no respine). The live spine's OWN policy: its manifest's
        # DecompConfig, its recorded capability configs, verified by config hash.
        dm = ctx["decomp_manifest"]
        cfg = decomp_config_from(dm.get("config") or {})
        cfg.graph_capability = graph_id
        parent_transcribers = list((tm.get("config") or {}).get("transcriber_capabilities") or [])
        text_from = cfg.text_from or (parent_transcribers[0] if len(parent_transcribers) == 1 else None)
        if not text_from or text_from not in parent_transcribers:
            raise ValueError(f"the live run's text_from {cfg.text_from!r} is not among the manifest's "
                             f"transcribers {parent_transcribers}")
        recorded = dm.get("capabilities") or {}
        stack = [cfg.vad_capability, cfg.fa_capability] + ([cfg.seg_capability] if cfg.sentence_split else [])
        configs = {iid: dict((recorded.get(iid) or {}).get("config") or {}) for iid in stack}
        for iid in stack:
            configs[iid].pop("db_path", None)
        load_order = ([sysmon_capability] if sysmon_capability else []) + stack
        try:
            load_capabilities(manager, load_order, configs={k: v for k, v in configs.items() if v} or None)
        except SystemExit as e:
            raise ValueError(f"the policy stack did not load ({e}) — nothing was written; the worker "
                             f"diagnostics name the cause") from e
        live_info = collect_capability_info(manager, stack)
        for iid in stack:
            want = str((recorded.get(iid) or {}).get("config_hash") or "")
            got = str((live_info.get(iid) or {}).get("config_hash") or "")
            if want and got and want != got:
                for x in reversed(load_order):
                    try:
                        manager.unload_capability(x)
                    except Exception:
                        pass
                raise ValueError(f"{iid} loaded with config hash {got[:19]}… but the live spine was cut "
                                 f"under {want[:19]}… — a different policy would cut this chunk "
                                 f"differently from its neighbours; refuse")

        # 1. LAND the paste — the transcription core's one landing (the app's `i` import).
        landing = {"producer": PRODUCER_EXTERNAL, "reason": reason, "parent_run_id": tm.get("run_id"),
                   "actor": actor, "landed_at": time.time(), "model_id": model_id,
                   "prompt_hash": prompt_hash, "text_source": text_source, "text_shape": shape,
                   "respine": {"decomp_run_id": run_id, "skeleton_hash": skeleton_hash}}
        metadata = {"model": model_id, "source_start_time": float(seg.get("start", 0.0)),
                    "source_end_time": float(seg.get("end", 0.0)), "landing": landing}
        rec = await land_chunk_transcript(
            queue, graph_id, src_entry, seg, transcriber=tname, config_hash=chash, text=text,
            metadata=metadata, prior_config_hash=prior_hash, producer=PRODUCER_EXTERNAL, reason=reason,
            actor=actor, run_id=land_run, journal_path=journal_path, node_actor=actor,
            method="external-landing")
        transcript_id = str(rec["transcript"])
        op_id = respine_op_id(source_id, transcript_id)
        if any(str(e.get("op_id") or "") == op_id for e in existing_entries):
            readout.update({"transcript": transcript_id, "op_id": op_id, "noop": True,
                            "status": "already respined with this transcript — no-op"})
            _journal_run_event(manager, SubstrateEventType.RUN_FINISHED.value, run_id, actor, {
                "core": "cjm-transcript-decomp-core", "kind": "respine-chunk", "status": "noop",
                "transcript": transcript_id})
            for x in reversed(load_order):
                try:
                    manager.unload_capability(x)
                except Exception:
                    pass
            return readout
        derived = derive_manifest(tm, run_id=land_run, parent_path=ctx["transcription_manifest_path"],
                                  kind="add-transcript")
        apply_chunk_update(derived, si, int(seg.get("index", -1)), tname, {
            "job_id": f"{land_run}_src{si}_seg{int(seg.get('index', -1)):04d}_external", "text": text,
            "metadata": metadata, "config_hash": chash,
            "landing": {**landing, "transcript_id": transcript_id, "supersedes": rec["supersedes"]}},
            capability_info={"name": tname, "version": "manual", "db_path": None, "config_hash": chash,
                             "config": {"model_id": model_id, "prompt_hash": prompt_hash,
                                        "text_source": text_source}})
        derived_path = save_manifest(derived, runs_dir / f"{land_run}.json", workspace=workspace)
        d_src = derived["sources"][si]
        d_caps = derived.get("capabilities") or {}
        transcribers = list((derived.get("config") or {}).get("transcriber_capabilities") or [])

        # 2. RE-DERIVE the chunk under the live spine's OWN policy (the stack proven in 0).
        try:
            spans: Optional[List[Tuple[float, float]]] = None
            if cfg.event_split:
                ptr = str(ctx["decomp_record"].get("event_propset") or dm.get("event_propset") or "")
                if not ptr:
                    raise ValueError("the live run carved by event proposal set but recorded no pointer for this source")
                spans = resolve_event_propsets([ptr], [d_src], list(cfg.event_classes))[0][1]
            _, aligned = await decompose_source(queue, cfg, d_src, si, transcribers, text_from,
                                               event_spans=spans, pseg_indices={pos})
        finally:
            for iid in reversed(load_order):
                try:
                    manager.unload_capability(iid)
                except Exception as e:  # best-effort teardown, never mask the outcome
                    logger.warning(f"unload {iid} failed: {e}")
        if not aligned:
            raise ValueError("the re-derivation produced no segments for this chunk — the landed text "
                             "aligned to nothing (check the paste against the chunk's audio)")

        # 3. EXTEND the live spine: salted ids, the chunk's first index onward.
        spine_rows = await live_spine_index(queue, graph_id, ctx["renditions"], skeleton_hash)
        old_idx = [r["index"] for r in old]
        tail = [(sid, i) for sid, i in spine_rows if i > max(old_idx)]
        first, delta, renumber = plan_renumber(old_idx, len(aligned), tail)
        for k, a in enumerate(aligned):
            a.index = first + k
        policy_label = "+".join(p for p in (dm.get("split_policy"), dm.get("event_split_policy"),
                                            dm.get("word_rescue_policy")) if p) or None
        nodes, edges, ids = build_extension_payload(d_src, d_caps, skeleton_hash, text_from, aligned,
                                                    split_policy=policy_label, identity_salt=transcript_id)
        prev_id = next((sid for sid, i in reversed(spine_rows) if i < min(old_idx)), None)
        next_id = next((sid for sid, i in spine_rows if i > max(old_idx)), None)
        edges.extend(bridge_edges(prev_id, ids["segments"], next_id))
        new_ids = list(ids["segments"])
        res = await journal_extend(queue, graph_id, nodes, edges, journal_path=journal_path,
                                   verb="spine-extension", actor="pipeline:cjm-transcript-decomp-core",
                                   run=run_id,
                                   args={"act": "chunk-respine", "source_id": source_id,
                                         "audio_segment": ctx["audio_segment"], "transcript": transcript_id,
                                         "segments": len(new_ids), "text_from": text_from,
                                         "skeleton_hash": skeleton_hash})
        if res.nodes_added:
            d = Derivation(actor="host:cjm-transcript-decomp-core", method="alignment-fold/v1",
                           input_ids=ids["transcripts_used"], output_ids=[source_id],
                           properties={"run_id": run_id, "segments": len(new_ids), "text_from": text_from,
                                       "act": "chunk-respine", "audio_segment": ctx["audio_segment"]})
            dn, de = derivation_to_graph(d)
            await journal_extend(queue, graph_id, [dn], de, journal_path=journal_path, verb="derivation",
                                 actor="pipeline:cjm-transcript-decomp-core", run=run_id,
                                 args={"method": "alignment-fold/v1", "act": "chunk-respine"})
        vr = await verify_source(queue, graph_id, source_id, [ctx["rendition"]], segment_ids=new_ids)
        _journal_run_event(manager, SubstrateEventType.VERIFY_OUTCOME.value, run_id, actor, {
            "core": "cjm-transcript-decomp-core", "kind": "respine-chunk", "source_node_id": source_id,
            "found": vr is not None, "ok": (vr.ok if vr is not None else False)})

        # 4. TRANSFER the two source-truth classes by time (the correction core's handler).
        carried: List[str] = []
        straddles: List[str] = []
        transfer: Dict[str, Any] = {}
        if transferable and handler is not None:
            transfer = await handler(queue, graph_id, source_id=source_id, old_segment_ids=old_ids,
                                     new_segment_ids=new_ids, journal_path=journal_path, actor=actor,
                                     run_id=run_id) or {}
            carried = [str(c) for c in (transfer.get("carried") or [])]
            straddles = [str(s) for s in (transfer.get("straddles") or [])]
        stranded_ids = [str(c["id"]) for c in dep["stranded"]]
        if transferable and handler is None:
            stranded_ids += [str(c["id"]) for c in dep["events"] + dep["speakers"]]

        # 5. ONE FACT: stamps + renumber + the Source's record.
        entry = RespinedChunkEntry(audio_segment=ctx["audio_segment"], transcript=transcript_id, run=run_id,
                                   op_id=op_id, ts=time.time(), old_segments=old_ids, new_segments=new_ids,
                                   stranded=stranded_ids, carried=carried, straddles=straddles)
        op = build_respine_op(source_id=source_id, entry=entry, skeleton_hash=skeleton_hash,
                              renumber=renumber, existing_entries=existing_entries, actor=actor)
        await apply_spine_fact(queue, graph_id, op)
        if journal_path:
            append_op(journal_path, op, dedup=False)

        # 6. The chunk-scoped decomp manifest (0.2.7): a run row the decomp app paints as ⤷ respine.
        manifest = DecompManifest(
            run_id=run_id, created_at=time.time(), config=cfg.to_dict(),
            source_manifest=str(derived_path), source_format=str(derived.get("format") or ""),
            source_version=str(derived.get("version") or ""),
            capabilities={**{k: v for k, v in recorded.items() if k in stack}, graph_id: (recorded.get(graph_id) or {})},
            skeleton_config_hash=skeleton_hash, split_policy=dm.get("split_policy"),
            event_split_policy=dm.get("event_split_policy"), word_rescue_policy=dm.get("word_rescue_policy"),
            event_propset_id=str(ctx["decomp_record"].get("event_propset_id") or ""),
            event_propset=str(ctx["decomp_record"].get("event_propset") or ""),
            parent_run_id=str(dm.get("run_id") or ""), respined_chunk=entry.to_dict())
        manifest.sources.append(DecompSourceRecord(
            source_node_id=source_id, source_path=str(d_src.get("source_path") or ""),
            title=str(ctx["decomp_record"].get("title") or Path(str(d_src.get("source_path") or "")).stem),
            segment_count=len(new_ids), segment_ids=new_ids, skeleton_config_hash=skeleton_hash,
            event_propset_id=str(ctx["decomp_record"].get("event_propset_id") or ""),
            event_propset=str(ctx["decomp_record"].get("event_propset") or "")))
        decomp_path = manifest.save(runs_dir / f"{run_id}.json", workspace=workspace)
        _journal_run_event(manager, SubstrateEventType.RUN_FINISHED.value, run_id, actor, {
            "core": "cjm-transcript-decomp-core", "kind": "respine-chunk", "status": "completed",
            "source_id": source_id, "audio_segment": ctx["audio_segment"], "transcript": transcript_id,
            "old_segments": len(old_ids), "new_segments": len(new_ids), "renumbered": len(renumber),
            "stranded": len(stranded_ids), "carried": len(carried), "straddles": len(straddles),
            "manifest": str(decomp_path)})
    except BaseException as e:
        _journal_run_event(manager, SubstrateEventType.RUN_FINISHED.value, run_id, actor, {
            "core": "cjm-transcript-decomp-core", "kind": "respine-chunk", "status": "failed",
            "error": repr(e), "source_id": source_id, "audio_segment": ctx["audio_segment"]})
        raise
    readout.update({"run_id": run_id, "landing_run_id": land_run, "transcript": transcript_id,
                    "supersedes": rec.get("supersedes"), "op_id": op_id, "new_segments": len(new_ids),
                    "first_index": first, "delta": delta, "renumbered": len(renumber),
                    "carried": len(carried), "straddles": len(straddles), "stranded": len(stranded_ids),
                    "transfer": transfer, "verify_ok": (vr.ok if vr is not None else None),
                    "nodes_added": res.nodes_added, "nodes_verified": res.nodes_verified,
                    "derived_manifest": str(derived_path), "decomp_manifest": str(decomp_path),
                    "new_segment_ids": new_ids, "old_segment_ids": old_ids, "noop": False})
    return readout
