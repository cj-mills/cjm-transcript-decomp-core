"""Graph-spine EXTENSION + skeptical-lens verification (stage 5, CR-18 revolution 2). Decomp no longer creates a Document: it RECOMPUTES the transcription-emitted root's deterministic node ids from the consumed manifest (no search), verifies the root exists, and attaches the fine Segment spine under the existing AudioSegment nodes — PART_OF to the owning rendition, STARTS_WITH per rendition (the coarse-seam jump anchor), source-wide NEXT. Each Segment carries the audio TimeSlice ref plus per-transcriber CharSlice refs into the Transcript nodes (the D4/P10 framing, finally expressible). Commit goes through the layer's idempotent extend_graph."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from cjm_context_graph_layer.grammar import grouped_spine_edges, SpineRelations
from cjm_context_graph_layer.ops import graph_task
from cjm_context_graph_primitives.graph import GraphNode
from cjm_context_graph_primitives.provenance import SourceRef
from cjm_context_graph_primitives.query import (EdgeQuery, EdgeQueryResult, NodeQuery,
                                                NodeQueryResult, OrderBy, PropertyPredicate,
                                                RelationPredicate)
from cjm_substrate.core.queue import JobQueue
from cjm_transcript_decomp_core.models import DecompSegment
from cjm_transcript_graph_schema.schema import (audio_rendition_node_id, audio_segment_node_id,
                                                SegmentNode, source_node_id, transcript_node_id,
                                                TranscriptGraphLabels, TranscriptSliceRef)

# Stage 4: typed query expressions + results — importing the result classes IS
# the host-side wire registration (F8); the tuple keeps these SIDE-EFFECT
# imports referenced so the canonical emit cannot prune them. Stage 5: the
# graph-aware layer owns the shared plumbing (graph_task is THE shared copy
# now) and the fractal spine grammar; the audio-transcript schema owns the
# node nouns + deterministic identity tuples. cjm_graph_domains is GONE
# (Document dissolved).
_REGISTERED_WIRE_KINDS = (NodeQueryResult, EdgeQueryResult)


def resolve_root_ids(
    source_entry: Dict[str, Any],            # One source entry from the transcription manifest (0.3.0)
    capabilities_info: Dict[str, Dict[str, Any]], # The transcription manifest's capabilities block (config hashes)
) -> Dict[str, Any]:  # {"source", "chain", "audio_segments": [{audio_segment, rendition, start, end, model_input_hash, transcripts}]}
    """Recompute the transcription-emitted root node ids from manifest data.

    Deterministic identity makes the extender lookup a RECOMPUTATION, not a
    search: Source = content hash; AudioSegment = (source, boundary range);
    AudioRendition = (audio segment, preprocessing chain) — the per-source
    `chain` ([] = raw convert-only) carried by the 0.3.0 manifest; Transcript =
    (rendition, transcriber, config_hash). No stored-id coupling between the
    cores. A pre-0.3.0 manifest (no `chain`) recomputes the raw rendition ids
    (empty chain), which is correct for any non-preprocessed run.
    """
    content_hash = str(source_entry.get("content_hash") or "")
    if not content_hash:
        raise ValueError(
            f"source {source_entry.get('source_path')!r} has no content_hash — "
            "re-run transcription on the 0.3.0 manifest schema")
    source_id = source_node_id(content_hash)
    chain = list(source_entry.get("chain") or [])  # [] = raw convert-only rendition
    asegs: List[Dict[str, Any]] = []
    for pseg in source_entry.get("segments") or []:
        start, end = float(pseg.get("start", 0.0)), float(pseg.get("end", 0.0))
        aseg_id = audio_segment_node_id(source_id, start, end)
        rendition_id = audio_rendition_node_id(aseg_id, chain)
        transcripts = {
            t: transcript_node_id(rendition_id, t, str((capabilities_info.get(t) or {}).get("config_hash") or ""))
            for t in (pseg.get("transcripts") or {})
        }
        asegs.append({"audio_segment": aseg_id, "rendition": rendition_id,
                      "start": start, "end": end,
                      "model_input_hash": str(pseg.get("model_input_hash") or ""),
                      "transcripts": transcripts})
    return {"source": source_id, "chain": chain, "audio_segments": asegs}


def build_extension_payload(
    source_entry: Dict[str, Any],             # One source entry from the transcription manifest
    capabilities_info: Dict[str, Dict[str, Any]],  # The transcription manifest's capabilities block
    skeleton_config_hash: str,                # THIS run's skeleton-config hash (Segment identity input; = the VAD config hash when no split stage ran)
    text_from: str,                           # Authoritative transcriber (layer-0 text designation)
    segments: List[DecompSegment],            # Ordered aligned segments (per-transcriber variants attached)
    split_policy: Optional[str] = None,       # Split policy+version that refined the skeleton (node metadata, never identity)
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:  # (nodes, edges, ids)
    """Build the fine-spine EXTENSION payload (pure; no capability calls).

    Segment identity is audio-side (audio RENDITION, VAD config, chunk range) —
    shared across transcribers by construction, distinct per rendition (vocals
    isolation can yield different VAD chunking than raw). Each Segment carries
    the audio `TimeSlice` ref into its rendition + one `CharSlice` ref per
    transcriber variant; `text_from` is recorded per segment as provenance
    (which Transcript the layer-0 text came from), never as global config. Edges
    come from `grouped_spine_edges`: PART_OF to the OWNING AudioRendition,
    STARTS_WITH per rendition, NEXT chained source-wide across coarse boundaries.
    """
    roots = resolve_root_ids(source_entry, capabilities_info)
    source_id = roots["source"]
    asegs = roots["audio_segments"]

    nodes: List[Dict[str, Any]] = []
    seg_ids: List[str] = []
    groups: Dict[int, List[str]] = {}
    used_transcripts: set = set()
    for seg in segments:
        a = asegs[seg.pseg_index]
        slices: List[TranscriptSliceRef] = []
        for var in seg.variants:
            tid = a["transcripts"].get(var.transcriber)
            if tid is None or not var.text or var.start_char is None or var.end_char is None:
                continue
            slices.append(TranscriptSliceRef(transcript=tid, start_char=var.start_char,
                                             end_char=var.end_char, text=var.text))
            used_transcripts.add(tid)
        node = SegmentNode(
            rendition=a["rendition"], vad_config_hash=skeleton_config_hash,
            chunk_start=seg.chunk_start, chunk_end=seg.chunk_end,
            index=seg.index, start_time=seg.start_time, end_time=seg.end_time,
            text=seg.text, audio_hash=a["model_input_hash"], source=source_id,
            text_from=(a["transcripts"].get(text_from) if seg.text.strip() else None),
            split_policy=split_policy, text_slices=slices,
        )
        nodes.append(node.to_graph_node())
        seg_ids.append(node.id)
        groups.setdefault(seg.pseg_index, []).append(node.id)

    # The fine spine hangs under each segment's RENDITION (PART_OF rendition).
    edges = grouped_spine_edges([(asegs[i]["rendition"], groups[i]) for i in sorted(groups)])
    ids = {"source": source_id, "segments": seg_ids,
           "audio_segments": [a["audio_segment"] for a in asegs],
           "renditions": [a["rendition"] for a in asegs],
           "transcripts_used": sorted(used_transcripts)}
    return nodes, edges, ids


@dataclass
class SourceVerification:
    """Skeptical-lens verification of one Source's fine-spine extension under a
    specific rendition set, computed by querying the graph directly (never
    trusting run state)."""
    source_id: str             # Verified Source node id
    title: str                 # Source title (read back from the graph)
    audio_segment_count: int   # AudioSegment nodes found under the Source (coarse spine; rendition-independent)
    rendition_count: int       # AudioRendition nodes the fine spine was scoped to (the run's extended renditions)
    segment_count: int         # Fine Segment nodes found under those renditions
    source_starts_with: bool   # Exactly 1 STARTS_WITH Source -> first AudioSegment
    aseg_next_complete: bool   # Coarse NEXT count == audio_segment_count - 1
    seg_next_complete: bool    # Fine NEXT count == segment_count - 1 (source-wide chain)
    part_of_complete: bool     # Fine PART_OF count == segment_count (Segment -> rendition)
    rendition_starts_with_complete: bool  # STARTS_WITH from renditions == #renditions owning >=1 Segment
    all_have_timing: bool      # Every Segment has start_time + end_time
    all_have_sources: bool     # Every Segment has >=1 SourceRef (the audio ref at minimum)
    source_locators: List[str] = field(default_factory=list)  # Distinct provenance locator URIs

    @property
    def ok(self) -> bool:  # True when every structural check passes
        """All structural checks pass."""
        return (self.source_starts_with and self.aseg_next_complete
                and self.seg_next_complete and self.part_of_complete
                and self.rendition_starts_with_complete and self.all_have_timing
                and self.all_have_sources)


async def verify_source(
    queue: JobQueue,          # Started job queue
    graph_id: str,            # Graph-storage capability id
    source_id: str,           # Source node id to verify
    rendition_ids: List[str], # The AudioRendition ids the run extended (the fine spine is scoped to these)
    segment_ids: Optional[List[str]] = None,  # THIS run's committed Segment ids — id-scopes the fine-layer checks (parallel spines); None = all segments under the renditions
) -> Optional[SourceVerification]:  # Result, or None if the Source is not found
    """Verify a Source's committed extension via server-side AGGREGATES (D13/D19).

    Two-layer verification for the Source-rooted schema: the coarse spine
    (Source → AudioSegments) is checked with id-list edge counts (rendition-
    independent); the fine spine hangs under the run's AudioRendition set, so it
    is scoped through the batched far-end constraint
    `RelationPredicate("PART_OF", node_ids=rendition_ids)` (the C17 batch shape) —
    never a whole-neighborhood materialization, and never mixing a sibling
    rendition's spine (raw vs vocals coexist under the same AudioSegments). One
    bounded projection pass (sources + rendition_id) yields missing-sources,
    distinct locators, AND the per-rendition ownership set for the STARTS_WITH
    check.
    """
    src = await graph_task(queue, graph_id, "get_node", node_id=source_id)
    if src is None:
        return None
    props = src.properties if isinstance(src, GraphNode) else ((src or {}).get("properties") or {})

    async def _ecount(**kw):
        q = EdgeQuery(count=True, **kw)
        res = await graph_task(queue, graph_id, "query_edges", query=q.to_dict())
        return int(res.count or 0)

    # Coarse layer: ordered AudioSegment ids under the Source (rendition-independent).
    aq = NodeQuery(label=TranscriptGraphLabels.AUDIO_SEGMENT,
                   related=RelationPredicate(SpineRelations.PART_OF, node_id=source_id),
                   order_by=OrderBy("index"), project=["index"])
    ares = await graph_task(queue, graph_id, "query_nodes", query=aq.to_dict())
    aseg_ids = [r["id"] for r in (ares.rows or [])]
    n_asegs = len(aseg_ids)
    rendition_ids = list(rendition_ids or [])
    if not aseg_ids or not rendition_ids:
        return SourceVerification(
            source_id=source_id, title=props.get("title", "Untitled"),
            audio_segment_count=n_asegs, rendition_count=len(rendition_ids),
            segment_count=0, source_starts_with=False,
            aseg_next_complete=False, seg_next_complete=False, part_of_complete=False,
            rendition_starts_with_complete=False, all_have_timing=False, all_have_sources=False)

    src_starts = await _ecount(relation_type=SpineRelations.STARTS_WITH, source_id=source_id)
    aseg_next = await _ecount(relation_type=SpineRelations.NEXT, source_ids=aseg_ids)

    # Fine layer, scoped via the batched far-end constraint onto THIS run's renditions.
    part_of_rend = RelationPredicate(SpineRelations.PART_OF, node_ids=rendition_ids)

    async def _ncount(**kw):
        q = NodeQuery(label=TranscriptGraphLabels.SEGMENT, related=part_of_rend, count=True, **kw)
        res = await graph_task(queue, graph_id, "query_nodes", query=q.to_dict())
        return int(res.count or 0)

    proj_rows: List[Dict[str, Any]] = []
    if segment_ids is None:
        n_segs = await _ncount()
        seg_next = await _ecount(relation_type=SpineRelations.NEXT, source_related=part_of_rend)
        seg_part_of = await _ecount(relation_type=SpineRelations.PART_OF, target_ids=rendition_ids)
        rend_starts = await _ecount(relation_type=SpineRelations.STARTS_WITH, source_ids=rendition_ids)
        missing_start = await _ncount(where=[PropertyPredicate("start_time", "is_null")])
        missing_end = await _ncount(where=[PropertyPredicate("end_time", "is_null")])
        # Bounded projection: sources + owning rendition in ONE pass.
        sq = NodeQuery(label=TranscriptGraphLabels.SEGMENT, related=part_of_rend,
                       project=["sources", "rendition_id"])
        res = await graph_task(queue, graph_id, "query_nodes", query=sq.to_dict())
        proj_rows = list(res.rows or [])
    else:
        # Id-scoped mode (parallel spines, DEC f1024568): every fine-layer
        # aggregate counts ONLY this run's committed segments — a coexisting
        # sibling skeleton under the same renditions (its own NEXT chain +
        # STARTS_WITH anchor) must not fail the structural checks. Batched
        # (bounded reads; the 500-id convention).
        async def _icount(ids, **kw):
            q = NodeQuery(ids=list(ids), label=TranscriptGraphLabels.SEGMENT,
                          count=True, **kw)
            res = await graph_task(queue, graph_id, "query_nodes", query=q.to_dict())
            return int(res.count or 0)

        n_segs = seg_next = seg_part_of = rend_starts = missing_start = missing_end = 0
        for i in range(0, len(segment_ids), 500):
            b = list(segment_ids[i:i + 500])
            n_segs += await _icount(b)
            seg_next += await _ecount(relation_type=SpineRelations.NEXT, source_ids=b)
            seg_part_of += await _ecount(relation_type=SpineRelations.PART_OF, source_ids=b)
            rend_starts += await _ecount(relation_type=SpineRelations.STARTS_WITH, target_ids=b)
            missing_start += await _icount(b, where=[PropertyPredicate("start_time", "is_null")])
            missing_end += await _icount(b, where=[PropertyPredicate("end_time", "is_null")])
            pq = NodeQuery(ids=b, label=TranscriptGraphLabels.SEGMENT,
                           project=["sources", "rendition_id"])
            res = await graph_task(queue, graph_id, "query_nodes", query=pq.to_dict())
            proj_rows.extend(res.rows or [])

    missing_sources = 0
    locators = set()
    owning_renditions = set()
    for r in proj_rows:
        sources = r.get("sources") or []
        if not sources:
            missing_sources += 1
        for s in sources:
            ref = SourceRef.from_dict(s) if isinstance(s, dict) else s
            locators.add(ref.locator.to_uri())
        if r.get("rendition_id"):
            owning_renditions.add(r["rendition_id"])

    return SourceVerification(
        source_id=source_id,
        title=props.get("title", "Untitled"),
        audio_segment_count=n_asegs,
        rendition_count=len(rendition_ids),
        segment_count=n_segs,
        source_starts_with=src_starts == 1,
        aseg_next_complete=aseg_next == max(0, n_asegs - 1),
        seg_next_complete=seg_next == max(0, n_segs - 1),
        part_of_complete=seg_part_of == n_segs,
        rendition_starts_with_complete=rend_starts == len(owning_renditions),
        all_have_timing=(missing_start == 0 and missing_end == 0),
        all_have_sources=missing_sources == 0,
        source_locators=sorted(locators),
    )
