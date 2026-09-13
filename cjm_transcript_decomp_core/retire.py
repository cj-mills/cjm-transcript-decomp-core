"""Spine RETIREMENT + COMPACTION — safe removal of superseded decomposition spines (ruling a7617bd4, item eaefebd2).

Every decomposition run mints a parallel fine spine per source (a distinct skeleton hash);
nothing ever removed one, so 273 sources carried 723 spines with 645k segments in spines
nothing references (census 2026-09-13). Two verbs, three rules:

  `retire`  — a journaled FACT (`spine-retire`): the Source node's `retired_spines` map gains
              an entry {skeleton_hash: {reason, successor, actor, ts}}; pickers, loaders and
              the census filter on it. Reversible (`unretire`), immediate, no bytes move.
  `compact` — a separate maintenance verb over ALREADY-RETIRED spines: their wires move out
              of the active journal family into the archive (`cjm_context_graph_layer.compact`),
              their nodes are deleted from the live db, and a `spine-compaction` fact records
              where they went. Provenance kept, nothing lost, the workspace shrinks.

  Rule (a) the refusal is NOT "is the latest spine" — creation order is not preference — it is
           "would leave the source with ZERO live spines"; any spine may retire at any age and
           the successor may be ANY live spine, an older one included.
  Rule (b) refuse with a LISTING when dependents exist (corrections, reviews, sessions bound to
           the spine's segments): never a blind cascade — transfer them or retire them first.
  Rule (c) the default spine selection stops reading creation order: the most recently declared
           successor, else the newest live spine (`default_live_spine`).

Replay: both verbs are property-merge ops on the Source (`updates`), applied by
`apply_spine_fact`; after a compaction the archived wires are simply absent from the family,
so a rebuild converges without ever materialising them.
"""

import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

from cjm_context_graph_layer.compact import CompactReport, compact_journal, scan_references
from cjm_context_graph_layer.grammar import OverlayRelations, SpineRelations
from cjm_context_graph_layer.ops import graph_task
from cjm_context_graph_primitives.graph import GraphNode
from cjm_context_graph_primitives.journal import append_op
from cjm_context_graph_primitives.query import EdgeQuery, NodeQuery, PropertyPredicate, RelationPredicate
from cjm_transcript_graph_schema.schema import TranscriptGraphLabels

logger = logging.getLogger(__name__)

RETIRE_VERB = "spine-retire"           # The retirement fact op (retire / unretire)
COMPACTION_VERB = "spine-compaction"   # The compaction fact op (where the wires went)
RETIRED_SPINES_PROP = "retired_spines" # Source-node property: {spine key: entry}
LEGACY_KEY = "legacy"                  # Map key for the pre-split spine (skeleton_hash None)

# Correction-overlay relations the dependents gate reads. Literal on purpose: the decomp core
# does not depend on the correction core; these mirror CorrectionRelations.CORRECTS / REVIEWED.
_CORRECTS = "CORRECTS"
_REVIEWED = "REVIEWED"
_ID_BATCH = 500  # ids per batched edge query (SQLite bind-variable headroom)


# ---- pure: keys, rows, rules ---------------------------------------------------------------

def spine_key(skeleton_hash: Optional[str]) -> str:  # The retired_spines map key for a spine
    return skeleton_hash or LEGACY_KEY


def retired_spines(source_props: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """The Source's retirement map (pure): {spine key: {reason, successor, actor, ts[, compacted]}}."""
    m = (source_props or {}).get(RETIRED_SPINES_PROP) or {}
    return {str(k): dict(v) for k, v in m.items() if isinstance(v, dict)}


def annotate_spines(
    spines: List[Dict[str, Any]],       # Spine rows ({"skeleton_hash", "split_policy", "segments", "created_at"})
    retired: Dict[str, Dict[str, Any]], # The Source's retirement map
) -> List[Dict[str, Any]]:  # The same rows + retired / retired_reason / successor / compacted
    """Mark each spine row with its retirement state (pure; rows copied)."""
    out = []
    for s in spines:
        r = dict(s)
        e = retired.get(spine_key(s.get("skeleton_hash")))
        r["retired"] = e is not None
        r["retired_reason"] = (e or {}).get("reason")
        r["successor"] = (e or {}).get("successor")
        r["compacted"] = bool((e or {}).get("compacted"))
        out.append(r)
    return out


def live_spines(spines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:  # Rows not retired
    return [s for s in spines if not s.get("retired")]


def resolve_spine(
    spines: List[Dict[str, Any]],  # Spine rows (annotated or not)
    selector: str,                 # "legacy" | a skeleton hash (full, or a case-insensitive prefix of it / its hex tail)
) -> Dict[str, Any]:  # The one matching row
    """Resolve a picker-style selector to exactly one spine row (pure); refuses with the roster."""
    sel = (selector or "").strip().lower()
    if not sel:
        raise ValueError("empty spine selector")
    if sel == LEGACY_KEY:
        hits = [s for s in spines if not s.get("skeleton_hash")]
    else:
        hits = [s for s in spines if s.get("skeleton_hash") and
                (str(s["skeleton_hash"]).lower().startswith(sel)
                 or str(s["skeleton_hash"]).split(":")[-1].lower().startswith(sel))]
    if len(hits) != 1:
        raise ValueError(f"spine selector {selector!r} matches {len(hits)} spine(s) "
                         f"(available: {[spine_label(s) for s in spines]})")
    return hits[0]


def spine_label(spine: Dict[str, Any]) -> str:  # "policy · hash8 [· retired]"
    h = spine.get("skeleton_hash")
    tag = spine.get("split_policy") or ("vad-only" if h else "vad-only (pre-split)")
    base = f"{tag} · {str(h).split(':')[-1][:8]}" if h else tag
    return base + (" · retired" if spine.get("retired") else "")


def default_live_spine(
    spines: List[Dict[str, Any]],  # Annotated spine rows
) -> Optional[Dict[str, Any]]:  # Rule (c): the most recently declared successor, else the newest live spine
    """Which spine opens by default (pure). Preference is a DECLARED fact, never creation order:
    the successor named by the most recent retirement wins when it is live; otherwise the
    newest live spine (by created_at); None when every spine is retired."""
    live = live_spines(spines)
    if not live:
        return None
    by_key = {spine_key(s.get("skeleton_hash")): s for s in live}
    declared = sorted(((float(s.get("retired_ts") or 0.0), s.get("successor"))
                       for s in spines if s.get("retired") and s.get("successor")), reverse=True)
    for _, succ in declared:
        if succ in by_key:
            return by_key[succ]
    return max(live, key=lambda s: float(s.get("created_at") or 0.0))


def plan_retire(
    spines: List[Dict[str, Any]],        # The source's spine rows (any annotation ignored; the map decides)
    retired: Dict[str, Dict[str, Any]],  # The Source's current retirement map
    selector: str,                       # Which spine (resolve_spine semantics)
    *,
    reason: str = "",                    # Why (journaled)
    successor: Optional[str] = None,     # Selector of the spine that takes over (any LIVE spine, older ok)
    unretire: bool = False,              # Reverse a retirement instead
    actor: str = "",                     # Who
    ts: Optional[float] = None,          # When (default now)
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:  # (the NEW full map, {"key", "act", "spine", "entry"})
    """Rules (a) + successor validation as a pure plan; the graph write is `journal_spine_retire`."""
    target = resolve_spine(spines, selector)
    key = spine_key(target.get("skeleton_hash"))
    new_map = {k: dict(v) for k, v in retired.items()}
    if unretire:
        if key not in new_map:
            raise ValueError(f"spine {spine_label(target)} is not retired")
        entry = new_map.pop(key)
        if entry.get("compacted"):
            raise ValueError(f"spine {spine_label(target)} was COMPACTED ({entry['compacted'].get('archive')}) "
                             "— its wires live in the archive; restore is a journal-side act, not an unretire")
        return new_map, {"key": key, "act": "unretire", "spine": target, "entry": entry}
    if key in new_map:
        raise ValueError(f"spine {spine_label(target)} is already retired "
                         f"({new_map[key].get('reason') or 'no reason recorded'})")
    remaining = [s for s in spines if spine_key(s.get("skeleton_hash")) not in new_map
                 and spine_key(s.get("skeleton_hash")) != key]
    if not remaining:
        raise ValueError(f"retiring {spine_label(target)} would leave the source with NO live spine "
                         "(rule (a)); retire the source instead, or decompose a successor first")
    succ_key: Optional[str] = None
    if successor:
        succ = resolve_spine(remaining, successor)
        succ_key = spine_key(succ.get("skeleton_hash"))
        if succ_key == key:
            raise ValueError("a spine cannot succeed itself")
    entry = {"reason": reason or "", "successor": succ_key, "actor": actor,
             "ts": float(ts if ts is not None else time.time())}
    new_map[key] = entry
    return new_map, {"key": key, "act": "retire", "spine": target, "entry": entry}


# ---- graph reads ---------------------------------------------------------------------------

def _rows(res: Any) -> List[Dict[str, Any]]:
    return list(getattr(res, "rows", None) or [])


async def get_source(queue: Any, graph_id: str, source_id: str) -> Optional[Dict[str, Any]]:
    """The Source node as a dict (None when absent)."""
    node = await graph_task(queue, graph_id, "get_node", node_id=source_id)
    if node is None:
        return None
    return node.to_dict() if isinstance(node, GraphNode) else dict(node)


async def resolve_source_id(
    queue: Any, graph_id: str,
    selector: str,  # A Source id, an id prefix, or a case-insensitive title substring
) -> Tuple[str, str]:  # (source id, title)
    """Resolve a human selector to ONE Source (refuses with the candidates)."""
    q = NodeQuery(label=TranscriptGraphLabels.SOURCE, project=["title"])
    rows = _rows(await graph_task(queue, graph_id, "query_nodes", query=q.to_dict()))
    sel = selector.strip()
    hits = [r for r in rows if r["id"] == sel or r["id"].startswith(sel)]
    if not hits:
        hits = [r for r in rows if sel.lower() in str(r.get("title") or "").lower()]
    if len(hits) != 1:
        names = [f"{r['id'][:8]} {str(r.get('title') or '')[:48]}" for r in hits[:12]]
        raise ValueError(f"source selector {selector!r} matches {len(hits)} source(s): {names}")
    return hits[0]["id"], str(hits[0].get("title") or "")


async def list_sources(
    queue: Any, graph_id: str,
    collection: Optional[str] = None,  # Restrict to one Collection title (exact, case-insensitive)
) -> List[Tuple[str, str]]:  # [(source id, title)]
    """Every Source (optionally one collection's members)."""
    if collection:
        cq = NodeQuery(label=TranscriptGraphLabels.COLLECTION, project=["title"])
        cols = [r for r in _rows(await graph_task(queue, graph_id, "query_nodes", query=cq.to_dict()))
                if str(r.get("title") or "").lower() == collection.lower()]
        if len(cols) != 1:
            raise ValueError(f"collection {collection!r} matches {len(cols)} collection(s)")
        q = NodeQuery(label=TranscriptGraphLabels.SOURCE, project=["title"],
                      related=RelationPredicate(SpineRelations.PART_OF, node_id=cols[0]["id"]))
    else:
        q = NodeQuery(label=TranscriptGraphLabels.SOURCE, project=["title"])
    rows = _rows(await graph_task(queue, graph_id, "query_nodes", query=q.to_dict()))
    return sorted(((r["id"], str(r.get("title") or "")) for r in rows), key=lambda x: x[1])


async def source_rendition_ids(queue: Any, graph_id: str, source_id: str) -> List[str]:
    """Every AudioRendition under a Source (all chains — retirement is per skeleton, not per chain)."""
    aq = NodeQuery(label=TranscriptGraphLabels.AUDIO_SEGMENT, project=[],
                   related=RelationPredicate(SpineRelations.PART_OF, node_id=source_id))
    asegs = [r["id"] for r in _rows(await graph_task(queue, graph_id, "query_nodes", query=aq.to_dict()))]
    if not asegs:
        return []
    rq = NodeQuery(label=TranscriptGraphLabels.AUDIO_RENDITION, project=[],
                   related=RelationPredicate(OverlayRelations.DERIVED_FROM, node_ids=asegs))
    return [r["id"] for r in _rows(await graph_task(queue, graph_id, "query_nodes", query=rq.to_dict()))]


async def list_spines(
    queue: Any, graph_id: str, source_id: str,
    rendition_ids: Optional[List[str]] = None,  # Pre-resolved renditions (default: all under the source)
) -> List[Dict[str, Any]]:  # Annotated spine rows, sorted by created_at
    """The source's coexisting spines, grouped by skeleton hash and annotated with the retirement map."""
    rends = rendition_ids if rendition_ids is not None else await source_rendition_ids(queue, graph_id, source_id)
    if not rends:
        return []
    q = NodeQuery(label=TranscriptGraphLabels.SEGMENT, project=["skeleton_hash", "split_policy", "created_at"],
                  related=RelationPredicate(SpineRelations.PART_OF, node_ids=list(rends)))
    groups: Dict[Optional[str], Dict[str, Any]] = {}
    for r in _rows(await graph_task(queue, graph_id, "query_nodes", query=q.to_dict())):
        key = r.get("skeleton_hash")
        g = groups.setdefault(key, {"skeleton_hash": key, "split_policy": None, "segments": 0, "created_at": 0.0})
        g["segments"] += 1
        if r.get("split_policy"):
            g["split_policy"] = r["split_policy"]
        c = float(r.get("created_at") or 0.0)
        if c and (not g["created_at"] or c < g["created_at"]):
            g["created_at"] = c
    src = await get_source(queue, graph_id, source_id)
    retired = retired_spines((src or {}).get("properties"))
    rows = annotate_spines(sorted(groups.values(), key=lambda g: g["created_at"]), retired)
    for r in rows:  # ride the retirement stamp so default_live_spine can order declarations
        e = retired.get(spine_key(r.get("skeleton_hash")))
        r["retired_ts"] = (e or {}).get("ts")
    return rows


async def spine_segment_ids(
    queue: Any, graph_id: str,
    rendition_ids: List[str],        # The source's renditions
    skeleton_hash: Optional[str],    # The spine (None = legacy, prop absent)
    page: int = 5000,
) -> List[str]:  # Every Segment id on that spine
    where = ([PropertyPredicate("skeleton_hash", "eq", skeleton_hash)] if skeleton_hash
             else [PropertyPredicate("skeleton_hash", "is_null")])
    ids: List[str] = []
    offset = 0
    while True:
        q = NodeQuery(label=TranscriptGraphLabels.SEGMENT, project=[], where=where,
                      related=RelationPredicate(SpineRelations.PART_OF, node_ids=list(rendition_ids)),
                      limit=page, offset=offset)
        rows = _rows(await graph_task(queue, graph_id, "query_nodes", query=q.to_dict()))
        ids.extend(r["id"] for r in rows)
        if len(rows) < page:
            return ids
        offset += len(rows)


async def spine_dependents(
    queue: Any, graph_id: str,
    segment_ids: List[str],  # The spine's segments
) -> Dict[str, Any]:  # {"corrections": n, "reviews": n, "sessions": [ids], "segments_touched": n}
    """Rule (b)'s evidence: what on the graph points at the spine's segments."""
    corrections = 0
    reviews = 0
    sessions: set = set()
    touched: set = set()
    for i in range(0, len(segment_ids), _ID_BATCH):
        batch = segment_ids[i:i + _ID_BATCH]
        cq = EdgeQuery(relation_type=_CORRECTS, target_ids=batch, project=[])
        for r in _rows(await graph_task(queue, graph_id, "query_edges", query=cq.to_dict())):
            corrections += 1
            touched.add(r.get("target_id"))
        rq = EdgeQuery(relation_type=_REVIEWED, target_ids=batch, project=[])
        for r in _rows(await graph_task(queue, graph_id, "query_edges", query=rq.to_dict())):
            reviews += 1
            sessions.add(r.get("source_id"))
            touched.add(r.get("target_id"))
    return {"corrections": corrections, "reviews": reviews,
            "sessions": sorted(s for s in sessions if s), "segments_touched": len(touched)}


def dependents_free(dep: Dict[str, Any]) -> bool:
    return not dep.get("corrections") and not dep.get("reviews") and not dep.get("sessions")


# The whole graph's dependents in ONE read, grouped by (source, spine): the typed surface
# would cost ~10 batched edge queries per spine (~4k queue jobs over the 350-spine GPU MODE
# sweep, measured 2026-09-13); the join from the 77k-row correction-edge side to Segments
# by primary key is sub-second. Same SG-41 raw_query escape the runaway census rides
# (typed-surface promotion candidate: an edge count grouped by a target property).
DEPENDENTS_SQL = """
SELECT json_extract(n.properties, '$.source_id') AS source_id,
       json_extract(n.properties, '$.skeleton_hash') AS skeleton_hash,
       SUM(CASE WHEN e.relation_type = 'CORRECTS' THEN 1 ELSE 0 END) AS corrections,
       SUM(CASE WHEN e.relation_type = 'REVIEWED' THEN 1 ELSE 0 END) AS reviews,
       COUNT(DISTINCT CASE WHEN e.relation_type = 'REVIEWED' THEN e.source_id END) AS sessions,
       COUNT(DISTINCT n.id) AS segments_touched
FROM edges e JOIN nodes n ON n.id = e.target_id
WHERE e.relation_type IN ('CORRECTS', 'REVIEWED') AND n.label = 'Segment'
GROUP BY 1, 2
"""


async def dependents_map(
    queue: Any, graph_id: str,
) -> Dict[Tuple[str, str], Dict[str, Any]]:  # (source id, spine key) -> spine_dependents shape
    """Every spine's dependents in one raw read (see DEPENDENTS_SQL); a spine absent from
    the map has none. Session ids are not enumerated here (count only) — `spine_dependents`
    lists them for the single-spine refusal message."""
    res = await graph_task(queue, graph_id, "raw_query",
                           query={"type": "raw_query", "text": DEPENDENTS_SQL, "backend": "sqlite", "params": []})
    cols = list(res.get("columns") or []) if isinstance(res, dict) else list(getattr(res, "columns", []))
    rows = list(res.get("rows") or []) if isinstance(res, dict) else list(getattr(res, "rows", []))
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in rows:
        d = dict(zip(cols, r))
        out[(str(d.get("source_id")), spine_key(d.get("skeleton_hash")))] = {
            "corrections": int(d.get("corrections") or 0), "reviews": int(d.get("reviews") or 0),
            "sessions": [f"{int(d.get('sessions') or 0)} session(s)"] if int(d.get("sessions") or 0) else [],
            "segments_touched": int(d.get("segments_touched") or 0)}
    return out


def _dep_for(dep_map: Optional[Dict[Tuple[str, str], Dict[str, Any]]], source_id: str, key: str) -> Optional[Dict[str, Any]]:
    """A spine's dependents from a prefetched map (None when no map was given)."""
    if dep_map is None:
        return None
    return dep_map.get((source_id, key)) or {"corrections": 0, "reviews": 0, "sessions": [], "segments_touched": 0}


# ---- journaled writes + replay --------------------------------------------------------------

async def apply_spine_fact(queue: Any, graph_id: str, op: Dict[str, Any]) -> None:
    """Replay handler for spine-retire / spine-compaction: property merges on the Source."""
    for u in op.get("updates") or []:
        await graph_task(queue, graph_id, "update_node", node_id=u["id"], properties=dict(u["properties"]))


def spine_fact_handlers() -> Dict[str, Any]:  # verb -> handler, for decomp_replay_handlers
    return {RETIRE_VERB: apply_spine_fact, COMPACTION_VERB: apply_spine_fact}


async def journal_spine_retire(
    queue: Any, graph_id: str,
    source_id: str,                        # The Source whose map changes
    new_map: Dict[str, Dict[str, Any]],    # The FULL new retired_spines map (plan_retire's first result)
    plan: Dict[str, Any],                  # plan_retire's second result
    *,
    journal_path: Optional[str],           # Sidecar journal (None = unjournaled)
    actor: str,
) -> Dict[str, Any]:  # The op as journaled
    """Apply + journal one retirement fact (write-side dual of apply_spine_fact)."""
    sp = plan["spine"]
    op = {"verb": RETIRE_VERB, "actor": actor,
          "args": {"act": plan["act"], "source_id": source_id, "spine": plan["key"],
                   "skeleton_hash": sp.get("skeleton_hash"), "split_policy": sp.get("split_policy"),
                   "segments": sp.get("segments"), **{k: v for k, v in plan["entry"].items() if k != "actor"}},
          "updates": [{"id": source_id, "properties": {RETIRED_SPINES_PROP: new_map}}]}
    await apply_spine_fact(queue, graph_id, op)
    if journal_path:
        append_op(journal_path, op, dedup=False)
    logger.info(f"{RETIRE_VERB} {plan['act']}: source {source_id[:8]} spine {plan['key'][:16]}")
    return op


async def retire_spine(
    queue: Any, graph_id: str,
    source_id: str,
    selector: str,
    *,
    reason: str = "",
    successor: Optional[str] = None,
    unretire: bool = False,
    journal_path: Optional[str],
    actor: str,
    force_dependents: bool = False,  # NOT offered by the CLI: rule (b) has no blind override; tests only
    dep_map: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,  # Prefetched `dependents_map` (batch callers); None = typed per-spine read
) -> Dict[str, Any]:  # {"op", "plan", "dependents"}
    """The whole retire act: list -> plan (rules a) -> dependents gate (rule b) -> journaled fact."""
    rends = await source_rendition_ids(queue, graph_id, source_id)
    spines = await list_spines(queue, graph_id, source_id, rends)
    if not spines:
        raise ValueError(f"source {source_id[:8]} has no decomposed spine")
    src = await get_source(queue, graph_id, source_id)
    current = retired_spines((src or {}).get("properties"))
    new_map, plan = plan_retire(spines, current, selector, reason=reason, successor=successor,
                                unretire=unretire, actor=actor)
    dep: Dict[str, Any] = {}
    if plan["act"] == "retire":
        dep = _dep_for(dep_map, source_id, plan["key"]) or {}
        if not dep:
            seg_ids = await spine_segment_ids(queue, graph_id, rends, plan["spine"].get("skeleton_hash"))
            dep = await spine_dependents(queue, graph_id, seg_ids)
        if not dependents_free(dep) and not force_dependents:
            raise ValueError(
                f"spine {spine_label(plan['spine'])} has DEPENDENTS (rule (b)): {dep['corrections']} "
                f"correction(s), {dep['reviews']} review(s) across {len(dep['sessions'])} session(s) "
                f"touching {dep['segments_touched']} segment(s) — transfer them to the successor "
                f"(correction-core transfer-wordless / the respine plan) or retire the sessions "
                f"first; nothing cascades")
    op = await journal_spine_retire(queue, graph_id, source_id, new_map, plan,
                                    journal_path=journal_path, actor=actor)
    return {"op": op, "plan": plan, "dependents": dep}


async def plan_superseded(
    queue: Any, graph_id: str,
    source_ids: Iterable[str],  # Sources to sweep
    dep_map: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,  # Prefetched `dependents_map`; None = typed per-spine reads
) -> List[Dict[str, Any]]:  # One row per candidate spine: {"source_id","spine","successor","dependents","eligible"}
    """The batch candidate list: every LIVE spine that is not the source's default (rule (c)) —
    eligible when dependent-free; the caller retires the eligible ones with successor = the default."""
    out: List[Dict[str, Any]] = []
    for sid in source_ids:
        rends = await source_rendition_ids(queue, graph_id, sid)
        spines = await list_spines(queue, graph_id, sid, rends)
        default = default_live_spine(spines)
        if default is None:
            continue
        for sp in live_spines(spines):
            if spine_key(sp.get("skeleton_hash")) == spine_key(default.get("skeleton_hash")):
                continue
            dep = _dep_for(dep_map, sid, spine_key(sp.get("skeleton_hash")))
            n_segs = int(sp.get("segments") or 0)
            if dep is None:
                seg_ids = await spine_segment_ids(queue, graph_id, rends, sp.get("skeleton_hash"))
                dep = await spine_dependents(queue, graph_id, seg_ids)
                n_segs = len(seg_ids)
            out.append({"source_id": sid, "spine": sp, "successor": default, "segments": n_segs,
                        "dependents": dep, "eligible": dependents_free(dep)})
    return out


async def compact_retired(
    queue: Any, graph_id: str,
    source_ids: Iterable[str],   # Sources to sweep for retired-but-uncompacted spines
    *,
    journal_path: str,           # The workflow journal's live tail
    archive_dir: str,            # Archive family location (outside the backup mirror)
    label: str,                  # Compaction label (manifest + split markers)
    actor: str,
    dry_run: bool = False,
    delete_batch: int = 2000,
) -> Dict[str, Any]:  # {"report": CompactReport, "targets": [...], "deleted": n, "dangling_after": n}
    """The compact act over every retired, not-yet-compacted spine of the given sources:
    ONE journal pass for all of them (the scan is O(journal), never O(spines)), then the live-db
    deletes, then one `spine-compaction` fact per spine, then the static re-scan (must be 0)."""
    targets: List[Dict[str, Any]] = []
    all_ids: List[str] = []
    for sid in source_ids:
        src = await get_source(queue, graph_id, sid)
        retired = retired_spines((src or {}).get("properties"))
        pending = {k: e for k, e in retired.items() if not e.get("compacted")}
        if not pending:
            continue
        # The retirement map is the truth: read each retired spine's segment ids by its
        # key directly (no spine listing — that 4-prop projection over EVERY segment of
        # the source was the sweep's dominant cost).
        rends = await source_rendition_ids(queue, graph_id, sid)
        for key, e in pending.items():
            skel = None if key == LEGACY_KEY else key
            seg_ids = await spine_segment_ids(queue, graph_id, rends, skel)
            if not seg_ids:
                continue
            sp = {"skeleton_hash": skel, "split_policy": None, "segments": len(seg_ids),
                  "retired": True, "retired_reason": e.get("reason"), "successor": e.get("successor")}
            targets.append({"source_id": sid, "key": key, "spine": sp, "segment_ids": seg_ids})
            all_ids.extend(seg_ids)
    report: CompactReport = compact_journal(journal_path, archive_dir, all_ids, label=label,
                                            dry_run=dry_run, actor=actor)
    result: Dict[str, Any] = {"report": report, "targets": targets, "deleted": 0, "dangling_after": 0}
    if dry_run or not targets:
        return result
    deleted = 0
    for t in targets:
        ids = t["segment_ids"]
        for i in range(0, len(ids), delete_batch):
            n = await graph_task(queue, graph_id, "delete_nodes", node_ids=ids[i:i + delete_batch])
            deleted += int(n or 0)
        src = await get_source(queue, graph_id, t["source_id"])
        retired = retired_spines((src or {}).get("properties"))
        entry = dict(retired.get(t["key"]) or {})
        entry["compacted"] = {"ts": time.time(), "label": label, "archive": report.archive_path,
                              "segments": len(ids), "actor": actor}
        retired[t["key"]] = entry
        op = {"verb": COMPACTION_VERB, "actor": actor,
              "args": {"act": "compact", "source_id": t["source_id"], "spine": t["key"],
                       "segments": len(ids), "archive": report.archive_path, "label": label},
              "updates": [{"id": t["source_id"], "properties": {RETIRED_SPINES_PROP: retired}}]}
        await apply_spine_fact(queue, graph_id, op)
        append_op(journal_path, op, dedup=False)
    result["deleted"] = deleted
    result["dangling_after"] = len(scan_references(journal_path, all_ids))
    return result
