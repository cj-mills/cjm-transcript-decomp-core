"""Run-manifest indexes for the decomp-batch TUI (work item 0ff6bf0f): the
transcription core's own runs/*.json read into a selectable batch (decomp
consumes RUN MANIFESTS, not media files), the decomp core's own manifests read
back as coverage chips + results rows, and the pure grouping fold the confirm
hand-off uses. Pure logic, Textual-free (the transcription TUI's results.py
precedent: everything below the paint path tests directly; the app only paints
it). Both cores share the cwd-relative runs/ default, so the manifest FORMAT
tag — the manifest-as-interchange contract (CR-20) — is what separates
transcription runs from decomp runs living in one directory."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cjm_substrate.core.workspace import resolve_recorded_tree


def _load_manifests(
    runs_dir: Path,   # Directory holding both cores' run manifests
    format_tag: str,  # Substring the manifest's format tag must carry
) -> List[Dict[str, Any]]:  # Matching manifest dicts, newest first (+ "_path")
    """Read every readable manifest in runs_dir whose format tag matches.

    The RunIndex forgiveness contract (transcription TUI results.py): the runs
    dir is shared ground — unreadable/foreign jsons are skipped, never raised,
    because one corrupt file must not hide the rest. The format-tag filter is
    load-bearing here where RunIndex could skip it: BOTH cores write manifests
    with run_id + sources into the same directory, so shape alone no longer
    separates them."""
    rows: List[Dict[str, Any]] = []
    try:
        files = sorted(runs_dir.glob("*.json"))
    except OSError:
        files = []
    for f in files:
        try:
            # ${WS}/ recorded paths (5daadfc4 rung f) resolve at load,
            # anchored at the manifest's own location.
            m = resolve_recorded_tree(json.loads(f.read_text()), f)
        except (OSError, ValueError):
            continue
        if not (isinstance(m, dict) and m.get("run_id")
                and format_tag in str(m.get("format", ""))
                and isinstance(m.get("sources"), list)):
            continue
        m["_path"] = str(f)
        rows.append(m)
    rows.sort(key=lambda m: float(m.get("created_at") or 0.0), reverse=True)
    return rows


class SourceRunIndex:
    """Transcription-core run manifests — the decomp workflow's SOURCES — plus
    the per-run facts the batch stage paints: transcriber lists, segment
    totals, and the authoritative-text default. Reads what the transcription
    runs wrote, no parallel record (results-layer principle)."""

    FORMAT_TAG = "transcription-core"

    def __init__(self, runs_dir: str = "runs"):  # Both cores' cwd-relative default
        self.runs_dir = Path(runs_dir)
        self.runs: List[Dict[str, Any]] = []  # Manifest dicts, newest first (+ "_path")

    def load(self) -> int:  # Number of manifests loaded
        """(Re)read every readable transcription-run manifest, newest first."""
        self.runs = _load_manifests(self.runs_dir, self.FORMAT_TAG)
        return len(self.runs)

    @staticmethod
    def transcribers(m: Dict[str, Any]) -> List[str]:  # Transcriber instance ids, manifest order
        """The run's transcriber ids (config snapshot; pre-0.2.0 single-key
        manifests fold to a one-element list — the pipeline's own tolerance)."""
        cfg = m.get("config") or {}
        ids = cfg.get("transcriber_capabilities") or []
        out = [str(i) for i in ids] if isinstance(ids, list) else []
        if not out and cfg.get("transcriber_capability"):
            out = [str(cfg["transcriber_capability"])]
        return out

    @staticmethod
    def segment_count(m: Dict[str, Any]) -> int:  # Pipeline segments across all sources
        """Total pipeline segments in the run (the batch-size signal a row paints)."""
        return sum(len(s.get("segments") or []) for s in m.get("sources") or [])

    @classmethod
    def default_text_from(cls, m: Dict[str, Any]) -> Optional[str]:  # Pre-picked authoritative transcriber
        """The default --text-from pick: the sole transcriber, else the LAST.

        The transcription TUI's confirmed pair lands [lightweight, accuracy],
        so last = the accuracy model — the natural layer-0 authority. A
        CONVENTION default only: the row paints it and t cycles it, so a
        hand-built manifest with a different order is one keypress away."""
        t = cls.transcribers(m)
        return t[-1] if t else None

    @staticmethod
    def recorded_graph_db(
        m: Dict[str, Any],      # A transcription-run manifest
        graph_capability: str,  # The graph capability's instance id
    ) -> Optional[str]:  # The db this run recorded writing to (None = not recorded)
        """The graph db the transcription run RECORDED writing to.

        The manifest's capabilities block is the provenance (finding e087d059:
        defaulting to the decomp stack's own configured db pointed the first
        live batch at the WRONG graph — 'Source root not found'). None means a
        pre-provenance or unjournaled run: the caller must WARN, never
        silently default."""
        cap = (m.get("capabilities") or {}).get(graph_capability) or {}
        db = cap.get("db_path")
        return str(db) if db else None

    @staticmethod
    def source_names(m: Dict[str, Any]) -> List[str]:  # Source display names, manifest order
        """Path stems of the run's sources (finding 614dd647: a run row must
        say WHAT was transcribed without a transcription-TUI round-trip)."""
        return [Path(str(s.get("source_path") or "?")).stem
                for s in m.get("sources") or []]


class DecompIndex:
    """Decomp-core run manifests read back: coverage chips for the batch stage
    (which transcription runs already have a decomp run) and the results
    view's rows. v0 stops at the manifest's own facts — segment TEXTS live in
    the graph, not the manifest, so deeper inspection views wait for real
    decomp-session demand (the work item's deliberate unshaping)."""

    FORMAT_TAG = "transcript-decomp-core"

    def __init__(self, runs_dir: str = "runs"):  # Both cores' cwd-relative default
        self.runs_dir = Path(runs_dir)
        self.runs: List[Dict[str, Any]] = []  # Manifest dicts, newest first (+ "_path")

    def load(self) -> int:  # Number of manifests loaded
        """(Re)read every readable decomp-run manifest, newest first."""
        self.runs = _load_manifests(self.runs_dir, self.FORMAT_TAG)
        return len(self.runs)

    def counts_by_source_manifest(self) -> Dict[str, int]:
        """resolved transcription-manifest path -> decomp runs that consumed it.

        Path-keyed and hash-free (the prior-run-chip pattern: browse-time chips
        must stay cheap); the decomp core records source_manifest RESOLVED, so
        resolving the browse key on lookup matches regardless of how the runs
        dir was spelled."""
        counts: Dict[str, int] = {}
        for m in self.runs:
            p = m.get("source_manifest")
            if p:
                key = str(Path(p).resolve())
                counts[key] = counts.get(key, 0) + 1
        return counts


def group_batches(
    picks: List[Tuple[str, Any]],  # Ordered (manifest_path, hand-off key) selection
) -> List[Tuple[Any, List[str]]]:  # (key, member paths) per hand-off invocation
    """Fold an ordered batch selection into headless hand-off groups.

    The key is EVERYTHING that applies invocation-wide on the core CLI —
    currently (text_from, graph_db_path): --text-from names one authority per
    invocation, and --graph-db-path points one graph per invocation (finding
    e087d059 — runs recorded against different dbs must not share a stack
    config). One core invocation per DISTINCT key is the finest split that
    keeps the ergonomic win: every manifest in a group rides the SAME loaded
    capability stack. First-seen order of groups and members preserves the
    operator's queueing order."""
    order: List[Any] = []
    groups: Dict[Any, List[str]] = {}
    for path, key in picks:
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(path)
    return [(key, groups[key]) for key in order]


class PropsetIndex:
    """Proposal-set manifests under the workspace proposals/ dir — the model's
    source-bound durable artifacts (DEC ae450551: the pattern is source-bound
    artifact + latest-per-source default + in-TUI pick to override, the same
    join rule the correction lane uses). The index is capability-generic: it
    reads the manifest's own facts (source binding, classes, counts, tiers) —
    nothing here knows what an inhale is."""

    FORMAT = "cjm-capability-pyannote/proposal-set-manifest"

    def __init__(self, proposals_dir: str = "proposals"):  # Workspace proposals/ (else cwd-relative)
        self.proposals_dir = Path(proposals_dir)
        self.sets: List[Dict[str, Any]] = []  # Manifest dicts, newest first (+ "_path")

    def load(self) -> int:  # Number of proposal sets loaded
        """(Re)read every readable proposal-set manifest, newest first (the
        _load_manifests forgiveness contract: unreadable/foreign jsons skip)."""
        rows: List[Dict[str, Any]] = []
        try:
            files = sorted(self.proposals_dir.glob("*/manifest.json"))
        except OSError:
            files = []
        for f in files:
            try:
                m = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            if not (isinstance(m, dict) and m.get("format") == self.FORMAT):
                continue
            m["_path"] = str(f)
            rows.append(m)
        rows.sort(key=lambda m: float(m.get("created_at") or 0.0), reverse=True)
        self.sets = rows
        return len(rows)

    def for_source(
        self,
        content_hash: Optional[str] = None,  # Source content hash (preferred join key)
        source_path: Optional[str] = None,   # Source media path (fallback join key)
    ) -> List[Dict[str, Any]]:  # Matching sets, newest first (head = the default pick)
        """Every set proposed over a source, newest first — the head is the
        latest-per-source default; the E-cycle walks the rest (older sets =
        earlier model generations, still auditable)."""
        out: List[Dict[str, Any]] = []
        for m in self.sets:
            src = m.get("source") or {}
            if ((content_hash and src.get("content_hash") == content_hash)
                    or (source_path and str(src.get("path") or "") == str(source_path))):
                out.append(m)
        return out

    @staticmethod
    def summary(m: Dict[str, Any]) -> str:  # One-chip description of a set
        """The row chip's text: set id tail + per-class tier-1 counts (+ tier-2
        total for a dual-tier set — propset manifest 0.2.0)."""
        counts = m.get("counts") or {}
        t2 = sum((m.get("tier2_counts") or {}).values())
        body = "+".join(f"{v}" for v in counts.values()) or "0"
        return (f"{str(m.get('proposal_set_id') or '?')[-8:]} "
                f"{body}{f'+{t2}t2' if t2 else ''}")


class TrainingRunIndex:
    """Training-run manifests under the workspace training-runs/ dir — the
    flywheel's model artifacts, discovered the PropsetIndex way (manifest-
    driven, forgiveness contract, capability-generic: the index reads the
    manifest's own facts — run id, classes, base model — and knows nothing
    about inhales). Serves the P propose-now pre-stage (DEC 1cfe6d0f): the
    workspace marker's flywheel.event_training_run pin names the CURRENT
    model as DATA; without a pin the newest run is only a footer-named
    default behind the two-step confirm — a wrong model burns a walk
    session (b9717422), so the human stays in the loop."""

    FORMAT = "cjm-capability-pyannote/training-run-manifest"

    def __init__(self, training_runs_dir: str = "training-runs"):  # Workspace training-runs/ (else cwd-relative)
        self.training_runs_dir = Path(training_runs_dir)
        self.runs: List[Dict[str, Any]] = []  # Manifest dicts, newest first (+ "_path")

    def load(self) -> int:  # Number of training runs loaded
        """(Re)read every readable training-run manifest, newest first (run
        ids embed mint time, so the id sort IS the recency sort)."""
        rows: List[Dict[str, Any]] = []
        try:
            files = sorted(self.training_runs_dir.glob("*/manifest.json"))
        except OSError:
            files = []
        for f in files:
            try:
                m = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            if not (isinstance(m, dict) and m.get("format") == self.FORMAT):
                continue
            m["_path"] = str(f)
            rows.append(m)
        rows.sort(key=lambda m: str(m.get("run_id") or ""), reverse=True)
        self.runs = rows
        return len(rows)

    def resolve(
        self,
        pin: Optional[str] = None,  # Workspace-marker pin: run id (or unambiguous id tail)
    ) -> Optional[Dict[str, Any]]:  # The propose-now model pointer, or None
        """Pin > newest run. A pin that matches nothing returns None LOUDLY
        (the caller names it) — silently falling back to a different model is
        exactly the field-failure class the register retired overlay-v1 for."""
        if pin:
            for m in self.runs:
                rid = str(m.get("run_id") or "")
                if rid == pin or rid.endswith(pin):
                    return m
            return None
        return self.runs[0] if self.runs else None

    @staticmethod
    def summary(m: Dict[str, Any]) -> str:  # One-line description of a run
        """Footer text for the two-step confirm: run id tail + trained classes."""
        classes = ",".join(m.get("classes") or []) or "?"
        return f"{str(m.get('run_id') or '?')[-8:]} [{classes}]"
