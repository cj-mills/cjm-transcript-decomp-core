"""The shared launch surface every decomp shell drives through: the argument
surface (build_parser), the split-flag contract (resolve_split_flags), the
resolution ladder (resolve_settings), and the headless hand-off tail
(batch_argv/hand_off) that runs each text-from group through the core CLI
in-process. Every group's equivalent cjm-transcript-decomp-core command
prints before execution — shell-queued batches stay reproducible by
copy-paste — and --plan-only stops at the printout. Absorbed from the Textual
shell's cli module (spine absorption 12f342f1) so every shell imports the
SAME ladder — a resolution drift between shells would fork the
reproducibility contract (DEC 6c574c89)."""

import argparse
import os
import shlex
from typing import Any, Dict, List, Optional

import yaml
from cjm_substrate.core.workspace import resolve_workspace

from .cli import main as core_main
from .discovery import discover_capability
from .state import load_state, save_state


def build_parser() -> argparse.ArgumentParser:  # Configured CLI parser
    """The TUI driver's argument surface (batch-setup options + core passthrough)."""
    p = argparse.ArgumentParser(
        prog="cjm-transcript-decomp-tui",
        description="Batch-setup TUI for the headless decomp pipeline: pick "
                    "transcription runs into an ordered batch (with per-run "
                    "authoritative-text choices), then hand off to "
                    "cjm-transcript-decomp-core's batch runner — one loaded "
                    "capability stack per text-from group.")
    p.add_argument("--workspace", default=None,
                   help="Workspace root (5daadfc4; default: CJM_WORKSPACE env, else upward walk "
                        "from cwd). Supplies runs/manifests defaults and is exported so the "
                        "core hand-off + capability workers resolve workspace-scoped paths")
    p.add_argument("--runs-dir", default=None,
                   help="Run-manifest directory (default: the workspace's runs/ when one is "
                        "active, else runs/ under the cwd — both cores' legacy default)")
    p.add_argument("--manifests-dir", default=None,
                   help="Capability manifests directory (default: the workspace's "
                        ".cjm/manifests when one is active, else .cjm/manifests under the cwd)")
    p.add_argument("--proposals-dir", default=None,
                   help="Proposal-set directory the propset picker discovers from "
                        "(default: the workspace's proposals/ when one is active, "
                        "else proposals/ under the cwd)")
    p.add_argument("--training-runs-dir", default=None,
                   help="Training-run directory the P propose-now pre-stage discovers "
                        "models from (default: the workspace's training-runs/ when one "
                        "is active, else training-runs/ under the cwd)")
    p.add_argument("--event-training-run", default=None,
                   help="Training-run id (or id tail) P proposes with; overrides the "
                        "workspace marker's flywheel.event_training_run pin "
                        "(default: the pin, else the newest run behind P's confirm)")
    p.add_argument("--vad-capability", default="cjm-capability-silero-vad",
                   help="VAD capability name (forwarded to the core)")
    p.add_argument("--fa-capability", default="cjm-capability-qwen3-forced-aligner",
                   help="Forced-alignment capability name (forwarded to the core)")
    p.add_argument("--graph-capability", default="cjm-capability-graph-sqlite",
                   help="Graph-storage capability the fine spine extends "
                        "(decomp REQUIRES one — there is no unjournaled decomp)")
    p.add_argument("--graph-db-path", default=None,
                   help="Explicit graph db path (default: last-used, else the "
                        "capability's configured db_path)")
    p.add_argument("--sysmon-capability", default=None,
                   help="monitor capability for GPU attribution (default: last-used, "
                        "else auto-discovered from manifests)")
    p.add_argument("--no-sysmon", action="store_true",
                   help="Explicitly disable the monitor (overrides state + discovery)")
    p.add_argument("--language", default="English",
                   help="Forced-alignment language (forwarded to the core)")
    p.add_argument("--sentence-split", action=argparse.BooleanOptionalAction, default=None,
                   help="Run every confirmed group with the post-FA sentence-split "
                        "stage (forwarded to the core; commits a PARALLEL spine). "
                        "Unset resolves ON (DEC 552bde8d) — unless --event-split is "
                        "armed (verdict ad963c57 flipped the respine seam default); "
                        "an explicit flag always wins. Also toggleable in-TUI with "
                        "s on the runs stage")
    p.add_argument("--split-min-chunk-s", type=float, default=0.5,
                   help="Sentence-split min sub-chunk duration guard, seconds "
                        "(forwarded to the core; identity input)")
    p.add_argument("--respine", action="store_true",
                   help="Run every confirmed group with --respine (DEC 9241564f): a FRESH "
                        "skeleton spine under the SAME config — the recovery for the "
                        "post-upgrade verify-collide (finding e8458f6e). Also toggleable "
                        "in-TUI with R on the runs stage")
    p.add_argument("--event-split", action="store_true",
                   help="Run every confirmed group with the post-FA event-carve stage "
                        "(forwarded to the core; verdict ad963c57 — model cuts replace "
                        "pysbd). Each run's propset resolves latest-by-source from "
                        "--proposals-dir (E cycles alternates); also toggleable "
                        "in-TUI with e on the runs stage")
    p.add_argument("--event-propset", default=None,
                   help="Explicit ProposalSetManifest pointer (manifest json or its "
                        "set dir) — the scripted path's PIN: disables per-source "
                        "resolution, so the batch must select exactly ONE manifest")
    p.add_argument("--event-classes", nargs="+", default=["inhale"],
                   help="Proposal classes that carve (forwarded to the core; default: "
                        "inhale — word-bearing classes must never cut)")
    p.add_argument("--force", action="store_true",
                   help="Bypass capability-side caches (forwarded to the core)")
    p.add_argument("--actor", default=None,
                   help="Forwarded journal attribution (default: cli:<user>)")
    p.add_argument("--gap-threshold", type=float, default=2.0,
                   help="Seconds of uncovered span between committed fine "
                        "segments that paints a gap row in the segments drill "
                        "(a chunk VAD never cut shows only as a gap)")
    p.add_argument("--plan-only", action="store_true",
                   help="Print each group's equivalent headless command and exit "
                        "WITHOUT running anything")
    return p


def batch_argv(
    batch: Dict[str, Any],       # One confirmed group: {"text_from", "graph_db_path", "manifests"}
    args: argparse.Namespace,    # The TUI's parsed args (passthrough run options)
    sysmon: Optional[str],       # Resolved monitor capability (None = disabled)
) -> List[str]:  # cjm-transcript-decomp-core argv (the reproducibility contract)
    """Render one hand-off group as headless decomp-core argv.

    Everything the TUI decided (the ordered member manifests, the group's
    authoritative transcriber, the group's graph db — the one the SOURCE runs
    recorded writing to, e087d059) plus everything it merely passes through
    (capabilities, language, force, sysmon, actor) lands in ONE argv — printed
    before execution so any TUI-queued batch can be replayed by hand.
    --output-dir pins decomp manifests to the SAME runs dir the TUI browsed
    (7dfd1177: cwd-relative output blinded the results view and the coverage
    chips).
    """
    argv = ["run", *batch["manifests"], "--yes",
            "--manifests-dir", args.manifests_dir,
            "--vad-capability", args.vad_capability,
            "--fa-capability", args.fa_capability,
            "--graph-capability", args.graph_capability,
            "--language", args.language,
            "--output-dir", args.runs_dir]
    if batch.get("text_from"):
        argv += ["--text-from", batch["text_from"]]
    if batch.get("graph_db_path"):
        argv += ["--graph-db-path", batch["graph_db_path"]]
    if sysmon:
        argv += ["--sysmon-capability", sysmon]
    if args.force:
        argv += ["--force"]
    if args.sentence_split:
        argv += ["--sentence-split", "--split-min-chunk-s", str(args.split_min_chunk_s)]
    else:
        # The core is default-on (DEC 552bde8d): an OFF toggle must ride the
        # argv explicitly or the hand-off silently re-enables the split.
        argv += ["--no-sentence-split"]
    if args.respine:
        # Core default is OFF, so only the ON state needs rendering — but it
        # MUST render (DEC 9241564f): a fresh spine is a deliberate, visible act.
        argv += ["--respine"]
    if args.event_split:
        # Armed-only (core default off); the propset pointer and the carve
        # classes render explicitly — the printed argv is the full contract.
        # The GROUP's resolved propset wins (per-source picker, DEC ae450551);
        # the CLI flag is the scripted path's pin.
        # A group's propset is ONE pointer per source (0.2.6): a str for a
        # single-source group, a list/tuple for a multi-source run.
        ps = batch.get("event_propset") or args.event_propset
        pointers = [ps] if isinstance(ps, str) else list(ps or [])
        argv += ["--event-split",
                 "--event-propset", *pointers,
                 "--event-classes", *args.event_classes]
    if args.actor:
        argv += ["--actor", args.actor]
    return argv


def resolve_settings(args: argparse.Namespace) -> Dict[str, Any]:  # Resolved batch-setup surface
    """Resolve the batch-setup settings every shell shares (flags > persisted
    state > manifest discovery), exporting CJM_WORKSPACE so the in-process core
    hand-off + capability workers are workspace-scoped (5daadfc4). Extracted
    from main() so the Qt shell's driver reuses the EXACT ladder (DEC 6c574c89
    — the dcf8a712 extraction replayed on the decomp lane): a resolution drift
    between shells would fork the reproducibility contract. Mutates the args
    dirs to their resolved defaults (batch_argv reads them there)."""
    ws = resolve_workspace(explicit=args.workspace)
    if ws is not None:
        os.environ["CJM_WORKSPACE"] = str(ws.root)
    if args.manifests_dir is None:
        args.manifests_dir = (str(ws.substrate_data_dir / "manifests")
                              if ws is not None else ".cjm/manifests")
    if args.runs_dir is None:
        args.runs_dir = str(ws.runs_dir) if ws is not None else "runs"
    if args.proposals_dir is None:
        args.proposals_dir = (str(ws.root / "proposals")
                              if ws is not None else "proposals")
    if args.training_runs_dir is None:
        args.training_runs_dir = (str(ws.root / "training-runs")
                                  if ws is not None else "training-runs")
    # The workspace marker carries the flywheel's CURRENT-model pin as DATA
    # (flywheel.event_training_run) — the flag wins, the pin is the standing
    # default, absence leaves P on newest-run + its two-step confirm.
    training_run_pin = args.event_training_run
    if training_run_pin is None and ws is not None:
        try:
            marker = yaml.safe_load((ws.root / "cjm-workspace.yaml").read_text())
        except (OSError, yaml.YAMLError):
            marker = None
        if isinstance(marker, dict):
            fw = marker.get("flywheel")
            if isinstance(fw, dict) and fw.get("event_training_run"):
                training_run_pin = str(fw["event_training_run"])
    state = load_state(args.manifests_dir)
    sysmon = None if args.no_sysmon else (
        args.sysmon_capability or state.get("sysmon_capability")
        or discover_capability(args.manifests_dir, "get_system_status"))
    return {"manifests_dir": args.manifests_dir,
            "runs_dir": args.runs_dir,
            "proposals_dir": args.proposals_dir,
            "training_runs_dir": args.training_runs_dir,
            "training_run_pin": training_run_pin,
            "sysmon_capability": sysmon,
            "graph_db_path": args.graph_db_path or state.get("graph_db_path"),
            "state": state}


def hand_off(
    plan: Optional[Dict[str, Any]],  # The app's confirmed exit plan (None = no batch)
    args: argparse.Namespace,        # Resolved args (post resolve_split_flags/resolve_settings)
) -> int:  # Aggregated exit code (max over the batch's core invocations)
    """The shared driver tail: persist the confirmed choices, adopt the in-app
    s/R/e toggles, gate the event-split hand-off, then print + run each group
    in-process (extracted from main() alongside resolve_settings — every
    shell's confirmed batch replays by copy-paste)."""
    if not plan:
        print("no batch confirmed")
        return 0
    save_state(args.manifests_dir,
               sysmon_capability=plan["sysmon_capability"],
               graph_db_path=plan["graph_db_path"])
    # The in-app s/R/e toggles win over the launch flags (the hub path has no flags).
    args.sentence_split = bool(plan.get("sentence_split"))
    args.respine = bool(plan.get("respine"))
    args.event_split = bool(plan.get("event_split"))
    err = event_split_batch_error(plan["batches"], args)
    if err:
        print(f"error: {err}")
        return 2
    rc = 0
    for i, batch in enumerate(plan["batches"]):
        argv = batch_argv(batch, args, plan["sysmon_capability"])
        print(f"batch {i + 1}/{len(plan['batches'])}: "
              + shlex.join(["cjm-transcript-decomp-core"] + argv))
        if args.plan_only:
            continue
        # Sequential by design: groups share the GPU; the core loads one stack
        # per group and its exit code aggregates that group's members.
        rc = max(rc, int(core_main(argv)))
    return rc


def resolve_split_flags(args: argparse.Namespace) -> argparse.Namespace:  # The same namespace, resolved
    """Resolve the post-parse split-flag contract (pure; main() calls it
    BEFORE the TUI launches so a bad combination fails loud and early).

    --sentence-split parses to a None sentinel: unset lands ON (DEC 552bde8d)
    — unless --event-split is armed, where the verdict ad963c57 flipped the
    respine seam default (sentence_split off, event_split on). An explicit
    flag always wins: the stages stay composable (DEC 6cc10fb7), the default
    just stopped fighting the ratified respine shape. --event-split needs no
    pointer here — each run's propset resolves latest-by-source in the TUI
    (DEC ae450551); a missing set fails loud at confirm and again at the
    event_split_batch_error hand-off gate."""
    if args.sentence_split is None:
        args.sentence_split = not args.event_split
    return args


def event_split_batch_error(
    batches: List[Dict[str, Any]],  # The confirmed plan's hand-off groups
    args: argparse.Namespace,       # Resolved TUI args (post resolve_split_flags)
) -> Optional[str]:  # Refusal message, or None when the hand-off is safe
    """One propset carves ONE source: the core applies --event-propset to every
    manifest it is handed with no source check (event_spans_from_propset loads
    by pointer), so a mismatched hand-off would carve wrong spans SILENTLY.

    Two safe shapes (DEC ae450551): (a) per-GROUP propsets from the TUI's
    per-source picker — the propset joins the group key, so any batch width is
    safe (each group carries its own source's set); (b) the scripted CLI pin
    (--event-propset) — one pointer, so the batch must select exactly ONE
    manifest. A group with NEITHER pointer has nothing to carve from: refuse."""
    if not args.event_split:
        return None
    if any(not (b.get("event_propset") or args.event_propset) for b in batches):
        return ("--event-split armed but a group has no proposal set — propose "
                "over its source first (or pass --event-propset)")
    if args.event_propset and not all(b.get("event_propset") for b in batches):
        total = sum(len(b["manifests"]) for b in batches)
        if total != 1:
            return (f"--event-propset pins ONE set for ONE source; the confirmed "
                    f"batch selects {total} manifests — select exactly 1")
    return None
