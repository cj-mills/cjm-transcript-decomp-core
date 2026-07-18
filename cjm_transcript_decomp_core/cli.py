"""The CLI driver — the decomposition core's first (and currently only) frontend.

Ships in-package as the `cjm-transcript-decomp-core` console script so the driver
can never skew from the core. GUI presentation drivers come later and consume the
same `pipeline` module; they never reimplement it (CLI-first / headless-core
principle).

Prerequisite runtime (once, from the repo root):

    cjm-ctl --cjm-config cjm.yaml setup-runtime
    cjm-ctl --cjm-config cjm.yaml install-all --capabilities capabilities_test.yaml --force

Then decompose transcription-core run manifest(s) — a multi-manifest batch
shares ONE loaded capability stack (models load once, per-manifest runs):

    cjm-transcript-decomp-core run path/to/transcription-run.json --yes
    cjm-transcript-decomp-core run runs/run_a.json runs/run_b.json --yes
    # GPU runs: opt into CR-7 GPU subtree attribution
    cjm-transcript-decomp-core run run.json --yes --sysmon-capability cjm-capability-monitor-nvidia
"""

import argparse
import asyncio
import getpass
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from cjm_substrate.core.manager import CapabilityManager
from cjm_substrate.core.queue import JobQueue
from cjm_transcript_decomp_core.models import DecompConfig
from cjm_transcript_decomp_core.pipeline import load_source_manifest, run_decomp

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:  # Configured CLI parser
    """Build the CLI parser (subcommands: run)."""
    parser = argparse.ArgumentParser(
        prog="cjm-transcript-decomp-core",
        description="Headless transcript decomposition: VAD + forced alignment -> fine-spine graph extension.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Extend a transcription-emitted graph root with the fine spine")
    run.add_argument("manifests", nargs="+",
                     help="Transcription-core run manifest JSON(s) — a batch shares ONE loaded capability stack")
    run.add_argument("--manifests-dir", default=".cjm/manifests", help="Capability manifests directory")
    run.add_argument("--vad-capability", default="cjm-capability-silero-vad", help="VAD capability name")
    run.add_argument("--fa-capability", default="cjm-capability-qwen3-forced-aligner", help="Forced-alignment capability name")
    run.add_argument("--graph-capability", default="cjm-capability-graph-sqlite", help="Graph-storage capability name")
    run.add_argument("--graph-db-path", default=None,
                     help="Explicit graph DB path override (F10; scratch-graph loop-backs; default: the capability's configured db_path)")
    run.add_argument("--text-from", default=None,
                     help="Authoritative transcriber for layer-0 text (the ACCURACY model; "
                          "required when the manifest carries multiple transcribers; "
                          "default: the manifest's sole transcriber)")
    run.add_argument("--sysmon-capability", default=None, help="monitor capability for GPU subtree attribution (CR-7); loaded first; default: no monitor")
    run.add_argument("--language", default="English", help="Forced-alignment language")
    run.add_argument("--force", action="store_true", help="Bypass capability-side caches (VAD + FA)")
    run.add_argument("-y", "--yes", action="store_true", help="Auto-accept HITL seams (headless mode)")
    run.add_argument("--output", default=None, help="Decomp-manifest output path (single-manifest runs only; default: runs/<run_id>.json)")
    run.add_argument("--actor", default=None,
                     help="Journal attribution for who/what initiated this run (default: cli:<username>)")
    run.add_argument("-v", "--verbose", action="store_true", help="DEBUG-level logging")
    return parser


def load_capabilities(
    manager: CapabilityManager,   # Freshly constructed manager
    instance_ids: List[str],  # Capability names to load (default instances), in order
    configs: Optional[Dict[str, Dict[str, Any]]] = None,  # Per-capability config overrides (caller-wins, C8)
) -> None:
    """Discover manifests + load each requested capability (default instance)."""
    manager.discover_manifests()
    discovered = {m.name: m for m in manager.discovered}
    for iid in instance_ids:
        meta = discovered.get(iid)
        if meta is None:
            raise SystemExit(
                f"capability {iid!r} not found in manifests "
                f"(discovered: {sorted(discovered)}) — run cjm-ctl install-all first"
            )
        if not manager.load_capability(meta, config=(configs or {}).get(iid)):
            raise SystemExit(f"failed to load capability {iid!r}")
        logger.info(f"loaded {iid}")


async def run_command(
    args: argparse.Namespace,  # Parsed CLI arguments for the `run` subcommand
) -> int:  # Process exit code (0 = every manifest fully extended + verified)
    """Execute the `run` subcommand: extend transcription-run manifest(s) with the fine spine.

    Batch shape (decomp-TUI demand, work item 0ff6bf0f): N manifests ride ONE
    capability stack — the models load once and every manifest reuses them,
    where N CLI invocations would pay N model loads. Each manifest keeps its
    own run_id, decomp manifest, and RUN_STARTED/RUN_FINISHED journal bracket,
    so a batch member is indistinguishable from a solo run downstream. A failed
    member records and the batch continues (unattended queueing must not sink
    the rest); the exit code aggregates across members.
    """
    manifest_paths = [str(Path(m).resolve()) for m in args.manifests]
    missing = [m for m in manifest_paths if not Path(m).exists()]
    if missing:
        raise SystemExit(f"source manifest(s) not found: {', '.join(missing)}")
    if args.output and len(manifest_paths) > 1:
        raise SystemExit("--output names ONE decomp manifest — omit it for batch runs "
                         "(each lands at runs/<run_id>.json)")

    cfg = DecompConfig(
        vad_capability=args.vad_capability,
        fa_capability=args.fa_capability,
        graph_capability=args.graph_capability,
        text_from=args.text_from,
        language=args.language,
        force=args.force,
        assume_yes=args.yes,
    )

    # CR-7 GPU subtree attribution is opt-in: --sysmon-capability threads the monitor
    # name into BOTH the manager and the queue; the monitor loads FIRST so GPU
    # capabilities' samples record gpu_memory_mb_peak (voxtral-vllm e2e pattern).
    manager = CapabilityManager(
        search_paths=[Path(args.manifests_dir)],
        sysmon_capability_name=args.sysmon_capability,
    )
    instance_ids = [cfg.vad_capability, cfg.fa_capability, cfg.graph_capability]
    load_order = ([args.sysmon_capability] if args.sysmon_capability else []) + instance_ids
    # F10: --graph-db-path threads a caller-wins config into the graph load
    # (the C8 pattern correction-core already used; scratch-graph loop-backs).
    configs = ({cfg.graph_capability: {"db_path": args.graph_db_path}}
               if args.graph_db_path else None)
    load_capabilities(manager, load_order, configs=configs)

    queue = JobQueue(deps=manager, sysmon_capability_name=args.sysmon_capability)
    await queue.start()
    all_ok = True
    try:
        # CR-14 follow-up: actor attribution (operator identity by default;
        # agents/services pass --actor explicitly).
        actor = args.actor or f"cli:{getpass.getuser()}"
        for mp in manifest_paths:
            try:
                manifest = await run_decomp(manager, queue, cfg, mp, actor=actor)
            except Exception as e:  # One member's failure must not sink the batch
                logger.error(f"decomp failed for {mp}: {e}")
                print(f"FAILED {mp}: {e}")
                all_ok = False
                continue
            out = Path(args.output) if args.output else Path("runs") / f"{manifest.run_id}.json"
            manifest.save(out)
            n_manifest_sources = len(load_source_manifest(mp).get("sources", []) or [])
            n_sources = len(manifest.sources)
            n_segs = sum(s.segment_count for s in manifest.sources)
            print(f"decomp manifest: {out}")
            print(f"sources extended: {n_sources}/{n_manifest_sources}  segment nodes: {n_segs}")
            all_ok = all_ok and (n_sources == n_manifest_sources)
    finally:
        await queue.stop()
        for iid in reversed(load_order):  # Reverse load order; the monitor unloads last
            try:
                manager.unload_capability(iid)
            except Exception as e:  # Best-effort teardown; never mask the run's outcome
                logger.warning(f"unload {iid} failed: {e}")
    return 0 if all_ok else 1


def main(
    argv: Optional[List[str]] = None,  # Argument list override (None = sys.argv)
) -> int:  # Process exit code
    """CLI entry point (console script: `cjm-transcript-decomp-core`)."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
    )
    if args.command == "run":
        return asyncio.run(run_command(args))
    raise SystemExit(f"unknown command: {args.command}")
