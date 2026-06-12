#!/usr/bin/env python
"""Cold-run M×(VAD ∥ T×FA) fan-out overlap measurement (stage-5 closeout;
the carried G11 discipline: parallelism claims need WALL-CLOCK assertions).

Decomposes a FRESH dual-transcriber transcription manifest and measures
per-lane overlap from in-process Job records:

  - the per-transcriber FA fan-out (T×FA per segment) — EXPECTED max
    in-flight 1: model workers stay serial-per-instance by design (SG-33
    unset = queue default of 1); this measurement documents the posture
  - VAD ∥ FA cross-instance co-run (the lane overlap admission permits)

Fresh sources make every VAD + FA call cold by content (no --force games).
Inspect `.cjm/plugin_configs.db` for leftover persisted configs BEFORE any
cold run (the I8 lesson).

As-measured baseline (2026-06-11, RTX 4090, HH1+HH2 = 12 segments x 2
transcribers): wall 16.2s vs summed 23.2s; FA lane 24 jobs max in-flight 1;
VAD∥FA co-run 7.6s = 98% of the shorter lane; verify_source all-True x2.

Run from the repo root in the cjm-transcript-decomp-core env:
    python tests_manual/measure_cold_fanout_overlap_e2e.py <transcription-manifest.json>
"""
import asyncio
import logging
import sys
import time
from pathlib import Path

from cjm_plugin_system.core.manager import PluginManager
from cjm_plugin_system.core.queue import JobQueue
from cjm_transcript_decomp_core.cli import load_capabilities
from cjm_transcript_decomp_core.models import DecompConfig
from cjm_transcript_decomp_core.pipeline import run_decomp

SCRATCH_DB = "/tmp/stage5_closeout_scratch/context_graph.db"
SYSMON = "cjm-system-monitor-nvidia"
VAD = "cjm-media-plugin-silero-vad"
FA = "cjm-transcription-plugin-qwen3-forced-aligner"
GRAPH = "cjm-graph-plugin-sqlite"
TEXT_FROM = "cjm-transcription-plugin-voxtral-hf"  # accuracy authority


def lane_intervals(jobs, instance_id):
    """(start, end) UTC intervals for completed jobs of one instance."""
    return sorted((j.started_at, j.completed_at) for j in jobs
                  if j.plugin_instance_id == instance_id and j.started_at and j.completed_at)


def merge(iv):
    """Merge overlapping intervals so a side can't double-count itself."""
    merged = []
    for s, e in sorted(iv):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def overlap_seconds(a, b):
    """Total seconds where any interval of A overlaps any interval of B."""
    total = 0.0
    for sa, ea in merge(a):
        for sb, eb in merge(b):
            lo, hi = max(sa, sb), min(ea, eb)
            if hi > lo:
                total += (hi - lo).total_seconds()
    return total


def self_overlap_max(iv):
    """Max simultaneously in-flight jobs within one lane + seconds at depth >= 2."""
    events = []
    for s, e in iv:
        events.append((s, 1))
        events.append((e, -1))
    events.sort(key=lambda t: (t[0], -t[1]))
    depth = max_depth = 0
    over2 = 0.0
    prev = None
    for ts, d in events:
        if prev is not None and depth >= 2:
            over2 += (ts - prev).total_seconds()
        depth += d
        max_depth = max(max_depth, depth)
        prev = ts
    return max_depth, over2


async def main():
    # Bypassing cli.main() means configuring logging HERE — otherwise the
    # pipeline's logger.info lines (incl. the verify results) are silently
    # dropped (Python's lastResort handler only surfaces WARNING+).
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
    if len(sys.argv) != 2:
        raise SystemExit("usage: measure_cold_fanout_overlap_e2e.py <transcription-manifest.json>")
    manifest_path = str(Path(sys.argv[1]).resolve())
    cfg = DecompConfig(
        vad_plugin=VAD, fa_plugin=FA, graph_plugin=GRAPH,
        text_from=TEXT_FROM, language="English",
        force=False,  # fresh sources are cold by content
        assume_yes=True,
    )
    manager = PluginManager(search_paths=[Path(".cjm/manifests")], sysmon_plugin_name=SYSMON)
    load_order = [SYSMON, VAD, FA, GRAPH]
    load_capabilities(manager, load_order, configs={GRAPH: {"db_path": SCRATCH_DB}})
    queue = JobQueue(deps=manager, sysmon_plugin_name=SYSMON)
    await queue.start()
    t0 = time.monotonic()
    try:
        manifest = await run_decomp(manager, queue, cfg, manifest_path)
        wall = time.monotonic() - t0
        jobs = list(queue._jobs.values())
        print(f"\n===== COLD-RUN FAN-OUT OVERLAP RESULTS =====")
        print(f"wall: {wall:.1f}s  jobs: {len(jobs)}")
        summed = sum((j.completed_at - j.started_at).total_seconds()
                     for j in jobs if j.started_at and j.completed_at)
        if summed:
            print(f"summed job durations: {summed:.1f}s -> wall/summed {wall/summed:.2f}")

        vad_iv = lane_intervals(jobs, VAD)
        fa_iv = lane_intervals(jobs, FA)
        vad_sum = sum((e - s).total_seconds() for s, e in vad_iv)
        fa_sum = sum((e - s).total_seconds() for s, e in fa_iv)
        co = overlap_seconds(vad_iv, fa_iv)
        fa_depth, fa_over2 = self_overlap_max(fa_iv)
        print(f"VAD lane: {len(vad_iv)} jobs, {vad_sum:.1f}s busy")
        print(f"FA lane:  {len(fa_iv)} jobs, {fa_sum:.1f}s busy, "
              f"max in-flight {fa_depth} (expect 1: serial-per-instance by design), "
              f">=2-deep {fa_over2:.1f}s")
        if min(vad_sum, fa_sum):
            print(f"VAD∥FA cross-instance CO-RUN: {co:.1f}s "
                  f"({100*co/min(vad_sum, fa_sum):.0f}% of the shorter lane)")

        for s in manifest.sources:
            print(f"extended [{Path(s.source_path).name}]: {s.segment_count} segments "
                  f"(source node {s.source_node_id})")
        out = Path("runs") / f"{manifest.run_id}.json"
        manifest.save(out)
        print(f"manifest: {out}")
    finally:
        await queue.stop()
        for iid in reversed(load_order):
            try:
                manager.unload_plugin(iid)
            except Exception as e:
                print(f"unload {iid} failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
