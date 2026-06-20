#!/usr/bin/env python3
"""Stage-7 stress part 6 — volume regression on the REAL corpus.

Re-decomposes Supernova-I (55 pipeline segments × 2 transcribers → 165
compute jobs + the graph task-channel traffic) against a SCRATCH copy of
the corpus graph DB (checkpoint-then-copy; the baseline corpus stays
pristine), then asserts the journal's composition at volume:

  - RUN_STARTED / RUN_FINISHED bracket the run (run-manifest ↔ journal
    linkage by run_id) with the --actor attribution;
  - every job row in the run window carries the run_id (item-3 threading);
  - one ADMISSION_DECIDED row per dispatched job;
  - liveness telemetry (PROGRESS_CHANGED / RESOURCE_SNAPSHOT) is verifiably
    ABSENT from the journal — at real volume, not just in unit fakes;
  - VERIFY_OUTCOME rows present + ok (I14: outcomes are rows);
  - terminal snapshots rehydrate (the `_history` rider) at volume;
  - correction-core parity on the scratch graph: SN-I baselines
    320 empty + 1,176 transcriber-divergence (arc baselines, stage 5).

Run (any env with sqlite3; spawns the cores' own envs):
  python tests_manual/validate_stage7_volume_journal_e2e.py

As-measured baseline (2026-06-12, post-stage-7, cache-hit re-derivation,
editable substrate host + worker envs): decomp wall 20.7s, 736 journal
rows in the run window (179 jobs — VAD + 2×FA + graph task channel — with
1:1 ADMISSION_DECIDED rows + RUN_* + VERIFY_OUTCOME); correction wall
2.5s, 78 run-tagged rows incl. live task_account, worklist 320 empty +
1,176 divergence (exact arc parity). First run of this script caught the
correction-core wrap bug (in-try `return` made the success RUN_FINISHED
unreachable) and the test-sys-mon missing-fastcore K3 replay — both via
the new `cjm-ctl logs --chunks` death rattle.

I8 note: this run is pure cache-hit + default configs — no persisted-config
mutation (plugin_configs.db inspected empty at kickoff).
"""
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

BASE = Path("/mnt/SN850X_8TB_EXT4/Projects/GitHub/cj-mills")
DECOMP = BASE / "cjm-transcript-decomp-core"
CORRECTION = BASE / "cjm-transcript-correction-core"
ENVS = Path.home() / "miniforge3/envs"
SN1_MANIFEST = BASE / "cjm-transcription-core/runs/stage5_sn1.json"
CORPUS_DB = DECOMP / ".cjm/data/cjm-capability-graph-sqlite/context_graph.db"
SCRATCH_DB = Path("/tmp/stage7_volume_scratch_graph.db")
DECOMP_OUT = Path("/tmp/stage7_volume_decomp.json")
ACTOR = "stress:stage7-part6"

# Stage-5 arc baselines for SN-I on the Source-rooted corpus.
BASELINE_EMPTY = 320
BASELINE_DIVERGENCE = 1176


def checkpoint_copy(src: Path, dst: Path) -> None:
    """Checkpoint-then-copy (the stage-3 G3 backup discipline — never copy
    beside a live -wal)."""
    dst.unlink(missing_ok=True)
    con = sqlite3.connect(src)
    try:
        bck = sqlite3.connect(dst)
        with bck:
            con.backup(bck)
        bck.close()
    finally:
        con.close()


def journal_rows(db: Path, where: str = "1=1", params=()):  # list of dict rows
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(
            f"SELECT * FROM journal WHERE {where} ORDER BY seq", params)]
    finally:
        con.close()


def main() -> None:
    assert SN1_MANIFEST.exists(), SN1_MANIFEST
    assert CORPUS_DB.exists(), CORPUS_DB

    print("== scratch graph (checkpoint-then-copy) ==")
    checkpoint_copy(CORPUS_DB, SCRATCH_DB)
    print(f"   {SCRATCH_DB} ({SCRATCH_DB.stat().st_size / 1e6:.1f} MB)")

    journal_db = DECOMP / ".cjm/journal.db"
    cursor = 0
    if journal_db.exists():
        with sqlite3.connect(journal_db) as con:
            cursor = con.execute("SELECT COALESCE(MAX(seq),0) FROM journal").fetchone()[0]

    print("== decomp volume run (SN-I, cache-hit re-derivation) ==")
    t0 = time.monotonic()
    proc = subprocess.run(
        [str(ENVS / "cjm-transcript-decomp-core/bin/cjm-transcript-decomp-core"),
         "run", str(SN1_MANIFEST),
         "--text-from", "cjm-capability-whisper",
         "--graph-db-path", str(SCRATCH_DB),
         "--sysmon-plugin", "cjm-capability-monitor-nvidia",
         "--actor", ACTOR,
         "--output", str(DECOMP_OUT), "--yes"],
        cwd=DECOMP, capture_output=True, text=True)
    wall = time.monotonic() - t0
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        raise SystemExit(f"decomp run failed rc={proc.returncode}")
    print(f"   wall {wall:.1f}s")

    manifest = json.loads(DECOMP_OUT.read_text())
    run_id = manifest["run_id"]
    n_sources = len(manifest["sources"])
    assert n_sources == 1, n_sources
    print(f"   run_id {run_id}; {manifest['sources'][0]['segment_count']} segments")

    # ---- journal assertions (the run window) ----
    rows = journal_rows(journal_db, "seq > ?", (cursor,))
    by_type = {}
    for r in rows:
        by_type.setdefault(r["event_type"], []).append(r)

    # 1. Liveness verifiably absent — over the WHOLE journal, not just the window.
    all_liveness = journal_rows(
        journal_db, "event_type IN ('progress_changed','resource_snapshot')")
    assert not all_liveness, f"liveness leaked into the journal: {len(all_liveness)} rows"

    # 2. RUN bracketing + actor + I14 verify rows.
    started = [r for r in by_type.get("run_started", []) if r["run_id"] == run_id]
    finished = [r for r in by_type.get("run_finished", []) if r["run_id"] == run_id]
    assert len(started) == 1 and len(finished) == 1, (len(started), len(finished))
    assert started[0]["actor"] == ACTOR and finished[0]["actor"] == ACTOR
    fin_payload = json.loads(finished[0]["payload"])
    assert fin_payload["status"] == "completed", fin_payload
    verifies = [r for r in by_type.get("verify_outcome", []) if r["run_id"] == run_id]
    assert len(verifies) == n_sources, len(verifies)
    for v in verifies:
        p = json.loads(v["payload"])
        assert p["found"] is True and p["ok"] is True, p

    # 3. Every job row in the run carries the run_id; admission 1:1 with jobs.
    st_rows = [r for r in by_type.get("state_transition", [])]
    job_ids = {r["job_id"] for r in st_rows}
    untagged = [r for r in st_rows if r["run_id"] != run_id]
    assert not untagged, f"{len(untagged)} state transitions missing the run_id"
    adm_rows = by_type.get("admission_decided", [])
    adm_jobs = {r["job_id"] for r in adm_rows}
    assert adm_jobs == job_ids, (
        f"admission/job mismatch: {len(adm_jobs)} admissions vs {len(job_ids)} jobs")
    assert all(r["run_id"] == run_id for r in adm_rows)
    assert len(adm_rows) == len(job_ids), "expected exactly one admission row per job"

    # 4. Terminal snapshots rehydrate at volume (the `_history` rider).
    terminals = [r for r in st_rows
                 if json.loads(r["payload"]).get("to") in ("completed", "failed", "cancelled")]
    assert len(terminals) == len(job_ids), (len(terminals), len(job_ids))
    failed = [r for r in terminals if json.loads(r["payload"])["to"] != "completed"]
    assert not failed, f"{len(failed)} non-completed jobs in the volume run"
    for r in terminals[:25]:
        snap = json.loads(r["payload"])["job_snapshot"]
        assert snap["id"] == r["job_id"] and snap["completed_at"], snap

    print(f"   journal: {len(rows)} rows in window — {len(job_ids)} jobs, "
          f"{len(adm_rows)} admissions, {len(verifies)} verify, liveness 0  ✓")

    # ---- correction-core parity on the scratch graph ----
    print("== correction parity run (prune + worklist on scratch) ==")
    t0 = time.monotonic()
    corr_journal = CORRECTION / ".cjm/journal.db"
    corr_cursor = 0
    if corr_journal.exists():
        with sqlite3.connect(corr_journal) as con:
            corr_cursor = con.execute("SELECT COALESCE(MAX(seq),0) FROM journal").fetchone()[0]
    proc = subprocess.run(
        [str(ENVS / "cjm-transcript-correction-core/bin/cjm-transcript-correction-core"),
         "run", str(DECOMP_OUT), "--actor", ACTOR, "--yes",
         "--output", "/tmp/stage7_volume_correction.json"],
        cwd=CORRECTION, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        raise SystemExit(f"correction run failed rc={proc.returncode}")
    print(f"   wall {time.monotonic() - t0:.1f}s")

    corr = json.loads(Path("/tmp/stage7_volume_correction.json").read_text())
    src = corr["sources"][0]
    assert src["empty_segments"] == BASELINE_EMPTY, (
        f"empty parity broke: {src['empty_segments']} != {BASELINE_EMPTY}")
    assert src["transcriber_divergences"] == BASELINE_DIVERGENCE, (
        f"divergence parity broke: {src['transcriber_divergences']} != {BASELINE_DIVERGENCE}")
    crows = journal_rows(corr_journal, "seq > ? AND run_id = ?",
                         (corr_cursor, corr["run_id"]))
    ctypes = {r["event_type"] for r in crows}
    assert {"run_started", "run_finished"} <= ctypes, ctypes
    print(f"   parity: {src['empty_segments']} empty + "
          f"{src['transcriber_divergences']} divergence == baselines  ✓")
    print(f"   correction journal: {len(crows)} run-tagged rows ({sorted(ctypes)})")

    SCRATCH_DB.unlink(missing_ok=True)
    print("== stage-7 volume regression: PASS ==")


if __name__ == "__main__":
    main()
