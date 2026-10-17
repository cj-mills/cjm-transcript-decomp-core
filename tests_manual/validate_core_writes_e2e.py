#!/usr/bin/env python
"""The three transcript cores on the write order, end to end (eeda5977 L3 slice E2, ruling 6683f49e).

On a real SQLite graph store, every write path E2 moved onto the writer runs live -- each op
journaled, then applied through its verb's replay handler:
- transcription: a source emission + the Derivation it plans (one write window), curation acts
  (file, confirm, order, rename = merge, bind urls, add + retract a reference);
- decomp: two spines extended by hand over the emitted rendition, the old one retired (its plan
  and dependents gate inside the writer), then compacted OP FIRST (the fact's deletes journaled
  before the storage move, `compact_family`);
- correction: a session, a text correction and its re-edit, a boundary shift, a mark, the
  pipeline's prune (the new `prune` verb), the session's status.
Then a FRESH db is rebuilt from the journal (replay, cursor stamped) and must equal live ELEMENT
FOR ELEMENT through the layer's comparator; the live cursor must sit at the family's end, the
compaction's fact must carry its deletes, and the family must name none of the compacted ids
outside those deletes. The crash proof through every verb is E3's.

Usage (an env holding all three cores + the graph capability's manifest):
    python tests_manual/validate_core_writes_e2e.py <manifests_dir> <scratch_dir>
Exit 0 = clean, 1 = a difference (printed).
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

from cjm_context_graph_layer.compact import scan_references
from cjm_context_graph_layer.journal import journal_extend
from cjm_context_graph_layer.ops import GraphCursorStore
from cjm_context_graph_layer.rebuild import diff_dbs, open_store, rebuild_db
from cjm_context_graph_primitives.journal import read_journal
from cjm_context_graph_primitives.writer import family_name, ORIGIN, read_tail
from cjm_transcript_correction_core.graph import (commit_boundary_shift_correction,
                                                  commit_mark_correction, commit_prune_correction,
                                                  commit_text_correction, set_session_status,
                                                  start_session)
from cjm_transcript_decomp_core import retire as R
from cjm_transcription_core.curation import (add_reference, bind_source_urls, confirm_collection,
                                             file_sources, rename_collection, retract_reference,
                                             set_collection_order)
from cjm_transcription_core.emission import emit_source_graph
from cjm_transcription_core.models import SegmentRecord, SourceResult

H_OLD = "sha256:e2e0000000000000000000000000000000000000000000000000000000000old"
H_NEW = "sha256:e2e0000000000000000000000000000000000000000000000000000000000new"
ACTOR = "agent:e2e-write-order"


def fixture_source():  # One source, two chunks, one transcriber (the emission tests' shape)
    recs = [SegmentRecord(index=i, start=280.0 * i, end=280.0 * (i + 1), duration=280.0,
                          segment_path=f"/cuts/s{i}.mp3", model_input_path=f"/cache/s{i}.wav",
                          model_input_hash=f"sha256:e2e-wav{i}",
                          transcripts={"whisper": {"job_id": f"j{i}", "text": f"chunk {i}",
                                                   "metadata": {}}})
            for i in range(2)]
    src = SourceResult(source_path="/media/e2e-write-order.mp3", duration=560.0, vad_chunk_count=2,
                       batch_key="bk", content_hash="sha256:e2e-write-order-src", segments=recs)
    return src, {"whisper": "sha256:e2e-cfg"}


def spine(skeleton, rendition, source_id, born):  # Three fine Segments PART_OF the rendition
    nodes, edges = [], []
    for k in range(3):
        sid = f"e2e-seg-{skeleton[-3:]}-{k}"
        nodes.append({"id": sid, "label": "Segment", "sources": [], "created_at": born + k,
                      "updated_at": born + k,
                      "properties": {"skeleton_hash": skeleton, "split_policy": "e2e/v1", "index": k,
                                     "source_id": source_id, "start_time": 10.0 * k,
                                     "end_time": 10.0 * (k + 1), "text": f"segment {k}"}})
        edges.append({"id": f"e2e-part-{sid}", "source_id": sid, "target_id": rendition,
                      "relation_type": "PART_OF", "properties": {}})
    return nodes, edges


async def live(manifests, db, j, archive):  # Every E2 write path, live
    async with open_store(db, manifests) as g:
        q, gid = g.queue, g.graph_id
        src, hashes = fixture_source()
        rec = await emit_source_graph(q, gid, src, hashes, "run-e2e", journal_path=j)
        source_id = rec["source_node_id"]
        rendition = (await R.source_rendition_ids(q, gid, source_id))[0]
        for skeleton, born in ((H_OLD, 1_000.0), (H_NEW, 2_000.0)):
            nodes, edges = spine(skeleton, rendition, source_id, born)
            await journal_extend(q, gid, nodes, edges, journal_path=j, verb="spine-extension",
                                 actor=ACTOR, args={"source_id": source_id, "segments": 3})
        filed = await file_sources(q, gid, "E2E Write Order", [source_id], ACTOR, journal_path=j,
                                   status="proposed")
        coll = filed["collection_node_id"]
        await confirm_collection(q, gid, coll, ACTOR, journal_path=j)
        await set_collection_order(q, gid, coll, [source_id], ACTOR, journal_path=j)
        await rename_collection(q, gid, coll, "E2E Write Order Renamed", ACTOR, journal_path=j)
        await bind_source_urls(q, gid, [{"source_id": source_id, "url": "https://example.org/v"}],
                               ACTOR, journal_path=j)
        ref = await add_reference(q, gid, source_id, label="E2E ref", url="https://example.org/r",
                                  actor=ACTOR, journal_path=j)
        await retract_reference(q, gid, ref["reference_id"], actor=ACTOR, journal_path=j)
        new = [f"e2e-seg-{H_NEW[-3:]}-{k}" for k in range(3)]
        sess = await start_session(q, gid, [source_id], journal_path=j, actor=ACTOR)
        c1 = await commit_text_correction(q, gid, source_id, new[0], "fixed", sess.id,
                                          old_text="segment 0", actor=ACTOR, journal_path=j)
        await commit_text_correction(q, gid, source_id, new[0], "fixed again", sess.id,
                                     old_text="fixed", supersedes_id=c1, actor=ACTOR, journal_path=j)
        await commit_boundary_shift_correction(q, gid, source_id, new[0], new[1], "segment", "push",
                                               sess.id, actor=ACTOR, journal_path=j)
        await commit_mark_correction(q, gid, source_id, {"kind": "segment", "segment_id": new[2]},
                                     "needs-review", sess.id, actor=ACTOR, journal_path=j)
        await commit_prune_correction(q, gid, source_id, [SimpleNamespace(id=new[2])], sess.id,
                                      actor=ACTOR, journal_path=j)
        await set_session_status(q, gid, sess.id, "completed", journal_path=j, actor=ACTOR)
        await R.retire_spine(q, gid, source_id, H_OLD, reason="e2e", successor=H_NEW,
                             journal_path=j, actor=ACTOR)
        compacted = await R.compact_retired(q, gid, [source_id], journal_path=j,
                                            archive_dir=archive, label="e2e", actor=ACTOR)
        cursor = await GraphCursorStore(q, gid).read(family_name(j))
        return compacted, cursor


def main(argv):
    manifests, tmp = argv[0], Path(argv[1])
    tmp.mkdir(parents=True, exist_ok=True)
    for f in tmp.rglob("*"):
        if f.is_file():
            f.unlink()
    a_db, b_db, j = str(tmp / "live.db"), str(tmp / "rebuilt.db"), str(tmp / "live.writes.jsonl")
    compacted, cursor = asyncio.run(live(manifests, a_db, j, str(tmp / "archive")))
    verbs = [r["verb"] for r in read_journal(j)]
    counts = asyncio.run(rebuild_db(b_db, j, manifests))
    report = asyncio.run(diff_dbs(a_db, b_db, manifests))
    end = read_tail(j, ORIGIN)[-1][1]
    at_end = (cursor.segment, cursor.offset, cursor.ordinal) == (end.segment, end.offset, end.ordinal)
    ids = [i for t in compacted["targets"] for i in t["segment_ids"]]
    fact = [r for r in read_journal(j) if r["verb"] == R.COMPACTION_VERB]
    fact_ok = len(fact) == 1 and sorted(fact[0]["deletes"]["node_ids"]) == sorted(ids) and len(ids) == 3
    dangling = len(scan_references(j, ids))
    print(f"journaled ops ({len(verbs)}): {', '.join(sorted(set(verbs)))}")
    print(f"compaction: {compacted['report'].summary()}")
    print(f"rebuild: {counts}")
    print(f"diff clean {report['clean']} (nodes {report['nodes']['count_a']}/{report['nodes']['count_b']}, "
          f"edges {report['edges']['count_a']}/{report['edges']['count_b']}; "
          f"attributed {report.get('attributed')}) · cursor at the family's end {at_end} · "
          f"fact carries its deletes {fact_ok} · dangling {dangling}")
    if not report["clean"]:
        print(report)
    ok = report["clean"] and at_end and fact_ok and dangling == 0
    print("CLEAN" if ok else "DIFFERENCE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
