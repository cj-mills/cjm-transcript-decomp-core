"""THE PER-VERB CRASH PROOF of the decomp core (eeda5977 leg L3 slice E3, rulings 6683f49e +
3de795e4), on the layer's harness (`cjm_context_graph_layer.testing.run_proof`).

A source emitted and two spines extended over its rendition (the pipeline's writes, RE-RUN after a
crash); the old spine retired (a human act, healed by the next writer); then the retired spine
compacted OP FIRST -- its fact journaled before the storage move -- and re-run after a crash, so the
move completes from every boundary. Each run's live db equals its journal's replay element for
element, and every crashed run's journal equals the uncrashed one, op for op.

Needs the graph capability: CJM_PROOF_MANIFESTS, else this repo's .cjm/manifests."""

import os
from pathlib import Path

import pytest
from cjm_context_graph_layer.journal import journal_extend
from cjm_context_graph_layer.testing import crash_variants, ProofStep, run_proof
from cjm_transcript_decomp_core import retire as R
from cjm_transcript_graph_schema.schema import source_node_id
from cjm_transcription_core.emission import emit_source_graph
from cjm_transcription_core.models import SegmentRecord, SourceResult

MANIFESTS = os.environ.get("CJM_PROOF_MANIFESTS") or str(Path(__file__).resolve().parents[1] / ".cjm" / "manifests")
pytestmark = pytest.mark.skipif(not (Path(MANIFESTS) / "cjm-capability-graph-sqlite.json").exists(),
                                reason=f"graph capability not installed at {MANIFESTS}")

ACTOR = "agent:proof"
HASH = "sha256:proof-decomp-src"
SRC = source_node_id(HASH)
H_OLD = "sha256:proof00000000000000000000000000000000000000000000000000000000old"
H_NEW = "sha256:proof00000000000000000000000000000000000000000000000000000000new"


def _j(ctx):
    return {"journal_path": ctx.journal}


async def emit(ctx):
    segs = [SegmentRecord(index=i, start=280.0 * i, end=280.0 * (i + 1), duration=280.0,
                          segment_path=f"/cuts/d{i}.mp3", model_input_path=f"/cache/d{i}.wav",
                          model_input_hash=f"sha256:proof-dwav{i}",
                          transcripts={"whisper": {"job_id": f"d{i}", "text": f"chunk {i}", "metadata": {}}})
            for i in range(2)]
    src = SourceResult(source_path="/media/proof-decomp.mp3", duration=560.0, vad_chunk_count=2,
                       batch_key="bk", content_hash=HASH, segments=segs)
    await emit_source_graph(ctx.queue, ctx.graph_id, src, {"whisper": "sha256:proof-cfg"}, "run-proof",
                            **_j(ctx))


async def _spine(ctx, skeleton, born):  # Three fine Segments PART_OF the rendition
    rendition = (await R.source_rendition_ids(ctx.queue, ctx.graph_id, SRC))[0]
    nodes, edges = [], []
    for k in range(3):
        sid = f"proof-seg-{skeleton[-3:]}-{k}"
        nodes.append({"id": sid, "label": "Segment", "sources": [], "created_at": born + k,
                      "updated_at": born + k,
                      "properties": {"skeleton_hash": skeleton, "split_policy": "proof/v1", "index": k,
                                     "source_id": SRC, "start_time": 10.0 * k, "end_time": 10.0 * (k + 1),
                                     "text": f"segment {k}"}})
        edges.append({"id": f"proof-part-{sid}", "source_id": sid, "target_id": rendition,
                      "relation_type": "PART_OF", "properties": {}})
    await journal_extend(ctx.queue, ctx.graph_id, nodes, edges, verb="spine-extension", actor=ACTOR,
                         args={"source_id": SRC, "segments": 3}, **_j(ctx))


async def spine_old(ctx):
    await _spine(ctx, H_OLD, 1_000.0)


async def spine_new(ctx):
    await _spine(ctx, H_NEW, 2_000.0)


async def retire(ctx):
    await R.retire_spine(ctx.queue, ctx.graph_id, SRC, H_OLD, reason="proof", successor=H_NEW,
                         actor=ACTOR, **_j(ctx))


async def compact(ctx):
    await R.compact_retired(ctx.queue, ctx.graph_id, [SRC], archive_dir=str(ctx.run_dir / "archive"),
                            label="proof", actor=ACTOR, **_j(ctx))


STEPS = [
    ProofStep("emit", emit, ("source-emission", "derivation"), retried=True),
    ProofStep("spine_old", spine_old, ("spine-extension",), retried=True),
    ProofStep("spine_new", spine_new, ("spine-extension",), retried=True),
    ProofStep("retire", retire, ("spine-retire",)),
    ProofStep("compact", compact, ("spine-compaction",), retried=True),
]


@pytest.fixture(scope="module")
def uncrashed(tmp_path_factory):
    return run_proof(tmp_path_factory.mktemp("uncrashed"), MANIFESTS, STEPS)


def test_the_decomp_verbs_are_proven_by_a_step(uncrashed):
    from cjm_transcript_decomp_core.pipeline import decomp_replay_handlers
    # chunk-respine's chain (landing -> extension -> Derivation -> transfer -> fact) is not yet a step
    assert set(decomp_replay_handlers()) - {v for s in STEPS for v in s.ops} == {"chunk-respine"}
    assert all(uncrashed["counts"])


# FINDING dbe14101: a compaction crashed between its fact and its storage move is never completed
# (every variant crashes the compact step at its one op); FINDING 249298fd: the seed emission's
# Derivation is lost by a crash at its first op -- strict, so the fixes flip these to failures
@pytest.mark.xfail(strict=True, reason="FINDING dbe14101 (and 249298fd)")
@pytest.mark.parametrize("variant", crash_variants(STEPS), ids=lambda v: f"{v[0]}@{v[1]}")
def test_a_crashed_verb_heals_to_the_uncrashed_run(tmp_path, uncrashed, variant):
    crashed = run_proof(tmp_path, MANIFESTS, STEPS, variant)
    assert crashed["counts"] == uncrashed["counts"]
    assert crashed["journal"] == uncrashed["journal"]
