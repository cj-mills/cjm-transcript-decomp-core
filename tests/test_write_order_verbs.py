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
from types import SimpleNamespace

import pytest
from cjm_context_graph_layer.journal import journal_extend
from cjm_context_graph_layer.testing import crash_variants, ProofStep, run_proof
from cjm_transcript_decomp_core import respine as RS, retire as R
from cjm_transcript_decomp_core.graph import build_extension_payload
from cjm_transcript_decomp_core.models import (DecompManifest, DecompSegment, DecompSourceRecord,
                                               SegmentVariant)
from cjm_transcript_graph_schema.schema import external_transcriber_name, source_node_id
from cjm_transcription_core.emission import emit_source_graph
from cjm_transcription_core.models import RunManifest, SegmentRecord, SourceResult

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
    ProofStep("emit", emit, ("source-emission", "derivation"), unit=True, retried=True),
    ProofStep("spine_old", spine_old, ("spine-extension",), retried=True),
    ProofStep("spine_new", spine_new, ("spine-extension",), retried=True),
    ProofStep("retire", retire, ("spine-retire",)),
    ProofStep("compact", compact, ("spine-compaction",), retried=True),
]


@pytest.fixture(scope="module")
def uncrashed(tmp_path_factory):
    return run_proof(tmp_path_factory.mktemp("uncrashed"), MANIFESTS, STEPS)


def test_the_decomp_verbs_are_proven_by_a_step(uncrashed, respined):
    from cjm_transcript_decomp_core.pipeline import decomp_replay_handlers
    assert set(decomp_replay_handlers()) <= {v for s in STEPS + RESPINE for v in s.ops}
    assert all(uncrashed["counts"]) and all(respined["counts"])


# A compaction crashed between its fact and its storage move completes on the re-run: the journaled
# fact's deletes join the plan (ruling 9d57273a, closing FINDING dbe14101)
@pytest.mark.parametrize("variant", crash_variants(STEPS), ids=lambda v: f"{v[0]}@{v[1]}")
def test_a_crashed_verb_heals_to_the_uncrashed_run(tmp_path, uncrashed, variant):
    crashed = run_proof(tmp_path, MANIFESTS, STEPS, variant)
    assert crashed["counts"] == uncrashed["counts"]
    assert crashed["journal"] == uncrashed["journal"]


def _respine_source():
    segs = [SegmentRecord(index=i, start=300.0 * i, end=300.0 * (i + 1), duration=300.0,
                          segment_path=f"/cuts/r{i}.mp3", model_input_path=f"/cache/r{i}.wav",
                          model_input_hash=f"sha256:proof-rwav{i}",
                          transcripts={"whisper": {"job_id": f"r{i}", "text": t, "metadata": {}}})
            for i, t in enumerate(CHUNK_TEXT)]
    return SourceResult(source_path="/media/proof-respine.mp3", duration=900.0, vad_chunk_count=6,
                        batch_key="bk", content_hash=R_HASH, segments=segs)


def _cut(pseg, first, words, transcriber):  # One fine Segment per word, evenly over a 300 s chunk
    span = 300.0 / len(words)
    out, at = [], 0
    for k, w in enumerate(words):
        out.append(DecompSegment(index=first + k, text=w, start_time=300.0 * pseg + span * k,
                                 end_time=300.0 * pseg + span * (k + 1), chunk_start=span * k,
                                 chunk_end=span * (k + 1), vad_chunk_index=k, pseg_index=pseg,
                                 variants=[SegmentVariant(transcriber, w, at, at + len(w))],
                                 text_from=transcriber))
        at += len(w) + 1
    return out


async def respine_emit(ctx):
    await emit_source_graph(ctx.queue, ctx.graph_id, _respine_source(), {"whisper": "sha256:proof-cfg"},
                            "run-proof-respine", **_j(ctx))


async def respine_decomp(ctx):  # The live spine and the lineage a respine resolves through
    src = _respine_source()
    runs = ctx.run_dir / "runs"
    tm_path = RunManifest(run_id="run-proof-respine", created_at=1.0,
                          config={"transcriber_capabilities": ["whisper"]}, capabilities=R_CFG,
                          sources=[src]).save(runs / "run-proof-respine.json")
    segs = [s for p, t in enumerate(CHUNK_TEXT) for s in _cut(p, 2 * p, t.split(), "whisper")]
    nodes, edges, ids = build_extension_payload(src.to_dict(), R_CFG, H_LIVE, "whisper", segs,
                                                split_policy="proof/v1")
    DecompManifest(run_id="decomp-proof-live", created_at=2.0,
                   config={"text_from": "whisper", "sentence_split": False}, source_manifest=str(tm_path),
                   sources=[DecompSourceRecord(source_node_id=R_SRC, source_path=src.source_path,
                                               title="proof-respine", segment_count=len(ids["segments"]),
                                               segment_ids=ids["segments"], skeleton_config_hash=H_LIVE)],
                   skeleton_config_hash=H_LIVE, split_policy="proof/v1").save(runs / "decomp-proof-live.json")
    await journal_extend(ctx.queue, ctx.graph_id, nodes, edges, verb="spine-extension", actor=ACTOR,
                         args={"source_id": R_SRC, "segments": len(ids["segments"])}, **_j(ctx))


async def respine(ctx):  # Chunk 1 respined from the paste: three segments for its two
    async def rederive(queue, cfg, d_src, si, transcribers, text_from, event_spans=None, pseg_indices=None):
        return None, _cut(1, 0, PASTE.split(), external_transcriber_name(MODEL))

    RS.decompose_source = rederive
    RS.collect_capability_info = lambda manager, stack: {}
    RS.new_run_id = lambda: "decomp-proof-respine"
    RS.new_landing_run_id = lambda: "run-proof-landing"
    runs = ctx.run_dir / "runs"
    chunk = await RS.resolve_chunk_context(ctx.queue, ctx.graph_id, source_id=R_SRC, skeleton_selector=None,
                                           runs_dir=runs, chunk=1)
    await RS.respine_chunk(SimpleNamespace(unload_capability=lambda iid: None), ctx.queue,
                           lambda manager, ids, configs=None: None, graph_id=ctx.graph_id,
                           journal_path=ctx.journal, ctx=chunk, text=PASTE, model_id=MODEL, prompt_hash="",
                           text_source="paste", reason="proof", actor=ACTOR, runs_dir=runs)


# THE RESPINE SCENARIO (ruling 3b76b63c, its third site): a source emitted, its live spine cut over
# three chunks with both cores' run manifests beside it, then chunk 1 respined from a paste -- the
# landing, the extension with its Derivation (ONE unit inside the chain's writer), the fact -- and
# RE-RUN after a crash. The policy stack (VAD / FA / SEG) is faked in the child: the re-derivation
# returns fixed segments. The run's minted ids are pinned there, as a minted uuid is abstracted.
H_LIVE = "sha256:proof0000000000000000000000000000000000000000000000000000000live"
R_HASH = "sha256:proof-respine-src"
R_SRC = source_node_id(R_HASH)
R_CFG = {"whisper": {"config_hash": "sha256:proof-cfg"}}
CHUNK_TEXT = ["alpha beta.", "gamma delta.", "epsilon zeta."]
PASTE = "gamma delta eta."
MODEL = "proof-model"

RESPINE = [
    ProofStep("emit", respine_emit, ("source-emission", "derivation"), unit=True, retried=True),
    ProofStep("decomp", respine_decomp, ("spine-extension",), retried=True),
    ProofStep("respine", respine, ("transcript-landing", "spine-extension", "derivation", "chunk-respine"),
              retried=True, units=(1, 2, 1)),
]


@pytest.fixture(scope="module")
def respined(tmp_path_factory):
    return run_proof(tmp_path_factory.mktemp("respined"), MANIFESTS, RESPINE)


@pytest.mark.parametrize("variant", crash_variants(RESPINE), ids=lambda v: f"{v[0]}@{v[1]}")
def test_a_crashed_respine_heals_to_the_uncrashed_run(tmp_path, respined, variant):
    crashed = run_proof(tmp_path, MANIFESTS, RESPINE, variant)
    assert crashed["counts"] == respined["counts"]
    assert crashed["journal"] == respined["journal"]
