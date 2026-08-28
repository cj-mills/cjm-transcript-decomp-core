"""Tests for the decomp-batch TUI's pure logic: manifest-format filtering, the
text-from default + grouping fold, coverage counts, and the hand-off argv
(everything below the paint path; the paint layer's verification is the
tests_manual pilot probe, per the TUI craft register)."""

import json
from pathlib import Path
from cjm_transcript_decomp_core.launch import (batch_argv, build_parser,
                                           event_split_batch_error,
                                           resolve_split_flags)
from cjm_transcript_decomp_core.runs import (DecompIndex, PropsetIndex,
                                            SourceRunIndex, group_batches)


def _src_manifest(run_id, created, transcribers):
    # Minimal but format-faithful transcription-core manifest (0.2.0 shape).
    return {"format": "cjm-transcription-core/run-manifest", "version": "0.2.0",
            "run_id": run_id, "created_at": created,
            "config": {"transcriber_capabilities": transcribers},
            "sources": [{"source_path": f"/tmp/{run_id}.mp3",
                         "segments": [{"index": 0}, {"index": 1}]}]}


def _dec_manifest(run_id, created, source_manifest):
    # Minimal but format-faithful decomp-core manifest (0.2.1 shape).
    return {"format": "cjm-transcript-decomp-core/run-manifest", "version": "0.2.1",
            "run_id": run_id, "created_at": created,
            "config": {"text_from": None},
            "source_manifest": source_manifest,
            "sources": [{"source_node_id": "abc123", "source_path": "/tmp/x.mp3",
                         "title": "x", "segment_count": 5, "segment_ids": []}]}


def test_format_filtering_and_order(tmp_path):
    # Both cores' manifests + junk share one runs dir; the format tag separates.
    (tmp_path / "a.json").write_text(json.dumps(_src_manifest("run_a", 100.0, ["w"])))
    (tmp_path / "b.json").write_text(json.dumps(_src_manifest("run_b", 200.0, ["w"])))
    (tmp_path / "d.json").write_text(
        json.dumps(_dec_manifest("decomp_x", 300.0, str(tmp_path / "a.json"))))
    (tmp_path / "junk.json").write_text("not json")
    (tmp_path / "foreign.json").write_text(json.dumps({"run_id": "x", "sources": []}))
    src = SourceRunIndex(str(tmp_path))
    assert src.load() == 2
    assert [m["run_id"] for m in src.runs] == ["run_b", "run_a"]  # newest first
    assert SourceRunIndex.segment_count(src.runs[0]) == 2
    dec = DecompIndex(str(tmp_path))
    assert dec.load() == 1
    assert dec.runs[0]["run_id"] == "decomp_x"


def test_decomp_coverage_counts(tmp_path):
    a = tmp_path / "a.json"
    a.write_text(json.dumps(_src_manifest("run_a", 1.0, ["w"])))
    (tmp_path / "d1.json").write_text(json.dumps(_dec_manifest("decomp_1", 2.0, str(a))))
    (tmp_path / "d2.json").write_text(json.dumps(_dec_manifest("decomp_2", 3.0, str(a))))
    dec = DecompIndex(str(tmp_path))
    dec.load()
    assert dec.counts_by_source_manifest() == {str(a.resolve()): 2}


def test_transcribers_and_default_text_from():
    multi = _src_manifest("m", 1.0, ["cap-light", "cap-acc"])
    assert SourceRunIndex.transcribers(multi) == ["cap-light", "cap-acc"]
    # The confirmed pair lands [lightweight, accuracy] -> last = accuracy slot.
    assert SourceRunIndex.default_text_from(multi) == "cap-acc"
    legacy = {"config": {"transcriber_capability": "cap-solo"}}
    assert SourceRunIndex.transcribers(legacy) == ["cap-solo"]
    assert SourceRunIndex.default_text_from(legacy) == "cap-solo"
    assert SourceRunIndex.default_text_from({"config": {}}) is None


def test_group_batches_preserves_order():
    # The key is (text_from, graph_db_path) — everything invocation-wide.
    groups = group_batches([("m1", ("acc", "/g1.db")), ("m2", ("solo", "/g1.db")),
                            ("m3", ("acc", "/g1.db")), ("m4", ("acc", "/g2.db"))])
    assert groups == [(("acc", "/g1.db"), ["m1", "m3"]),
                      (("solo", "/g1.db"), ["m2"]),
                      (("acc", "/g2.db"), ["m4"])]  # same authority, other db: own group
    assert group_batches([]) == []


def test_batch_argv_renders_the_handoff():
    args = build_parser().parse_args(["--runs-dir", "/data/runs", "--force",
                                      "--actor", "tui:test"])
    argv = batch_argv({"text_from": "cap-acc", "graph_db_path": "/tmp/g.db",
                       "manifests": ["a.json", "b.json"]},
                      args, "cjm-capability-monitor-nvidia")
    assert argv[:3] == ["run", "a.json", "b.json"]
    assert "--yes" in argv
    assert argv[argv.index("--text-from") + 1] == "cap-acc"
    assert argv[argv.index("--graph-db-path") + 1] == "/tmp/g.db"
    assert argv[argv.index("--sysmon-capability") + 1] == "cjm-capability-monitor-nvidia"
    # 7dfd1177: decomp manifests pin to the SAME dir the TUI browsed.
    assert argv[argv.index("--output-dir") + 1] == "/data/runs"
    assert "--force" in argv and "--actor" in argv
    # Optionals stay OUT when unset (a copy-pasteable minimal command).
    bare = batch_argv({"text_from": None, "graph_db_path": None, "manifests": ["a.json"]},
                      build_parser().parse_args([]), None)
    for flag in ("--text-from", "--graph-db-path", "--sysmon-capability",
                 "--force", "--actor"):
        assert flag not in bare


def test_parser_defaults():
    args = build_parser().parse_args([])
    # 5daadfc4: None sentinels — main() resolves the workspace's runs/ +
    # .cjm/manifests when one is active, else the legacy cwd-relative defaults
    assert args.runs_dir is None
    assert args.manifests_dir is None
    assert args.workspace is None
    assert args.fa_capability == "cjm-capability-qwen3-forced-aligner"
    assert args.graph_capability == "cjm-capability-graph-sqlite"
    assert not args.plan_only and not args.no_sysmon and not args.force


def test_provenance_reads():
    # e087d059: the graph db derives from the manifest's capabilities block.
    m = {"capabilities": {"cjm-capability-graph-sqlite": {"db_path": "/data/g.db"},
                          "cjm-capability-whisper": {"db_path": None}},
         "sources": [{"source_path": "/media/ep1.mp3", "segments": [{}]},
                     {"source_path": "/media/ep2.mp3", "segments": []}]}
    assert SourceRunIndex.recorded_graph_db(m, "cjm-capability-graph-sqlite") == "/data/g.db"
    assert SourceRunIndex.recorded_graph_db(m, "cjm-capability-other") is None
    assert SourceRunIndex.recorded_graph_db({}, "cjm-capability-graph-sqlite") is None
    # 614dd647: source identity paints straight off the manifest.
    assert SourceRunIndex.source_names(m) == ["ep1", "ep2"]


def test_batch_argv_sentence_split_passthrough():
    # Sentence-split parses to a None sentinel and resolves ON when unset
    # (DEC 552bde8d): the resolved args render the explicit ON pair, and an
    # OFF toggle must ride the argv as --no-sentence-split — the core default
    # would silently re-enable it.
    args = resolve_split_flags(build_parser().parse_args(["--split-min-chunk-s", "0.7"]))
    assert args.sentence_split is True
    argv = batch_argv({"text_from": None, "graph_db_path": None,
                       "manifests": ["a.json"]}, args, None)
    assert "--sentence-split" in argv
    assert argv[argv.index("--split-min-chunk-s") + 1] == "0.7"
    off_args = build_parser().parse_args(["--no-sentence-split"])
    off = batch_argv({"text_from": None, "graph_db_path": None,
                      "manifests": ["a.json"]}, off_args, None)
    assert "--no-sentence-split" in off
    assert "--sentence-split" not in off and "--split-min-chunk-s" not in off


def test_batch_argv_renders_respine():
    """DEC 9241564f: the R toggle rides the hand-off argv as --respine — a
    fresh spine is a deliberate, VISIBLE act; off = absent (core default off)."""
    args = build_parser().parse_args(["--manifests-dir", "m", "--runs-dir", "r",
                                      "--respine"])
    argv = batch_argv({"text_from": None, "graph_db_path": None,
                       "manifests": ["a.json"]}, args, None)
    assert "--respine" in argv
    args_off = build_parser().parse_args(["--manifests-dir", "m", "--runs-dir", "r"])
    assert not args_off.respine
    assert "--respine" not in batch_argv({"text_from": None, "graph_db_path": None,
                                          "manifests": ["a.json"]}, args_off, None)


def test_resolve_split_flags_event_default_flip():
    """Verdict ad963c57: --event-split flips the unset sentence-split default
    OFF (the ratified respine shape); an explicit flag wins either way
    (composable stages, DEC 6cc10fb7). No pointer needed at parse time —
    per-source resolution owns it (DEC ae450551), so a bare --event-split
    launches the TUI armed."""
    bare = resolve_split_flags(build_parser().parse_args([]))
    assert bare.sentence_split is True
    armed = resolve_split_flags(build_parser().parse_args(["--event-split"]))
    assert armed.sentence_split is False
    assert armed.event_propset is None
    both = resolve_split_flags(build_parser().parse_args(
        ["--event-split", "--event-propset", "/sets/p", "--sentence-split"]))
    assert both.sentence_split is True


def test_batch_argv_renders_event_flags():
    """The event trio rides the hand-off argv ONLY when armed (core default
    off), and renders the propset pointer + carve classes explicitly — the
    printed argv is the full reproducibility contract."""
    args = resolve_split_flags(build_parser().parse_args(
        ["--event-split", "--event-propset", "/sets/propset_x",
         "--event-classes", "inhale", "exhale"]))
    argv = batch_argv({"text_from": None, "graph_db_path": None,
                       "manifests": ["a.json"]}, args, None)
    assert argv[argv.index("--event-propset") + 1] == "/sets/propset_x"
    i = argv.index("--event-classes")
    assert argv[i + 1:i + 3] == ["inhale", "exhale"]
    assert "--event-split" in argv
    # The flipped default rides too: the core is sentence-split default-ON.
    assert "--no-sentence-split" in argv
    off = batch_argv({"text_from": None, "graph_db_path": None,
                      "manifests": ["a.json"]},
                     resolve_split_flags(build_parser().parse_args([])), None)
    for flag in ("--event-split", "--event-propset", "--event-classes"):
        assert flag not in off
    # The GROUP's resolved propset wins over the CLI pin (per-source picker).
    per = batch_argv({"text_from": None, "graph_db_path": None,
                      "event_propset": "/sets/propset_y",
                      "manifests": ["a.json"]}, args, None)
    assert per[per.index("--event-propset") + 1] == "/sets/propset_y"


def test_event_split_hand_off_guard():
    """DEC ae450551: per-GROUP propsets (the per-source picker) make any batch
    width safe; the scripted --event-propset PIN is one pointer so it demands
    exactly one manifest; a group with neither pointer has nothing to carve
    from and refuses — silently-wrong carves must never leave the TUI."""
    pin = resolve_split_flags(build_parser().parse_args(
        ["--event-split", "--event-propset", "/sets/p"]))
    one = [{"text_from": None, "graph_db_path": None, "manifests": ["a.json"]}]
    assert event_split_batch_error(one, pin) is None
    wide = [{"text_from": None, "graph_db_path": None, "manifests": ["a.json"]},
            {"text_from": "cap", "graph_db_path": None,
             "manifests": ["b.json", "c.json"]}]
    err = event_split_batch_error(wide, pin)
    assert err is not None and "3 manifests" in err
    # Per-group propsets: the same width is SAFE — each group carries its set.
    armed = resolve_split_flags(build_parser().parse_args(["--event-split"]))
    grouped = [{**b, "event_propset": f"/sets/p{i}"} for i, b in enumerate(wide)]
    assert event_split_batch_error(grouped, armed) is None
    # A group with no pointer at all refuses.
    err = event_split_batch_error(wide, armed)
    assert err is not None and "no proposal set" in err
    # Unarmed, everything passes — the guard is event-split's alone.
    plain = resolve_split_flags(build_parser().parse_args([]))
    assert event_split_batch_error(wide, plain) is None


def test_propset_index_discovery_and_source_join(tmp_path):
    """PropsetIndex (DEC ae450551): format-filtered discovery under
    proposals/, latest-per-source head, content-hash preferred over path,
    and the tier-aware summary chip."""
    def _set(name, created, chash, path, counts, tier2=None):
        d = tmp_path / name
        d.mkdir()
        (d / "manifest.json").write_text(json.dumps({
            "format": "cjm-capability-pyannote/proposal-set-manifest",
            "proposal_set_id": name, "created_at": created,
            "source": {"path": path, "content_hash": chash},
            "counts": counts,
            **({"tier2_counts": tier2} if tier2 else {})}))
    _set("propset_old", 100.0, "sha256:aaa", "/media/ep1.mp3", {"inhale": 400})
    _set("propset_new", 200.0, "sha256:aaa", "/media/ep1.mp3",
         {"inhale": 500}, tier2={"inhale": 27})
    _set("propset_other", 300.0, "sha256:bbb", "/media/ep2.mp3", {"inhale": 9})
    (tmp_path / "junk").mkdir()
    (tmp_path / "junk" / "manifest.json").write_text("{not json")
    idx = PropsetIndex(str(tmp_path))
    assert idx.load() == 3
    sets = idx.for_source(content_hash="sha256:aaa")
    assert [m["proposal_set_id"] for m in sets] == ["propset_new", "propset_old"]
    # Path fallback joins when no hash matches; misses return empty.
    assert [m["proposal_set_id"] for m in idx.for_source(
        content_hash="sha256:zzz", source_path="/media/ep2.mp3")] == ["propset_other"]
    assert idx.for_source(content_hash="sha256:zzz") == []
    assert PropsetIndex.summary(sets[0]) == "pset_new 500+27t2"


def test_training_run_index_discovery_and_resolve(tmp_path):
    """DEC 1cfe6d0f (P propose-now): manifest-driven discovery with the
    forgiveness contract; newest-first by run id; pin > newest, and a pin
    matching nothing returns None LOUDLY instead of silently falling back
    to a different model (the b9717422 field-failure class)."""
    from cjm_transcript_decomp_core.runs import TrainingRunIndex
    d = tmp_path / "training-runs"

    def _run(run_id, classes, extra=None):
        rd = d / run_id
        rd.mkdir(parents=True)
        m = {"format": TrainingRunIndex.FORMAT, "run_id": run_id, "classes": classes}
        m.update(extra or {})
        (rd / "manifest.json").write_text(json.dumps(m))

    _run("trainrun_20260731_002436_6f803b12", ["speech", "inhale"])
    _run("trainrun_20260805_104926_73000552", ["click", "inhale"])
    (d / "junk").mkdir()
    (d / "junk" / "manifest.json").write_text("{not json")
    (d / "foreign").mkdir()
    (d / "foreign" / "manifest.json").write_text(json.dumps({"format": "other/thing"}))

    idx = TrainingRunIndex(str(d))
    assert idx.load() == 2
    assert idx.runs[0]["run_id"].endswith("73000552")  # newest first
    # No pin: newest run is only a default (the two-step confirm names it)
    assert idx.resolve()["run_id"].endswith("73000552")
    # Pin (full id or tail) wins regardless of recency
    assert idx.resolve("6f803b12")["run_id"].endswith("6f803b12")
    assert idx.resolve("trainrun_20260731_002436_6f803b12")["run_id"].endswith("6f803b12")
    # A pin matching nothing is a loud None, never a fallback
    assert idx.resolve("deadbeef") is None
    assert "6f803b12" in TrainingRunIndex.summary(idx.resolve("6f803b12"))
    # Empty dir: no runs, loud None
    empty = TrainingRunIndex(str(tmp_path / "nowhere"))
    assert empty.load() == 0 and empty.resolve() is None


def test_training_run_resolve_pick_ladder_and_describe(tmp_path):
    """Work item a13ebc66 (the propose training-run picker): the model
    ladder is PICK > PIN > NEWEST with the reason named alongside, and every
    miss is a loud None (a vanished pick, a pin matching nothing) — never a
    silent fall-through to a different model. describe() reads a picker
    row off the manifest alone: id · base · [classes] · headline eval ·
    dataset tail · date."""
    from cjm_transcript_decomp_core.runs import TrainingRunIndex
    d = tmp_path / "training-runs"

    def _run(run_id, classes, extra=None):
        rd = d / run_id
        rd.mkdir(parents=True)
        m = {"format": TrainingRunIndex.FORMAT, "run_id": run_id, "classes": classes}
        m.update(extra or {})
        (rd / "manifest.json").write_text(json.dumps(m))

    _run("trainrun_20260731_002436_6f803b12", ["speech", "inhale"],
         {"base_model": {"model_id": "pyannote/segmentation-3.0"},
          "dataset_id": "dataset_20260730_e53b8daa", "created_at": 1785000000.0,
          "eval": {"holdout": {"auroc": {"macro": 0.99659}},
                   "metrics": {"loss/val": 0.0634}}})
    _run("trainrun_20260805_104926_73000552", ["click", "inhale"],
         {"eval": {"metrics": {"loss/val": 0.0812}}})
    idx = TrainingRunIndex(str(d))
    assert idx.load() == 2
    # No pick, no pin: newest is only a default, and the reason says so
    run, why = idx.resolve_pick(None, None)
    assert run["run_id"].endswith("73000552") and why == "newest run, no pin"
    # The pin is the standing default over recency
    run, why = idx.resolve_pick(None, "6f803b12")
    assert run["run_id"].endswith("6f803b12") and why == "workspace pin"
    # A pick outranks the pin — by _path, full id, or tail
    older = idx.runs[1]
    for pick in (older["_path"], older["run_id"], "6f803b12"):
        run, why = idx.resolve_pick(pick, "73000552")
        assert run["run_id"].endswith("6f803b12") and why == "picked"
    run, why = idx.resolve_pick("73000552", "6f803b12")
    assert run["run_id"].endswith("73000552") and why == "picked"
    # Misses are LOUD: a vanished pick never falls through to the pin
    run, why = idx.resolve_pick("deadbeef", "6f803b12")
    assert run is None and "deadbeef" in why and "no longer on disk" in why
    run, why = idx.resolve_pick(None, "deadbeef")
    assert run is None and "matches nothing" in why
    empty = TrainingRunIndex(str(tmp_path / "nowhere"))
    empty.load()
    assert empty.resolve_pick(None, None) == (None, f"no training runs under {empty.training_runs_dir}")
    # describe(): the row facts a picker paints
    line = TrainingRunIndex.describe(older)
    assert line.startswith("trainrun_20260731_002436_6f803b12 · segmentation-3.0 · [speech,inhale]")
    assert "auroc 0.997" in line and "ds …e53b8daa" in line and "2026-07-" in line
    # No holdout block: the first metric is the headline; no base/ds: elided
    line = TrainingRunIndex.describe(idx.runs[0])
    assert "loss/val 0.081" in line and "ds …" not in line and "segmentation" not in line


def test_batch_argv_renders_one_pointer_per_source():
    """Multi-source event carve (decomp manifest 0.2.6): a group's resolved
    propset may be a LIST/TUPLE (one pointer per source of a multi-source
    run) — batch_argv renders every pointer under one --event-propset, in
    order, and the single-source str form renders exactly as before."""
    args = resolve_split_flags(build_parser().parse_args(
        ["--event-split", "--event-classes", "inhale"]))
    multi = batch_argv({"text_from": None, "graph_db_path": None,
                        "event_propset": ("/sets/p_src0", "/sets/p_src1"),
                        "manifests": ["ab.json"]}, args, None)
    i = multi.index("--event-propset")
    assert multi[i + 1:i + 3] == ["/sets/p_src0", "/sets/p_src1"]
    assert multi[i + 3] == "--event-classes"
    single = batch_argv({"text_from": None, "graph_db_path": None,
                         "event_propset": "/sets/p_only",
                         "manifests": ["a.json"]}, args, None)
    j = single.index("--event-propset")
    assert single[j + 1] == "/sets/p_only" and single[j + 2] == "--event-classes"
