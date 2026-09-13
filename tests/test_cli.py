"""Tests for cjm_transcript_decomp_core.cli — parser smoke checks (no capabilities involved).

Projected from the cli notebook's parser-check cell at the golden-reference flip."""
from cjm_transcript_decomp_core.cli import build_parser


def test_run_defaults():
    p = build_parser()
    args = p.parse_args(["run", "m.json", "--yes", "--language", "English"])
    assert args.command == "run"
    assert args.manifests == ["m.json"]  # single manifest = a batch of one
    assert args.yes is True
    assert args.fa_capability == "cjm-capability-qwen3-forced-aligner"
    assert args.graph_capability == "cjm-capability-graph-sqlite"
    assert args.text_from is None  # authority defaults to the manifest's sole transcriber
    assert args.sysmon_capability is None
    # 5daadfc4: None sentinel — run_command resolves the workspace's runs/ when
    # one is active, else the legacy cwd-relative runs/; TUIs pin their browsed dir
    assert args.output_dir is None
    assert args.workspace is None


def test_text_from_and_sysmon_flags():
    p = build_parser()
    args = p.parse_args(["run", "m.json", "--text-from", "cjm-capability-voxtral-hf",
                         "--sysmon-capability", "cjm-capability-monitor-nvidia"])
    assert args.text_from == "cjm-capability-voxtral-hf"
    assert args.sysmon_capability == "cjm-capability-monitor-nvidia"


def test_batch_manifests():
    # Batch shape (work item 0ff6bf0f): N manifests, one invocation, one stack.
    p = build_parser()
    args = p.parse_args(["run", "a.json", "b.json", "c.json", "--yes",
                         "--text-from", "cjm-capability-voxtral-hf"])
    assert args.manifests == ["a.json", "b.json", "c.json"]
    assert args.text_from == "cjm-capability-voxtral-hf"


def test_spine_verbs_parse():
    # ruling a7617bd4: list-spines / retire-spine / retire-superseded / compact-spines
    p = build_parser()
    a = p.parse_args(["retire-spine", "--source", "Lecture 37", "--skeleton", "abc1", "--successor", "legacy",
                      "--reason", "prefer previous", "--graph-db-path", "/tmp/g.db"])
    assert a.command == "retire-spine" and a.successor == "legacy" and not a.unretire
    b = p.parse_args(["retire-superseded", "--collection", "GPU MODE", "--dry-run"])
    assert b.command == "retire-superseded" and b.reason == "superseded" and b.dry_run
    c = p.parse_args(["compact-spines", "--dry-run", "--prove-rebuild", "/tmp/fresh.db"])
    assert c.command == "compact-spines" and c.archive_dir is None and c.prove_rebuild == "/tmp/fresh.db"
    d = p.parse_args(["list-spines", "--dependents"])
    assert d.command == "list-spines" and d.dependents and d.source is None
