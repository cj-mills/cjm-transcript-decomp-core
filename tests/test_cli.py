"""Tests for cjm_transcript_decomp_core.cli — parser smoke checks (no capabilities involved).

Projected from the cli notebook's parser-check cell at the golden-reference flip."""
from cjm_transcript_decomp_core.cli import build_parser


def test_run_defaults():
    p = build_parser()
    args = p.parse_args(["run", "m.json", "--yes", "--language", "English"])
    assert args.command == "run"
    assert args.manifest == "m.json"
    assert args.yes is True
    assert args.fa_capability == "cjm-capability-qwen3-forced-aligner"
    assert args.graph_capability == "cjm-capability-graph-sqlite"
    assert args.text_from is None  # authority defaults to the manifest's sole transcriber
    assert args.sysmon_capability is None


def test_text_from_and_sysmon_flags():
    p = build_parser()
    args = p.parse_args(["run", "m.json", "--text-from", "cjm-capability-voxtral-hf",
                         "--sysmon-capability", "cjm-capability-monitor-nvidia"])
    assert args.text_from == "cjm-capability-voxtral-hf"
    assert args.sysmon_capability == "cjm-capability-monitor-nvidia"
