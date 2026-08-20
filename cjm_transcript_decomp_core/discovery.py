"""Capability-role discovery by manifest surface match — the journaling-by-
default mechanism (sysmon here; the decomp core REQUIRES its graph capability,
so only the monitor role auto-discovers). CARRIED COPY of the transcription
TUI's candidates.py pair, kept deliberately close to verbatim: this second
consumer is the N=2 promotion signal for cjm-substrate-tui-kit (cross-repo
move v2 is the vehicle), and a drifted copy would fork the discovery contract
before the move lands."""

import json
from pathlib import Path
from typing import Any, Dict, Optional


def manifests_with_method(
    manifests_dir: str,  # Capability manifests directory (the core CLI's --manifests-dir)
    method: str,         # Structural-surface method that identifies the role
) -> Dict[str, Dict[str, Any]]:  # capability name -> its manifest `code` section
    """Enumerate installed capabilities whose structural surface lists `method`.

    Capabilities qualify by SURFACE, not by name — the same signal the
    substrate's adapter auto-binding matches against a task protocol, read
    cheaply off the manifest json (no worker spawn). Role key in use here:
    `get_system_status` (system monitor). Adapter unit manifests carry no
    `code` section and are skipped; unreadable files are skipped rather than
    failing enumeration.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for f in sorted(Path(manifests_dir).glob("*.json")):
        try:
            manifest = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        code = manifest.get("code") if isinstance(manifest, dict) else None
        if not isinstance(code, dict):
            continue
        methods = ((code.get("structural_surface") or {}).get("methods") or [])
        if any(m.get("name") == method for m in methods):
            out[code.get("name") or f.stem] = code
    return out


def discover_capability(
    manifests_dir: str,  # Capability manifests directory
    method: str,         # Surface method that identifies the role
) -> Optional[str]:  # First matching capability name (sorted), or None
    """Pick a DEFAULT capability for a role by surface match.

    Journaling-by-default's mechanism (transcription-TUI drive-1 finding):
    when the runtime has a monitor installed, the TUI should use it without
    being told — forgetting must take an explicit opt-out, not a forgotten
    flag. Sorted-first keeps the pick deterministic when several qualify; the
    operator's persisted choice (state.py) wins over discovery.
    """
    names = sorted(manifests_with_method(manifests_dir, method))
    return names[0] if names else None
