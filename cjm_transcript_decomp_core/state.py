"""Sidecar TUI state: last-used batch settings persisted across sessions (the
transcription TUI's state pattern — settings the operator picked once, the
sysmon choice and the graph db override, must never need retyping). The file
lives next to the manifests dir (one level up — .cjm/ in practice) so state is
per-project, not per-user."""

from pathlib import Path
from typing import Any, Dict

from cjm_substrate.utils.sidecar import SidecarState

STATE_BASENAME = "decomp-tui-state.json"


def state_path(
    manifests_dir: str,  # Capability manifests directory (the state key)
) -> Path:  # The sidecar state file (manifests dir's PARENT — .cjm/ in practice)
    """Where this project's TUI state lives."""
    return Path(manifests_dir).resolve().parent / STATE_BASENAME


def load_state(
    manifests_dir: str,  # Capability manifests directory (the state key)
) -> Dict[str, Any]:  # Persisted state ({} when absent/unreadable — never raises)
    """Read this project's persisted TUI state."""
    return SidecarState(state_path(manifests_dir)).load()


def save_state(
    manifests_dir: str,  # Capability manifests directory (the state key)
    **updates: Any,      # Keys to merge into the persisted state
) -> Dict[str, Any]:  # The merged state as written
    """Merge updates into the persisted state and write it back (best-effort:
    a read-only location must not break the batch hand-off)."""
    return SidecarState(state_path(manifests_dir)).save(**updates)
