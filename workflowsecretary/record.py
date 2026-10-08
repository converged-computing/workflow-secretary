"""Everything about one run, written as a single directory."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Optional

# Longest tool result kept in the record; the agent still receives the full result.
RESULT_LIMIT = 20000


def _version(dist: str) -> Optional[str]:
    """Installed version of a distribution, or None."""
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def _sha256(path: Path) -> str:
    """Streaming sha256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def outputs_manifest(steps_dir: Path) -> list[dict[str, Any]]:
    """Path, size and sha256 of every file under steps/."""
    files = []
    if not steps_dir.exists():
        return files
    for p in sorted(steps_dir.rglob("*")):
        if p.is_file():
            files.append(
                {
                    "path": str(p.relative_to(steps_dir.parent)),
                    "size_bytes": p.stat().st_size,
                    "sha256": _sha256(p),
                }
            )
    return files


class Recorder:
    """Accumulates tool calls during a run and writes the run directory at the end."""

    def __init__(
        self, run_id: str, goal: str, context: str = "", plan: Optional[str] = None
    ):
        self.run_id = run_id
        self.goal = goal
        self.context = context
        self.plan = plan
        self.started = time.time()
        self.calls: list[dict[str, Any]] = []

    def call(self, tool: str, args: dict, result: Any, seconds: float) -> None:
        """Record one tool call and its (possibly truncated) result."""
        text = json.dumps(result, default=str)
        self.calls.append(
            {
                "n": len(self.calls) + 1,
                "time": round(time.time() - self.started, 3),
                "tool": tool,
                "args": args,
                "seconds": round(seconds, 3),
                "result": (
                    text
                    if len(text) <= RESULT_LIMIT
                    else text[:RESULT_LIMIT] + "...(truncated)"
                ),
            }
        )

    def versions(self, provider, runner) -> dict[str, Any]:
        """Versions of everything that can change a run's behavior."""
        return {
            "workflow-secretary": _version("workflow-secretary"),
            "behalf": _version("behalf"),
            "python": platform.python_version(),
            "runner": type(runner).__name__,
            "model": getattr(runner, "model", None),
            "provider": provider.name,
            **provider.versions(),
        }

    def write(
        self, out_dir: Path, provider, runner, finish: Optional[dict], backend: str
    ) -> Path:
        """Write run.json, the provider's artifacts, and the outputs manifest; return the run dir."""
        run_dir = Path(out_dir) / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        for name, path in provider.artifacts().items():
            if Path(path).exists():
                shutil.copy(path, run_dir / name)
        record = {
            "id": self.run_id,
            "plan": self.plan,
            "goal": self.goal,
            "context": self.context,
            "backend": backend,
            "provider": provider.name,
            "versions": self.versions(provider, runner),
            "workdir": str(provider.workdir),
            "seconds": round(time.time() - self.started, 3),
            "status": (finish or {}).get("status", "no-finish"),
            "finish": finish,
            "steps": provider.history,
            "attempts": provider.attempts,
            "outputs": outputs_manifest(provider.steps_dir),
            "calls": self.calls,
        }
        (run_dir / "run.json").write_text(json.dumps(record, indent=2, default=str))
        return run_dir
