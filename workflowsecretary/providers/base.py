"""The interface every workflow tool provider implements, plus the machinery they share.

A provider turns the agent's tool calls into steps of a workflow run by one
workflow tool (Snakemake, Nextflow, a plain shell, ...). The task, recorder and
CLI only talk to this interface. What is shared, because the agent's view of a
run is the same whatever runs it:

    WORK_DIR/
    ├── input/      staged input data, symlinked, read only
    ├── steps/      one directory per executed step, NN_stepname/
    └── logs/       one log per step

Inputs are addressed relative to input/ or as steps/NN_stepname/file, outputs
relative to the step's own directory. A provider adds whatever files its tool
needs (a Snakefile, a pipeline script) next to these.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

# Tail of stdout/stderr returned to the agent per execution.
OUTPUT_LIMIT = 8000

PATH_CONVENTIONS = {
    "input_files": (
        "Relative to WORK_DIR/input/. "
        "Example: 'samples/A.fastq' resolves to WORK_DIR/input/samples/A.fastq"
    ),
    "output_files": (
        "Relative to this step's directory. "
        "Example: 'A.bam' resolves to WORK_DIR/steps/NN_stepname/A.bam"
    ),
    "chained_inputs": (
        "To use a prior step's output as input, prefix with 'steps/NN_stepname/'. "
        "Example: 'steps/01_bwa_mem_A/A.bam'"
    ),
    "never_use": (
        "Absolute paths, environment variable names, or bare filenames "
        "without the correct prefix."
    ),
}


@dataclass
class Tool:
    """A tool a provider offers the agent. The task adapts it to the model backend.

    fn takes the argument mapping and returns a JSON-serializable result with a
    'success' key. kind is 'read' for tools that only inspect, 'action' for tools
    that change the workflow; action tools can be gated on user confirmation.
    """

    name: str
    description: str
    schema: dict = field(default_factory=dict)
    fn: Callable[[dict], dict] = lambda args: {"success": True}
    kind: str = "read"


def tail(text: Optional[str], limit: int = OUTPUT_LIMIT) -> str:
    """Return at most the last `limit` characters of text."""
    text = text or ""
    return text if len(text) <= limit else "...(truncated)...\n" + text[-limit:]


def listing(root: Path) -> list[dict[str, Any]]:
    """Recursive sorted listing of a directory with type and size."""
    items = []
    for p in sorted(root.rglob("*")):
        items.append(
            {
                "path": str(p.relative_to(root)),
                "type": "dir" if p.is_dir() else "file",
                "size_bytes": p.stat().st_size if p.is_file() else 0,
            }
        )
    return items


class WorkflowProvider(ABC):
    """One workflow tool, as seen by the agent.

    Subclasses implement `history`, `tools` and `instructions`, and usually
    `versions`, `artifacts` and the CLI hooks. The helpers below cover the
    shared layout so a provider only has to render and run its own steps.
    """

    # Registry key and the name shown to the agent, e.g. "snakemake".
    name: str = "workflow"

    def __init__(
        self,
        workdir: Path,
        input_dir: Optional[Path] = None,
        run_fn: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ):
        self.workdir = Path(workdir).resolve()
        self.input_dir = Path(input_dir).resolve() if input_dir else None
        self._run_fn = run_fn
        # Every execution, successful or not: {step, success, returncode, seconds, ...}
        self.attempts: list[dict[str, Any]] = []

    # Interface

    @property
    @abstractmethod
    def history(self) -> list[dict[str, Any]]:
        """Ordered steps currently in the workflow: [{step, name, step_dir}, ...]."""

    @abstractmethod
    def tools(self) -> list[Tool]:
        """The provider's tools. The task adds get_environment, list_input_dir,
        list_work_dir and finish itself."""

    @abstractmethod
    def instructions(self) -> str:
        """The provider's section of the system prompt: how its tools fit the
        shared strategy, naming rules, and anything the agent must know."""

    def environment(self) -> dict[str, Any]:
        """Extra fields for get_environment, e.g. a catalog version."""
        return {}

    def versions(self) -> dict[str, Any]:
        """Versions of everything that can change a run's behavior, for the record."""
        return {}

    def artifacts(self) -> dict[str, Path]:
        """Files to copy into the run record, by name, e.g. {'Snakefile': path}."""
        return {}

    # CLI hooks

    @classmethod
    def add_arguments(cls, parser) -> None:
        """Add the provider's options to the `run` command."""

    @classmethod
    def from_args(cls, args, workdir: Path, input_dir: Optional[Path]):
        """Build the provider from parsed `run` arguments."""
        return cls(workdir, input_dir)

    @classmethod
    def add_prepare_arguments(cls, parser) -> None:
        """Add the provider's options to the `prepare` command."""

    @classmethod
    def prepare(cls, args) -> int:
        """One-time setup before runs, e.g. fetching a catalog. Return an exit code."""
        print(f"{cls.name}: nothing to prepare.")
        return 0

    # Layout

    @property
    def input_root(self) -> Path:
        return self.workdir / "input"

    @property
    def steps_dir(self) -> Path:
        return self.workdir / "steps"

    @property
    def logs_dir(self) -> Path:
        return self.workdir / "logs"

    def setup(self) -> None:
        """Create input/, steps/ and logs/ and symlink input data into input/."""
        for d in (self.input_root, self.steps_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)
        if not self.input_dir:
            return
        for src in self.input_dir.rglob("*"):
            dest = self.input_root / src.relative_to(self.input_dir)
            if src.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
            elif src.is_file() and not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.symlink_to(src.resolve())

    def layout(self) -> dict[str, str]:
        """The layout as described to the agent. Providers may add entries."""
        return {
            "input": "WORK_DIR/input/  — staged input data (read-only, do not write here)",
            "steps": "WORK_DIR/steps/  — one subdirectory per executed step (NN_stepname/)",
            "logs": "WORK_DIR/logs/   — per-step log files written automatically",
        }

    # Common tools, exposed by the task

    def get_environment(self) -> dict[str, Any]:
        """Layout, path conventions, provider details and step history."""
        return {
            "success": True,
            "provider": self.name,
            "work_dir_structure": self.layout(),
            "path_conventions": PATH_CONVENTIONS,
            **self.environment(),
            "steps_completed": len(self.history),
            "history": self.history,
        }

    def list_input_dir(self) -> dict[str, Any]:
        """Recursive listing of WORK_DIR/input/."""
        if not self.input_root.exists():
            return {
                "success": False,
                "error": "Input staging directory does not exist.",
            }
        return {"success": True, "root": "input/", "items": listing(self.input_root)}

    def list_work_dir(self) -> dict[str, Any]:
        """Recursive listing of WORK_DIR/steps/ with step history."""
        if not self.steps_dir.exists():
            return {"success": False, "error": "Steps directory does not exist."}
        return {
            "success": True,
            "root": "steps/",
            "items": listing(self.steps_dir),
            "history": self.history,
        }

    # Path handling shared by step-based providers

    def inside(self, path: Path, root: Path) -> bool:
        """True if path is lexically inside root; staged inputs are symlinks, so links are not followed."""
        return Path(os.path.normpath(os.path.abspath(path))).is_relative_to(root)

    def resolve_input(self, item: Any) -> str:
        """Resolve one input path against steps/ or input/."""
        item = str(item)
        if item.startswith("steps/"):
            return str(self.workdir / item)
        return str(self.input_root / item)

    def resolve_inputs(self, inputs: dict) -> dict:
        """Resolve every input value, keeping list values as lists."""
        return {
            k: (
                [self.resolve_input(i) for i in v]
                if isinstance(v, list)
                else self.resolve_input(v)
            )
            for k, v in inputs.items()
        }

    def resolve_outputs(self, output: dict, step_dir: Path) -> dict:
        """Resolve every output value relative to the step directory."""
        return {
            k: (
                [str(step_dir / str(i)) for i in v]
                if isinstance(v, list)
                else str(step_dir / str(v))
            )
            for k, v in output.items()
        }

    @staticmethod
    def check_names(kind: str, mapping: dict) -> Optional[str]:
        """Return an error if any input/output name is not a valid Python identifier."""
        bad = [k for k in mapping if not str(k).isidentifier()]
        if bad:
            return (
                f"{kind} names must be valid Python identifiers (e.g. 'reads', 'r1'); "
                f"got {bad}."
            )
        return None

    def check_paths(
        self, resolved_inputs: dict, resolved_outputs: dict
    ) -> Optional[str]:
        """Return an error if inputs leave WORK_DIR or outputs leave WORK_DIR/steps."""
        for v in resolved_inputs.values():
            for p in v if isinstance(v, list) else [v]:
                if not self.inside(Path(p), self.workdir):
                    return f"Security Error: Read access denied. Path '{p}' is outside WORK_DIR."
        for v in resolved_outputs.values():
            for p in v if isinstance(v, list) else [v]:
                if not self.inside(Path(p), self.steps_dir):
                    return (
                        f"Security Error: Write access denied. Path '{p}' is outside "
                        "WORK_DIR/steps/."
                    )
        return None

    def next_step_dir(self, step_name: str) -> Path:
        """Next numbered step directory, e.g. steps/03_samtools_sort."""
        numbers = [0]
        for h in self.history:
            try:
                numbers.append(int(Path(h["step_dir"]).name.split("_")[0]))
            except (ValueError, IndexError):
                pass
        return self.steps_dir / f"{max(numbers) + 1:02d}_{step_name}"

    def remove_step_dir(self, step_dir: str) -> None:
        """Delete a step directory, refusing anything outside steps/."""
        path = (self.workdir / step_dir).resolve()
        if (
            path.exists()
            and self.inside(path, self.steps_dir)
            and path != self.steps_dir
        ):
            shutil.rmtree(path, ignore_errors=True)

    # Execution

    def run(self, cmd: list[str], **attempt: Any) -> tuple[bool, dict[str, Any]]:
        """Run a command in WORK_DIR, log the attempt, and return (success, response).

        The response carries the tail of stdout and stderr and a listing of
        steps/, which is what every execute tool returns to the agent. Extra
        keyword arguments are stored with the attempt, e.g. step and kind.
        """
        start = time.time()
        result = self._run_fn(cmd, capture_output=True, text=True, cwd=self.workdir)
        seconds = round(time.time() - start, 3)
        success = result.returncode == 0
        self.attempts.append(
            {
                **attempt,
                "success": success,
                "returncode": result.returncode,
                "seconds": seconds,
            }
        )
        response = {
            "success": success,
            "returncode": result.returncode,
            "stdout": tail(result.stdout),
            "stderr": tail(result.stderr),
            "work_dir_listing": listing(self.steps_dir),
        }
        return success, response
