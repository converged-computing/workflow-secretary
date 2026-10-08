"""Snakemake as a workflow provider: one rule at a time from a pinned snakemake-wrappers catalog."""

from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from ..base import Tool, WorkflowProvider
from .catalog import DEFAULT_VERSION, Catalog

DEFAULT_CACHE = (
    Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    / "workflow-secretary"
)

INSTRUCTIONS = """### SNAKEMAKE
Steps are Snakemake rules, appended to a Snakefile that is managed for you: never
reference or edit it directly (view_snakefile shows it). Every rule gets a log path
automatically; do not set one. Cores are set by the environment.

Tools:
- search_wrappers: keyword search of the pinned snakemake-wrappers catalog.
- get_wrapper_details: README, conda packages and the example rule of one wrapper.
  The input/output key names and params in the example rule are exactly what you
  must pass to execute_wrapper. REQUIRED before any execute_wrapper call.
- execute_wrapper: append a rule that uses a catalog wrapper and run it now.
- execute_rule: append a custom shell rule and run it now, for glue steps with no
  suitable wrapper. Only system binaries are on PATH; scientific tools are available
  only through wrappers or the rule's conda_packages.
- create_sample_sheet: for more than one sample, write input/samples.csv and define
  SAMPLES and samples_df in the Snakefile. Later rules can then use the {sample}
  wildcard in input and output paths to run once per sample.
- delete_rule: remove a rule that succeeded but should not be in the workflow.
- view_snakefile: the current Snakefile.

Rules:
- You MUST prioritize wrappers over custom shell rules.
- Rule names must be unique, valid Python identifiers (snake_case, no spaces).
- A rule that fails is removed from the Snakefile automatically, so never delete a
  failed rule; fix the arguments and call again with the same name."""

EXECUTE_DOC = """
inputs: mapping of input names to paths. Staged files are relative to WORK_DIR/input/
(e.g. "samples/A.fastq"); prior step outputs are prefixed with 'steps/NN_rulename/'.
A value may be a list of paths. Use {} for a rule with no inputs.
output: mapping of output names to paths relative to this step's directory.
params: tool parameters as a mapping. Use {} for none.
threads: threads for the rule (1 if unsure).
Every rule gets an automatic log path. Always check 'success' in the result."""


def default_wrappers_dir(version: str) -> Path:
    """Cache location for a wrappers checkout at one version."""
    return DEFAULT_CACHE / f"snakemake-wrappers-{version}"


def _conda_frontend() -> Optional[str]:
    """Return 'mamba' or 'conda' depending on which is on PATH."""
    if shutil.which("mamba"):
        return "mamba"
    if shutil.which("conda"):
        return "conda"
    return None


def _wildcards(paths: list[str]) -> list[str]:
    """Sorted wildcard names used in paths, e.g. ['sample'] for 'x/{sample}.bam'."""
    found = set()
    for path in paths:
        found.update(re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", path))
    return sorted(found)


def _snakemake_version(binary: str) -> Optional[str]:
    """Version reported by the snakemake executable that runs the workflow."""
    try:
        out = subprocess.run(
            [binary, "--version"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return None


class SnakemakeProvider(WorkflowProvider):
    """Owns the accumulating Snakefile and runs it after every appended rule."""

    name = "snakemake"

    def __init__(
        self,
        workdir: Path,
        catalog: Catalog,
        input_dir: Optional[Path] = None,
        cores: int = 1,
        conda_frontend: Optional[str] = None,
        conda_prefix: Optional[Path] = None,
        snakemake: Optional[str] = None,
        run_fn: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ):
        super().__init__(workdir, input_dir, run_fn)
        self.catalog = catalog
        self.cores = cores
        # The frontend is recorded either way, but only an explicit choice reaches
        # snakemake: recent releases ignore --conda-frontend with a warning.
        self._explicit_frontend = conda_frontend
        self.conda_frontend = conda_frontend or _conda_frontend()
        self.conda_prefix = Path(conda_prefix).resolve() if conda_prefix else None
        self.snakemake = snakemake or shutil.which("snakemake") or "snakemake"

        # Each rule: {name, text, step_dir (relative to workdir), resolved_outputs}
        self._rules: list[dict[str, Any]] = []
        self._header = ""
        self._parallel_mode = False

    # CLI

    @classmethod
    def add_arguments(cls, parser) -> None:
        parser.add_argument(
            "--wrappers", help="wrappers checkout (default: user cache)"
        )
        parser.add_argument(
            "--wrappers-version", default=DEFAULT_VERSION, help="wrappers release tag"
        )
        parser.add_argument("--cores", type=int, default=1, help="cores for snakemake")
        parser.add_argument(
            "--conda-frontend",
            choices=["conda", "mamba"],
            help="passed to snakemake only when given",
        )
        parser.add_argument(
            "--conda-prefix", help="shared conda environment cache across runs"
        )

    @classmethod
    def from_args(cls, args, workdir: Path, input_dir: Optional[Path]):
        wrappers = (
            Path(args.wrappers)
            if args.wrappers
            else default_wrappers_dir(args.wrappers_version)
        )
        return cls(
            workdir=workdir,
            catalog=Catalog.ensure(wrappers, args.wrappers_version),
            input_dir=input_dir,
            cores=args.cores,
            conda_frontend=args.conda_frontend,
            conda_prefix=Path(args.conda_prefix) if args.conda_prefix else None,
        )

    @classmethod
    def add_prepare_arguments(cls, parser) -> None:
        parser.add_argument(
            "--wrappers-version", default=DEFAULT_VERSION, help="wrappers release tag"
        )
        parser.add_argument("--dest", help="checkout directory (default: user cache)")

    @classmethod
    def prepare(cls, args) -> int:
        """Clone (if needed) and index the wrappers at a version."""
        dest = (
            Path(args.dest)
            if args.dest
            else default_wrappers_dir(args.wrappers_version)
        )
        catalog = Catalog.ensure(dest, args.wrappers_version)
        print(
            f"{catalog.path}  version={catalog.version}  commit={catalog.commit}  "
            f"wrappers={len(catalog.index)}"
        )
        return 0

    # Interface

    @property
    def snakefile_path(self) -> Path:
        """Location of the accumulating Snakefile."""
        return self.workdir / "Snakefile"

    @property
    def history(self) -> list[dict[str, Any]]:
        return [
            {"step": i + 1, "name": r["name"], "step_dir": r["step_dir"]}
            for i, r in enumerate(self._rules)
        ]

    def instructions(self) -> str:
        return INSTRUCTIONS

    def layout(self) -> dict[str, str]:
        return {
            **super().layout(),
            "snakefile": "WORK_DIR/Snakefile — accumulating workflow, appended each step",
        }

    def environment(self) -> dict[str, Any]:
        return {"wrapper_version": self.catalog.version}

    def versions(self) -> dict[str, Any]:
        return {
            "snakemake": _snakemake_version(self.snakemake),
            "wrappers": {
                "version": self.catalog.version,
                "commit": self.catalog.commit,
            },
            "conda_frontend": self.conda_frontend,
        }

    def artifacts(self) -> dict[str, Path]:
        return {"Snakefile": self.snakefile_path}

    def tools(self) -> list[Tool]:
        return [
            Tool(
                "search_wrappers",
                "Keyword search of the snakemake-wrappers catalog over path, name, "
                "description, category and conda packages, e.g. 'bwa', 'samtools sort'.",
                {"query": str},
                lambda a: self.search_wrappers(a.get("query", "")),
            ),
            Tool(
                "get_wrapper_details",
                "Full documentation for one wrapper: README, conda packages, and the example "
                "rule whose input/output names and params execute_wrapper must follow. "
                "Required before execute_wrapper.",
                {"wrapper_path": str},
                lambda a: self.get_wrapper_details(a.get("wrapper_path", "")),
            ),
            Tool(
                "view_snakefile",
                "The current Snakefile and its rule count.",
                {},
                lambda a: self.view_snakefile(),
            ),
            Tool(
                "create_sample_sheet",
                "Write input/samples.csv from rows and define SAMPLES and samples_df in the "
                "Snakefile. Use for workflows with multiple samples. Every row must have the "
                "same keys and one key must be 'sample'; paths are relative to input/.",
                {"samples": list},
                lambda a: self.create_sample_sheet(a.get("samples") or []),
                kind="action",
            ),
            Tool(
                "execute_wrapper",
                "Append a rule that uses a catalog wrapper (wrapper_path as returned by "
                "search_wrappers, e.g. 'bio/bwa/mem') and run it now. Outputs go to "
                "WORK_DIR/steps/NN_rulename/. A failed rule is removed automatically."
                + EXECUTE_DOC,
                {
                    "rule_name": str,
                    "wrapper_path": str,
                    "inputs": dict,
                    "output": dict,
                    "params": dict,
                    "threads": int,
                },
                lambda a: self.execute_wrapper(
                    a.get("rule_name", ""),
                    a.get("wrapper_path", ""),
                    a.get("inputs") or {},
                    a.get("output") or {},
                    a.get("params") or {},
                    a.get("threads") or 1,
                ),
                kind="action",
            ),
            Tool(
                "execute_rule",
                "Append a custom shell rule and run it now, for glue steps with no suitable "
                "wrapper. The shell command uses Snakemake placeholders like {input.reads} "
                "and {output.bam}. Only system binaries are on PATH; scientific tools are "
                "available only through wrappers or conda_packages (a list of conda package "
                "names, [] for none). A failed rule is removed automatically."
                + EXECUTE_DOC,
                {
                    "rule_name": str,
                    "inputs": dict,
                    "output": dict,
                    "shell": str,
                    "params": dict,
                    "threads": int,
                    "conda_packages": list,
                },
                lambda a: self.execute_rule(
                    a.get("rule_name", ""),
                    a.get("inputs") or {},
                    a.get("output") or {},
                    a.get("shell", ""),
                    a.get("params") or {},
                    a.get("threads") or 1,
                    a.get("conda_packages") or [],
                ),
                kind="action",
            ),
            Tool(
                "delete_rule",
                "Remove a named rule that succeeded but should not be in the workflow, with "
                "its step directory. Failed rules are already removed; do not delete them.",
                {"rule_name": str},
                lambda a: self.delete_rule(a.get("rule_name", "")),
                kind="action",
            ),
        ]

    # Snakefile assembly

    @staticmethod
    def _section(name: str, resolved: dict) -> list[str]:
        """Render an input or output section with quoted values."""
        lines = [f"    {name}:"]
        for k, v in resolved.items():
            if isinstance(v, list):
                lines.append(f"        {k}=[{', '.join(repr(i) for i in v)}],")
            else:
                lines.append(f"        {k}={v!r},")
        return lines

    def _rule_lines(
        self,
        rule_name: str,
        inputs: dict,
        output: dict,
        params: dict,
        threads: int,
        step_dir: Path,
    ) -> tuple[list[str], dict, dict]:
        """Render the shared part of a rule; return (lines, resolved_inputs, resolved_outputs)."""
        resolved_inputs = self.resolve_inputs(inputs)
        resolved_outputs = self.resolve_outputs(output, step_dir)
        lines = [f"rule {rule_name}:"]
        if resolved_inputs:
            lines += self._section("input", resolved_inputs)
        lines += self._section("output", resolved_outputs)
        if params:
            lines.append("    params:")
            for k, v in params.items():
                lines.append(f"        {k}={v!r},")
        # Snakemake requires the log to use the same wildcards as the outputs.
        outputs = [
            p
            for v in resolved_outputs.values()
            for p in (v if isinstance(v, list) else [v])
        ]
        log_name = ".".join([rule_name] + [f"{{{w}}}" for w in _wildcards(outputs)])
        lines.append(f"    log: {str(self.logs_dir / f'{log_name}.log')!r}")
        lines.append(f"    threads: {max(int(threads or 1), 1)}")
        return lines, resolved_inputs, resolved_outputs

    def _rebuild_snakefile(self) -> None:
        """Write header, a 'rule all' targeting the last rule's outputs, then every rule."""
        parts = []
        if self._header:
            parts.append(self._header.strip())
        if self._rules:
            target = ["rule all:", "    input:"]
            for path in self._rules[-1]["resolved_outputs"]:
                if self._parallel_mode and "{sample}" in path:
                    target.append(f"        expand({path!r}, sample=SAMPLES),")
                else:
                    target.append(f"        {path!r},")
            parts.append("\n".join(target))
        for r in self._rules:
            parts.append(r["text"].strip())
        self.snakefile_path.write_text("\n\n".join(parts) + "\n")

    # Execution

    def command(self) -> list[str]:
        """The snakemake command line used for every execution."""
        cmd = [
            self.snakemake,
            "--snakefile",
            str(self.snakefile_path),
            "--cores",
            str(self.cores),
            "--wrapper-prefix",
            self.catalog.prefix,
            "--use-conda",
        ]
        if self._explicit_frontend:
            cmd += ["--conda-frontend", self._explicit_frontend]
        if self.conda_prefix:
            cmd += ["--conda-prefix", str(self.conda_prefix)]
        return cmd

    def _execute(
        self,
        rule_name: str,
        kind: str,
        wrapper: Optional[str],
        lines: list[str],
        resolved_outputs: dict,
        step_dir: Path,
    ) -> dict[str, Any]:
        """Append the rule, run snakemake, and remove the rule again if it failed."""
        outputs = []
        for v in resolved_outputs.values():
            outputs += v if isinstance(v, list) else [v]
        text = "\n".join(lines)
        self._rules.append(
            {
                "name": rule_name,
                "text": text,
                "step_dir": str(step_dir.relative_to(self.workdir)),
                "resolved_outputs": outputs,
            }
        )
        step_dir.mkdir(parents=True, exist_ok=True)
        self._rebuild_snakefile()

        success, response = self.run(
            self.command(),
            rule_name=rule_name,
            kind=kind,
            wrapper=wrapper,
            rule_text=text,
        )
        response["step_dir"] = str(step_dir.relative_to(self.workdir))
        if not success:
            self._remove(rule_name)
            response["work_dir_listing"] = self.list_work_dir()["items"]
            response["note"] = (
                f"The failed rule '{rule_name}' was removed automatically. Fix the arguments "
                "and call again; the same rule name may be reused."
            )
        return response

    def _remove(self, rule_name: str) -> Optional[dict]:
        """Drop a rule and its step directory, then rewrite the Snakefile."""
        idx = next(
            (i for i, r in enumerate(self._rules) if r["name"] == rule_name), None
        )
        if idx is None:
            return None
        rule = self._rules.pop(idx)
        self.remove_step_dir(rule["step_dir"])
        self._rebuild_snakefile()
        return rule

    def _precheck(self, rule_name: str, inputs: dict, output: dict) -> Optional[str]:
        """Shared validation for both execute calls."""
        if not rule_name.isidentifier():
            return f"Rule name '{rule_name}' must be a valid Python identifier (snake_case)."
        if any(r["name"] == rule_name for r in self._rules):
            return (
                f"Rule '{rule_name}' already exists. Pick a new name, or use "
                f"delete_rule('{rule_name}') first."
            )
        if not output:
            return "At least one output is required."
        return self.check_names("Input", inputs) or self.check_names("Output", output)

    # Tools

    def search_wrappers(self, query: str) -> dict[str, Any]:
        """Keyword search over the wrapper catalog."""
        results = self.catalog.search(query)
        return {"success": True, "count": len(results), "results": results}

    def get_wrapper_details(self, wrapper_path: str) -> dict[str, Any]:
        """Full catalog entry for one wrapper."""
        entry = self.catalog.get(wrapper_path)
        if not entry:
            return {
                "success": False,
                "error": f"Wrapper '{wrapper_path}' not found in index.",
            }
        return {"success": True, **entry}

    def view_snakefile(self) -> dict[str, Any]:
        """Current Snakefile content and rule count."""
        if not self.snakefile_path.exists():
            return {"success": True, "content": "", "rule_count": 0}
        content = self.snakefile_path.read_text()
        count = sum(
            1 for line in content.splitlines() if line.strip().startswith("rule ")
        )
        return {"success": True, "content": content, "rule_count": count}

    def create_sample_sheet(self, samples: list) -> dict[str, Any]:
        """Write input/samples.csv and add SAMPLES and samples_df to the Snakefile header."""
        if not samples or not isinstance(samples[0], dict):
            return {
                "success": False,
                "error": "samples must be a non-empty list of objects.",
            }
        keys = list(samples[0].keys())
        if "sample" not in keys:
            return {"success": False, "error": "The keys must include 'sample'."}
        if any(
            not isinstance(row, dict) or list(row.keys()) != keys for row in samples
        ):
            return {
                "success": False,
                "error": "Every row must have exactly the same keys.",
            }
        out_path = self.input_root / "samples.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(samples)
        self._header = (
            "import pandas as pd\n"
            "samples_df = pd.read_csv('input/samples.csv').set_index('sample', drop=False)\n"
            "SAMPLES = samples_df['sample'].tolist()"
        )
        self._parallel_mode = True
        self._rebuild_snakefile()
        return {
            "success": True,
            "message": "Sample sheet created and header updated.",
            "usage": (
                "SAMPLES (list of sample names) and samples_df (indexed by sample) are now "
                "defined. Use '{sample}' in input and output paths to run a rule per sample."
            ),
        }

    def execute_wrapper(
        self,
        rule_name: str,
        wrapper_path: str,
        inputs: dict,
        output: dict,
        params: dict,
        threads: int,
    ) -> dict[str, Any]:
        """Append and run a rule that uses a catalog wrapper or meta-wrapper."""
        entry = self.catalog.get(wrapper_path)
        if not entry:
            return {
                "success": False,
                "error": f"Wrapper '{wrapper_path}' not found in index.",
            }
        inputs, output, params = inputs or {}, output or {}, params or {}
        err = self._precheck(rule_name, inputs, output)
        if err:
            return {"success": False, "error": err}
        step_dir = self.next_step_dir(rule_name)
        lines, r_in, r_out = self._rule_lines(
            rule_name, inputs, output, params, threads, step_dir
        )
        err = self.check_paths(r_in, r_out)
        if err:
            return {"success": False, "error": err}
        directive = "meta_wrapper" if entry["type"] == "meta_wrapper" else "wrapper"
        lines.append(f"    {directive}: {entry['path']!r}")
        return self._execute(
            rule_name, "wrapper", entry["path"], lines, r_out, step_dir
        )

    def execute_rule(
        self,
        rule_name: str,
        inputs: dict,
        output: dict,
        shell: str,
        params: dict,
        threads: int,
        conda_packages: list,
    ) -> dict[str, Any]:
        """Append and run a custom shell rule, optionally with its own conda environment."""
        if not shell or not shell.strip():
            return {"success": False, "error": "shell must be a non-empty command."}
        inputs, output, params = inputs or {}, output or {}, params or {}
        err = self._precheck(rule_name, inputs, output)
        if err:
            return {"success": False, "error": err}
        step_dir = self.next_step_dir(rule_name)
        lines, r_in, r_out = self._rule_lines(
            rule_name, inputs, output, params, threads, step_dir
        )
        err = self.check_paths(r_in, r_out)
        if err:
            return {"success": False, "error": err}
        if conda_packages:
            step_dir.mkdir(parents=True, exist_ok=True)
            env_path = step_dir / "env.yaml"
            env_path.write_text(
                yaml.safe_dump(
                    {
                        "channels": ["conda-forge", "bioconda", "nodefaults"],
                        "dependencies": list(conda_packages),
                    }
                )
            )
            lines.append(f"    conda: {str(env_path.relative_to(self.workdir))!r}")
        lines.append(f"    shell: {shell!r}")
        return self._execute(rule_name, "shell", None, lines, r_out, step_dir)

    def delete_rule(self, rule_name: str) -> dict[str, Any]:
        """Remove a named rule and its step directory from the workflow."""
        if not rule_name:
            return {"success": False, "error": "rule_name is required."}
        rule = self._remove(rule_name)
        if rule is None:
            return {
                "success": False,
                "error": f"Rule '{rule_name}' not found in history.",
            }
        return {
            "success": True,
            "rule_name": rule["name"],
            "step_dir": rule["step_dir"],
        }
