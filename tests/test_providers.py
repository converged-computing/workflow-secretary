"""The provider interface: a minimal provider proves the task, recorder and CLI need nothing from Snakemake."""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from behalf import run_task

from workflowsecretary.providers import PROVIDERS, get_provider
from workflowsecretary.providers.base import Tool, WorkflowProvider
from workflowsecretary.record import Recorder
from workflowsecretary.scripted import ScriptedRunner
from workflowsecretary.task import WorkflowTask


class ShellProvider(WorkflowProvider):
    """Runs each step as a shell command in its own step directory. No workflow tool."""

    name = "shell"

    def __init__(self, workdir, input_dir=None, run_fn=subprocess.run):
        super().__init__(workdir, input_dir, run_fn)
        self._steps = []

    @property
    def history(self):
        return [
            {"step": i + 1, "name": s["name"], "step_dir": s["step_dir"]}
            for i, s in enumerate(self._steps)
        ]

    def instructions(self):
        return "### SHELL\nUse run_step to run one shell command per step."

    def versions(self):
        return {"shell": os.environ.get("SHELL", "sh")}

    def artifacts(self):
        return {"steps.json": self.workdir / "steps.json"}

    def tools(self):
        return [
            Tool(
                "run_step",
                "Run a shell command; {input} and {output} are resolved paths.",
                {"name": str, "inputs": dict, "output": dict, "command": str},
                lambda a: self.run_step(
                    a["name"], a.get("inputs") or {}, a["output"], a["command"]
                ),
                kind="action",
            )
        ]

    def run_step(self, name, inputs, output, command):
        if not name.isidentifier():
            return {"success": False, "error": "name must be an identifier"}
        step_dir = self.next_step_dir(name)
        r_in = self.resolve_inputs(inputs)
        r_out = self.resolve_outputs(output, step_dir)
        err = self.check_paths(r_in, r_out)
        if err:
            return {"success": False, "error": err}
        step_dir.mkdir(parents=True)
        cmd = command.format(
            **{f"input_{k}": v for k, v in r_in.items()},
            **{f"output_{k}": v for k, v in r_out.items()},
        )
        self._steps.append(
            {"name": name, "step_dir": str(step_dir.relative_to(self.workdir))}
        )
        (self.workdir / "steps.json").write_text(json.dumps(self._steps))
        success, response = self.run(["sh", "-c", cmd], name=name, command=cmd)
        if not success:
            self._steps.pop()
            self.remove_step_dir(str(step_dir.relative_to(self.workdir)))
        return response

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument("--shell-flag", default="x")


def make(tmp: Path) -> ShellProvider:
    data = tmp / "data"
    data.mkdir()
    (data / "a.txt").write_text("hello\n")
    p = ShellProvider(tmp / "work", input_dir=data)
    p.setup()
    return p


def test_registry_resolves_snakemake_lazily():
    assert "snakemake" in PROVIDERS
    cls = get_provider("snakemake")
    assert cls.name == "snakemake" and issubclass(cls, WorkflowProvider)
    try:
        get_provider("nope")
    except SystemExit as e:
        assert "unknown provider" in str(e)
    else:
        raise AssertionError("unknown provider must fail")


def test_base_layout_and_common_tools():
    with tempfile.TemporaryDirectory() as t:
        p = make(Path(t))
        assert (p.input_root / "a.txt").is_symlink() and p.steps_dir.is_dir()
        env = p.get_environment()
        assert env["provider"] == "shell" and env["history"] == []
        assert set(env["work_dir_structure"]) == {"input", "steps", "logs"}
        assert [i["path"] for i in p.list_input_dir()["items"]] == ["a.txt"]
        assert p.next_step_dir("one") == p.steps_dir / "01_one"


def test_provider_steps_chain_and_failures_roll_back():
    with tempfile.TemporaryDirectory() as t:
        p = make(Path(t))
        out = p.run_step(
            "upper",
            {"a": "a.txt"},
            {"o": "A.txt"},
            "tr a-z A-Z < {input_a} > {output_o}",
        )
        assert out["success"], out
        assert (p.steps_dir / "01_upper" / "A.txt").read_text() == "HELLO\n"
        out = p.run_step("bad", {}, {"o": "x"}, "exit 3")
        assert not out["success"] and out["returncode"] == 3
        assert [h["name"] for h in p.history] == ["upper"]
        assert not (p.steps_dir / "02_bad").exists()
        out = p.run_step(
            "copy",
            {"u": "steps/01_upper/A.txt"},
            {"o": "B.txt"},
            "cp {input_u} {output_o}",
        )
        assert out["success"] and (p.steps_dir / "02_copy" / "B.txt").exists()
        assert [a["success"] for a in p.attempts] == [True, False, True]
        assert (
            "Read access denied"
            in p.run_step("esc", {"x": "../../etc/passwd"}, {"o": "y"}, "true")["error"]
        )


def test_task_composes_prompt_and_tools_from_provider():
    with tempfile.TemporaryDirectory() as t:
        p = make(Path(t))
        task = WorkflowTask(p, Recorder("r", "g"))
        prompt = task.execute_system_prompt({"goal": "g"})
        assert "workflows with shell" in prompt and "### SHELL" in prompt
        assert "Snakefile" not in prompt and "wrapper" not in prompt
        names = [s.name for s in task.build_tools({"goal": "g"})]
        assert names == [
            "get_environment",
            "list_input_dir",
            "list_work_dir",
            "run_step",
            "finish",
        ]
        kinds = {s.name: (s.kind, s.confirm) for s in task.build_tools({"goal": "g"})}
        assert kinds["run_step"] == ("action", False) and kinds["finish"] == (
            "read",
            False,
        )
        assert (
            WorkflowTask(p, Recorder("r", "g"), confirm_actions=True)
            .build_tools({})[3]
            .confirm
        )


def test_scripted_run_and_record_with_another_provider():
    with tempfile.TemporaryDirectory() as t:
        p = make(Path(t))
        recorder = Recorder("run1", "shout", plan="unit")
        task = WorkflowTask(p, recorder)
        runner = ScriptedRunner(
            [
                {"tool": "get_environment"},
                {
                    "tool": "run_step",
                    "args": {
                        "name": "upper",
                        "inputs": {"a": "a.txt"},
                        "output": {"o": "A.txt"},
                        "command": "tr a-z A-Z < {input_a} > {output_o}",
                    },
                },
                {"tool": "list_work_dir"},
                {
                    "tool": "finish",
                    "args": {
                        "status": "complete",
                        "summary": "s",
                        "issues": [],
                        "reason": "r",
                    },
                },
            ]
        )
        outcome = asyncio.run(
            run_task(
                task,
                runner,
                manifest={"goal": "shout", "context": ""},
                confirm_fn=lambda n, a: True,
            )
        )
        run_dir = recorder.write(Path(t) / "runs", p, runner, outcome.result, "script")
        record = json.loads((run_dir / "run.json").read_text())
        assert (
            record["provider"] == "shell" and record["versions"]["provider"] == "shell"
        )
        assert "shell" in record["versions"] and record["status"] == "complete"
        assert record["steps"] == [
            {"step": 1, "name": "upper", "step_dir": "steps/01_upper"}
        ]
        assert record["outputs"][0]["path"] == "steps/01_upper/A.txt"
        assert json.loads((run_dir / "steps.json").read_text())[0]["name"] == "upper"


def test_cli_hooks_have_defaults():
    parser = argparse.ArgumentParser()
    ShellProvider.add_arguments(parser)
    args = parser.parse_args([])
    with tempfile.TemporaryDirectory() as t:
        p = ShellProvider.from_args(args, Path(t) / "w", None)
        assert isinstance(p, ShellProvider) and p.input_dir is None
    assert ShellProvider.prepare(args) == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"OK {name}")
    print("\nall provider tests passed")
