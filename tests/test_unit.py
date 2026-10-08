"""Unit tests that need neither snakemake nor a model."""

import ast
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from behalf import run_task

from workflowsecretary.providers.snakemake import Catalog, SnakemakeProvider
from workflowsecretary.record import Recorder
from workflowsecretary.scripted import ScriptedRunner
from workflowsecretary.task import WorkflowTask

# Types behalf's ADK runner maps to real parameter types; anything else becomes str.
ADK_TYPES = {str, int, float, bool, list, dict}


def make_catalog(root: Path) -> Catalog:
    """A two-wrapper catalog on disk."""
    for path, name, pkgs in [
        ("bio/samtools/faidx", "samtools faidx", ["samtools =1.21"]),
        ("meta/bio/bwa_mapping", "bwa mapping", ["bwa"]),
    ]:
        d = root / path
        (d / "test").mkdir(parents=True)
        (d / "meta.yaml").write_text(f"name: {name}\ndescription: index a fasta\n")
        (d / "environment.yaml").write_text(
            "dependencies:\n" + "".join(f"  - {p}\n" for p in pkgs)
        )
        (d / "test" / "Snakefile").write_text(
            f'rule x:\n    wrapper: "master/{path}"\n'
        )
    return Catalog(root, "v0.0.0")


class FakeSnakemake:
    """Stands in for subprocess.run: writes declared outputs, or fails when told to."""

    def __init__(self):
        self.fail_next = False
        self.commands = []

    def __call__(self, cmd, capture_output=True, text=True, cwd=None):
        self.commands.append(cmd)
        if self.fail_next:
            self.fail_next = False
            return subprocess.CompletedProcess(
                cmd, 1, "", "MissingInputException: boom"
            )
        snakefile = Path(cmd[cmd.index("--snakefile") + 1]).read_text()
        last = snakefile.split("\nrule ")[-1]
        for path in re.findall(
            r"=\s*'([^']+)'", last.split("    log:")[0].split("    output:")[1]
        ):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("data\n")
        return subprocess.CompletedProcess(cmd, 0, "Finished", "")


def setup_engine(tmp: Path):
    """Engine over a fake catalog, fake snakemake, and one staged input file."""
    data = tmp / "data"
    (data / "samples").mkdir(parents=True)
    (data / "genome.fa").write_text(">chr\nACGT\n")
    (data / "samples" / "A.fastq").write_text("@r\nACGT\n+\nIIII\n")
    fake = FakeSnakemake()
    engine = SnakemakeProvider(
        tmp / "work",
        make_catalog(tmp / "wrappers"),
        input_dir=data,
        run_fn=fake,
        conda_frontend="conda",
    )
    engine.setup()
    return engine, fake


def test_catalog_index_search_and_prefix():
    with tempfile.TemporaryDirectory() as t:
        c = make_catalog(Path(t))
        assert set(c.index) == {"bio/samtools/faidx", "meta/bio/bwa_mapping"}
        assert c.index["meta/bio/bwa_mapping"]["type"] == "meta_wrapper"
        assert [r["path"] for r in c.search("SAMTOOLS")] == ["bio/samtools/faidx"]
        assert c.get("master/bio/samtools/faidx") and c.get("v0.0.0/bio/samtools/faidx")
        assert c.prefix == f"file://{Path(t).resolve()}/"


def test_setup_stages_inputs_as_symlinks():
    with tempfile.TemporaryDirectory() as t:
        engine, _ = setup_engine(Path(t))
        listing = {i["path"] for i in engine.list_input_dir()["items"]}
        assert {"genome.fa", "samples", "samples/A.fastq"} <= listing
        assert (engine.workdir / "input" / "genome.fa").is_symlink()


def test_execute_wrapper_renders_pinned_local_wrapper():
    with tempfile.TemporaryDirectory() as t:
        engine, fake = setup_engine(Path(t))
        out = engine.execute_wrapper(
            "faidx",
            "bio/samtools/faidx",
            {"fa": "genome.fa"},
            {"fai": "genome.fa.fai"},
            {},
            2,
        )
        assert out["success"], out
        text = engine.snakefile_path.read_text()
        assert "    wrapper: 'bio/samtools/faidx'" in text
        assert f"    log: '{engine.workdir}/logs/faidx.log'" in text
        assert "    threads: 2" in text
        assert text.startswith("rule all:")
        cmd = fake.commands[-1]
        assert cmd[cmd.index("--wrapper-prefix") + 1] == engine.catalog.prefix
        assert (
            "--use-conda" in cmd and cmd[cmd.index("--conda-frontend") + 1] == "conda"
        )
        assert engine.history == [
            {"step": 1, "name": "faidx", "step_dir": "steps/01_faidx"}
        ]


def test_failed_rule_is_removed_and_name_reusable():
    with tempfile.TemporaryDirectory() as t:
        engine, fake = setup_engine(Path(t))
        fake.fail_next = True
        out = engine.execute_wrapper(
            "faidx", "bio/samtools/faidx", {"fa": "genome.fa"}, {"fai": "x.fai"}, {}, 1
        )
        assert not out["success"] and "removed automatically" in out["note"]
        assert engine.history == [] and not (engine.steps_dir / "01_faidx").exists()
        assert engine.execute_wrapper(
            "faidx", "bio/samtools/faidx", {"fa": "genome.fa"}, {"fai": "x.fai"}, {}, 1
        )["success"]
        assert [a["success"] for a in engine.attempts] == [False, True]


def test_delete_rule_needs_a_name_and_keeps_others():
    with tempfile.TemporaryDirectory() as t:
        engine, _ = setup_engine(Path(t))
        engine.execute_wrapper(
            "one", "bio/samtools/faidx", {"fa": "genome.fa"}, {"o": "1.fai"}, {}, 1
        )
        engine.execute_wrapper(
            "two", "bio/samtools/faidx", {"fa": "genome.fa"}, {"o": "2.fai"}, {}, 1
        )
        assert not engine.delete_rule("")["success"]
        assert not engine.delete_rule("nope")["success"]
        assert engine.delete_rule("one")["success"]
        assert [h["name"] for h in engine.history] == ["two"]
        assert (
            not (engine.steps_dir / "01_one").exists()
            and (engine.steps_dir / "02_two").exists()
        )


def test_sandbox_and_name_validation():
    with tempfile.TemporaryDirectory() as t:
        engine, _ = setup_engine(Path(t))
        w = "bio/samtools/faidx"
        assert (
            "Write access denied"
            in engine.execute_wrapper(
                "a", w, {"f": "genome.fa"}, {"o": "../../x"}, {}, 1
            )["error"]
        )
        assert (
            "Read access denied"
            in engine.execute_wrapper(
                "b", w, {"f": "../../../etc/passwd"}, {"o": "x"}, {}, 1
            )["error"]
        )
        assert (
            "identifiers"
            in engine.execute_wrapper("c", w, {"1bad": "genome.fa"}, {"o": "x"}, {}, 1)[
                "error"
            ]
        )
        assert (
            "identifier"
            in engine.execute_wrapper("bad name", w, {}, {"o": "x"}, {}, 1)["error"]
        )
        assert (
            "not found"
            in engine.execute_wrapper("d", "bio/nope", {}, {"o": "x"}, {}, 1)["error"]
        )
        assert engine.history == [] and engine.attempts == []


def test_shell_quoting_round_trips():
    with tempfile.TemporaryDirectory() as t:
        engine, _ = setup_engine(Path(t))
        shell = "awk -F'\\t' '{print \"x\" $1}' {input.reads} > {output.out}"
        assert engine.execute_rule(
            "glue",
            {"reads": "samples/A.fastq"},
            {"out": "a.txt"},
            shell,
            {},
            1,
            ["gawk"],
        )["success"]
        line = next(
            l
            for l in engine.snakefile_path.read_text().splitlines()
            if l.startswith("    shell:")
        )
        assert ast.literal_eval(line.split("shell:", 1)[1].strip()) == shell
        assert (engine.steps_dir / "01_glue" / "env.yaml").exists()
        assert "conda: 'steps/01_glue/env.yaml'" in engine.snakefile_path.read_text()


def test_log_path_carries_output_wildcards():
    """Snakemake rejects a rule whose log lacks the wildcards its outputs use."""
    with tempfile.TemporaryDirectory() as t:
        engine, _ = setup_engine(Path(t))
        engine.create_sample_sheet([{"sample": "A", "r1": "samples/A.fastq"}])
        assert engine.execute_rule(
            "per_sample",
            {"r": "samples/{sample}.fastq"},
            {"o": "{sample}.txt"},
            "cp {input} {output}",
            {},
            1,
            [],
        )["success"]
        text = engine.snakefile_path.read_text()
        assert f"    log: '{engine.workdir}/logs/per_sample.{{sample}}.log'" in text
        assert engine.execute_rule(
            "joint", {}, {"o": "all.txt"}, "touch {output}", {}, 1, []
        )["success"]
        text = engine.snakefile_path.read_text()
        assert f"    log: '{engine.workdir}/logs/joint.log'" in text
        # rule all only expands targets that use {sample}
        assert "expand(" not in text.split("\nrule per_sample")[0]
        assert f"        '{engine.steps_dir}/02_joint/all.txt'," in text


def test_conda_frontend_only_passed_when_explicit():
    with tempfile.TemporaryDirectory() as t:
        engine, fake = setup_engine(Path(t))
        engine._explicit_frontend = None
        engine.execute_rule("a", {}, {"o": "x"}, "touch {output}", {}, 1, [])
        assert "--conda-frontend" not in fake.commands[-1]
        assert engine.conda_frontend in ("conda", "mamba", None)


def test_sample_sheet_switches_target_to_expand():
    with tempfile.TemporaryDirectory() as t:
        engine, _ = setup_engine(Path(t))
        assert not engine.create_sample_sheet([{"name": "A"}])["success"]
        assert engine.create_sample_sheet([{"sample": "A", "r1": "samples/A.fastq"}])[
            "success"
        ]
        assert (
            (engine.workdir / "input" / "samples.csv")
            .read_text()
            .startswith("sample,r1")
        )
        engine.execute_rule(
            "cp",
            {"r": "samples/{sample}.fastq"},
            {"o": "{sample}.txt"},
            "cp {input} {output}",
            {},
            1,
            [],
        )
        text = engine.snakefile_path.read_text()
        assert (
            text.startswith("import pandas as pd")
            and "expand('" in text
            and "sample=SAMPLES)" in text
        )


def test_tool_schemas_use_types_behalf_adk_maps():
    with tempfile.TemporaryDirectory() as t:
        engine, _ = setup_engine(Path(t))
        task = WorkflowTask(engine, Recorder("r", "g"))
        tools = task.build_tools({"goal": "g"})
        for spec in tools:
            assert all(v in ADK_TYPES for v in spec.input_schema.values()), spec.name
        kinds = {s.name: s.kind for s in tools}
        assert {n for n, k in kinds.items() if k == "action"} == {
            "create_sample_sheet",
            "execute_wrapper",
            "execute_rule",
            "delete_rule",
        }


def test_adk_declarations_carry_every_argument():
    """behalf's Gemini runner synthesizes a signature; ADK must turn it into a schema."""
    pytest = __import__("pytest")
    pytest.importorskip("google.adk")
    from behalf.runner.adk import _to_adk_tool

    with tempfile.TemporaryDirectory() as t:
        engine, _ = setup_engine(Path(t))
        task = WorkflowTask(engine, Recorder("r", "g"))
        for spec in task.build_tools({"goal": "g"}):
            decl = _to_adk_tool(spec, lambda n, a: True)._get_declaration()
            schema = decl.parameters_json_schema or (
                decl.parameters.model_dump() if decl.parameters else {}
            )
            props = schema.get("properties") or {}
            assert set(props) == set(spec.input_schema), spec.name
            assert set(schema.get("required") or []) == set(
                spec.input_schema
            ), spec.name


def test_budget_and_finish_validation():
    with tempfile.TemporaryDirectory() as t:
        engine, _ = setup_engine(Path(t))
        task = WorkflowTask(engine, Recorder("r", "g"), max_calls=1)
        by = {s.name: s for s in task.build_tools({"goal": "g"})}
        read = lambda r: json.loads(r["content"][0]["text"])
        assert read(asyncio.run(by["get_environment"].handler({})))["success"]
        assert "budget" in read(asyncio.run(by["list_input_dir"].handler({})))["error"]
        assert not read(asyncio.run(by["finish"].handler({"status": "done"})))[
            "success"
        ]
        assert read(
            asyncio.run(
                by["finish"].handler(
                    {"status": "complete", "summary": "s", "issues": [], "reason": "r"}
                )
            )
        )["success"]
        assert (
            "already called"
            in read(asyncio.run(by["finish"].handler({"status": "complete"})))["error"]
        )
        assert task.finish["status"] == "complete" and len(task.recorder.calls) == 5


def test_scripted_run_through_behalf_writes_record():
    with tempfile.TemporaryDirectory() as t:
        engine, _ = setup_engine(Path(t))
        recorder = Recorder("run1", "index the genome", plan="unit")
        task = WorkflowTask(engine, recorder)
        runner = ScriptedRunner(
            [
                {"tool": "get_environment"},
                {"tool": "list_input_dir"},
                {
                    "tool": "get_wrapper_details",
                    "args": {"wrapper_path": "bio/samtools/faidx"},
                },
                {
                    "tool": "execute_wrapper",
                    "args": {
                        "rule_name": "faidx",
                        "wrapper_path": "bio/samtools/faidx",
                        "inputs": {"fa": "genome.fa"},
                        "output": {"fai": "genome.fa.fai"},
                        "params": {},
                        "threads": 1,
                    },
                },
                {"tool": "list_work_dir"},
                {
                    "tool": "finish",
                    "args": {
                        "status": "complete",
                        "summary": "indexed",
                        "issues": [],
                        "reason": "done",
                    },
                },
            ]
        )
        outcome = asyncio.run(
            run_task(
                task,
                runner,
                manifest={"goal": "index the genome", "context": ""},
                confirm_fn=lambda n, a: True,
            )
        )
        assert outcome.result["status"] == "complete"
        run_dir = recorder.write(
            Path(t) / "runs", engine, runner, outcome.result, "script"
        )
        record = json.loads((run_dir / "run.json").read_text())
        assert record["status"] == "complete" and len(record["calls"]) == 6
        assert (
            record["steps"][0]["name"] == "faidx" and record["attempts"][0]["success"]
        )
        assert (
            record["outputs"][0]["path"] == "steps/01_faidx/genome.fa.fai"
            and len(record["outputs"][0]["sha256"]) == 64
        )
        assert record["versions"]["wrappers"]["version"] == "v0.0.0"
        assert (run_dir / "Snakefile").read_text() == engine.snakefile_path.read_text()


def test_confirm_gate_blocks_actions():
    with tempfile.TemporaryDirectory() as t:
        engine, _ = setup_engine(Path(t))
        task = WorkflowTask(engine, Recorder("r", "g"), confirm_actions=True)
        runner = ScriptedRunner(
            [
                {"tool": "delete_rule", "args": {"rule_name": "x"}},
                {"tool": "get_environment"},
            ]
        )
        asyncio.run(
            run_task(
                task,
                runner,
                manifest={"goal": "g", "context": ""},
                confirm_fn=lambda n, a: False,
            )
        )
        assert [c["tool"] for c in task.recorder.calls] == ["get_environment"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"OK {name}")
    print("\nall unit tests passed")
