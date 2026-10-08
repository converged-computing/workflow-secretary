"""Integration tests: real snakemake, conda, pinned wrappers and the Snakemake tutorial data.

Environment:
  WORKFLOW_SECRETARY_TEST_DATA  the tutorial v5.4.5 'data' directory: genome.fa, its indexes, samples/{A,B,C}.fastq
  WORKFLOW_SECRETARY_WRAPPERS   wrappers checkout (default: cloned into the user cache)
  WORKFLOW_SECRETARY_CONDA_PREFIX  shared conda env cache (optional, speeds up reruns)
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workflowsecretary.cli import main
from workflowsecretary.providers.snakemake import (
    DEFAULT_VERSION,
    Catalog,
    SnakemakeProvider,
    default_wrappers_dir,
)

HERE = Path(__file__).parent
DATA = Path(os.environ.get("WORKFLOW_SECRETARY_TEST_DATA", "data"))
WRAPPERS = Path(
    os.environ.get("WORKFLOW_SECRETARY_WRAPPERS")
    or default_wrappers_dir(DEFAULT_VERSION)
)
CONDA_PREFIX = os.environ.get("WORKFLOW_SECRETARY_CONDA_PREFIX")
IDX = [f"genome.fa.{e}" for e in ("amb", "ann", "bwt", "pac", "sa")]


def require():
    """Fail early and clearly if the environment is missing a dependency."""
    missing = [b for b in ("snakemake", "git") if not shutil.which(b)]
    if not (shutil.which("mamba") or shutil.which("conda")):
        missing.append("conda or mamba")
    if not (DATA / "genome.fa").exists():
        missing.append(f"tutorial data at {DATA}")
    if missing:
        raise SystemExit(f"integration tests need: {', '.join(missing)}")


def engine(tmp: Path) -> SnakemakeProvider:
    """An engine over the tutorial data in a fresh work directory."""
    e = SnakemakeProvider(
        tmp / "work",
        Catalog.ensure(WRAPPERS, DEFAULT_VERSION),
        input_dir=DATA,
        conda_prefix=Path(CONDA_PREFIX) if CONDA_PREFIX else None,
    )
    e.setup()
    return e


def ok(result: dict) -> dict:
    """Assert an execution succeeded, showing stderr if not."""
    assert result["success"], result.get("error") or result.get("stderr")
    return result


def test_variant_calling_chain_sample_a():
    """bwa mem -> samtools sort -> samtools index -> bcftools mpileup -> bcftools call."""
    with tempfile.TemporaryDirectory() as t:
        e = engine(Path(t))
        rg = r"-R '@RG\tID:A\tSM:A'"
        ok(
            e.execute_wrapper(
                "bwa_mem_A",
                "bio/bwa/mem",
                {"reads": ["samples/A.fastq"], "idx": IDX},
                {"bam": "A.bam"},
                {"extra": rg, "sorting": "none"},
                1,
            )
        )
        ok(
            e.execute_wrapper(
                "samtools_sort_A",
                "bio/samtools/sort",
                {"bam": "steps/01_bwa_mem_A/A.bam"},
                {"bam": "A.sorted.bam"},
                {"extra": ""},
                1,
            )
        )
        ok(
            e.execute_wrapper(
                "samtools_index_A",
                "bio/samtools/index",
                {"bam": "steps/02_samtools_sort_A/A.sorted.bam"},
                {"bai": "A.sorted.bam.bai"},
                {},
                1,
            )
        )
        ok(
            e.execute_wrapper(
                "bcftools_mpileup",
                "bio/bcftools/mpileup",
                {
                    "alignments": ["steps/02_samtools_sort_A/A.sorted.bam"],
                    "ref": "genome.fa",
                    "index": "genome.fa.fai",
                },
                {"pileup": "all.pileup.bcf"},
                {"uncompressed_bcf": False, "extra": ""},
                1,
            )
        )
        ok(
            e.execute_wrapper(
                "bcftools_call",
                "bio/bcftools/call",
                {"pileup": "steps/04_bcftools_mpileup/all.pileup.bcf"},
                {"calls": "all.calls.bcf"},
                {"uncompressed_bcf": False, "caller": "-m", "extra": ""},
                1,
            )
        )
        calls = e.steps_dir / "05_bcftools_call" / "all.calls.bcf"
        assert calls.exists() and calls.stat().st_size > 0
        assert [h["name"] for h in e.history] == [
            "bwa_mem_A",
            "samtools_sort_A",
            "samtools_index_A",
            "bcftools_mpileup",
            "bcftools_call",
        ]


def test_real_failure_is_removed_then_retry_succeeds():
    with tempfile.TemporaryDirectory() as t:
        e = engine(Path(t))
        bad = e.execute_wrapper(
            "faidx", "bio/samtools/faidx", {"fa": "missing.fa"}, {"fai": "x.fai"}, {}, 1
        )
        assert not bad["success"] and e.history == []
        ok(
            e.execute_wrapper(
                "faidx",
                "bio/samtools/faidx",
                {"fa": "genome.fa"},
                {"fai": "genome.fa.fai"},
                {},
                1,
            )
        )
        assert (e.steps_dir / "01_faidx" / "genome.fa.fai").stat().st_size > 0


def test_shell_rule_with_system_tools():
    with tempfile.TemporaryDirectory() as t:
        e = engine(Path(t))
        ok(
            e.execute_rule(
                "count_reads",
                {"reads": "samples/A.fastq"},
                {"n": "A.count"},
                "awk 'END {{print NR/4}}' {input.reads} > {output.n}",
                {},
                1,
                [],
            )
        )
        assert float((e.steps_dir / "01_count_reads" / "A.count").read_text()) > 0


def test_cli_scripted_run_writes_record():
    with tempfile.TemporaryDirectory() as t:
        out = Path(t) / "runs"
        argv = [
            "run",
            "--plan",
            str(HERE.parent / "plans" / "variant-calling.yaml"),
            "--input",
            str(DATA),
            "--workdir",
            str(Path(t) / "work"),
            "--out",
            str(out),
            "--run-id",
            "scripted",
            "--wrappers",
            str(WRAPPERS),
            "--backend",
            "script",
            "--script",
            str(HERE / "scripts" / "faidx.yaml"),
        ]
        if CONDA_PREFIX:
            argv += ["--conda-prefix", CONDA_PREFIX]
        assert main(argv) == 0
        record = json.loads((out / "scripted" / "run.json").read_text())
        assert record["status"] == "complete" and record["plan"] == "variant-calling"
        assert record["provider"] == "snakemake"
        assert (
            record["versions"]["wrappers"]["commit"] and record["versions"]["snakemake"]
        )
        assert any(o["path"].endswith("genome.fa.fai") for o in record["outputs"])


def test_cli_scripted_variant_calling_all_samples():
    """The full plan for A, B and C with a sample sheet and {sample} wildcards."""
    with tempfile.TemporaryDirectory() as t:
        out = Path(t) / "runs"
        argv = [
            "run",
            "--plan",
            str(HERE.parent / "plans" / "variant-calling.yaml"),
            "--input",
            str(DATA),
            "--workdir",
            str(Path(t) / "work"),
            "--out",
            str(out),
            "--run-id",
            "vc",
            "--wrappers",
            str(WRAPPERS),
            "--backend",
            "script",
            "--script",
            str(HERE / "scripts" / "variant-calling.yaml"),
        ]
        if CONDA_PREFIX:
            argv += ["--conda-prefix", CONDA_PREFIX]
        assert main(argv) == 0
        record = json.loads((out / "vc" / "run.json").read_text())
        assert [a["success"] for a in record["attempts"]] == [True] * 5
        assert [s["name"] for s in record["steps"]] == [
            "bwa_mem",
            "samtools_sort",
            "samtools_index",
            "bcftools_mpileup",
            "bcftools_call",
        ]
        outputs = {o["path"]: o["size_bytes"] for o in record["outputs"]}
        for sample in "ABC":
            assert outputs[f"steps/02_samtools_sort/{sample}.sorted.bam"] > 0
        assert outputs["steps/05_bcftools_call/all.calls.bcf"] > 0
        snakefile = (out / "vc" / "Snakefile").read_text()
        assert "logs/bwa_mem.{sample}.log" in snakefile
        # the joint final step is the single target; per-sample steps feed it
        assert "expand(" not in snakefile.split("\nrule bwa_mem")[0]


if __name__ == "__main__":
    require()
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"OK {name}")
    print("\nall integration tests passed")
