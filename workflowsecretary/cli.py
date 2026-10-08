"""workflow-secretary: design and run a workflow toward a goal with a workflow tool provider."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

import yaml

from .providers import DEFAULT_PROVIDER, PROVIDERS, get_provider


def load_plan(path: Path) -> dict:
    """Load a plan with at least a goal."""
    plan = yaml.safe_load(Path(path).read_text()) or {}
    if not plan.get("goal"):
        raise SystemExit(f"{path}: plan has no goal.")
    return plan


def make_runner(backend: str, model: str | None, script: str | None):
    """A behalf runner by name, or the scripted runner."""
    if backend == "script":
        from .scripted import ScriptedRunner, load_script

        if not script:
            raise SystemExit("--backend script needs --script FILE")
        return ScriptedRunner(load_script(Path(script)))
    from behalf import make_runner as behalf_runner

    return behalf_runner(backend=backend, model=model)


def cmd_prepare(args) -> int:
    """One-time setup for a provider, e.g. fetching its catalog."""
    return get_provider(args.provider).prepare(args)


def cmd_run(args) -> int:
    """Run one workflow agent session and write its record."""
    from behalf import run_task

    from .record import Recorder
    from .task import WorkflowTask

    if args.plan:
        plan = load_plan(Path(args.plan))
        goal, context, plan_name = (
            plan["goal"],
            plan.get("context") or "",
            plan.get("name"),
        )
    elif args.goal:
        goal, context, plan_name = args.goal, "", None
    else:
        raise SystemExit("give --plan or --goal")
    if args.context:
        context = args.context

    workdir = Path(args.workdir)
    if workdir.exists() and any(workdir.iterdir()):
        raise SystemExit(
            f"{workdir} is not empty; each run needs a fresh work directory."
        )
    if args.input and not Path(args.input).is_dir():
        raise SystemExit(f"{args.input} is not a directory.")

    provider = get_provider(args.provider).from_args(
        args, workdir, Path(args.input) if args.input else None
    )
    provider.setup()

    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    recorder = Recorder(run_id, goal, context, plan=plan_name)
    task = WorkflowTask(
        provider, recorder, max_calls=args.max_calls, confirm_actions=args.confirm
    )
    runner = make_runner(args.backend, args.model, args.script)

    outcome = asyncio.run(
        run_task(
            task,
            runner,
            manifest={"goal": goal, "context": context},
            confirm_fn=_confirm(args.confirm),
        )
    )
    run_dir = recorder.write(
        Path(args.out), provider, runner, outcome.result, args.backend
    )
    status = (outcome.result or {}).get("status", "no-finish")
    print(
        f"status={status} steps={len(provider.history)} attempts={len(provider.attempts)} record={run_dir}"
    )
    return 0 if status == "complete" else 2


def _confirm(enabled: bool):
    """behalf's interactive confirmation when --confirm is set, else approve all."""
    if enabled:
        from behalf import default_confirm

        return default_confirm
    return lambda name, args: True


def _add_provider_option(parser) -> None:
    parser.add_argument(
        "--provider",
        default=os.environ.get("WORKFLOW_SECRETARY_PROVIDER", DEFAULT_PROVIDER),
        choices=sorted(PROVIDERS),
        help="workflow tool that runs the steps",
    )


def get_parser() -> argparse.ArgumentParser:
    """Argument parser for all subcommands. Each provider adds its own options."""
    p = argparse.ArgumentParser(prog="workflow-secretary", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="one-time setup for a provider")
    _add_provider_option(prep)
    for name in sorted(PROVIDERS):
        get_provider(name).add_prepare_arguments(
            prep.add_argument_group(f"{name} options")
        )

    r = sub.add_parser("run", help="design and run a workflow toward a goal")
    r.add_argument("--plan", help="plan YAML with a goal (and optional context)")
    r.add_argument("--goal", help="goal text, instead of --plan")
    r.add_argument("--context", help="additional context, overrides the plan's")
    r.add_argument(
        "--input", help="input data directory, symlinked into WORK_DIR/input"
    )
    r.add_argument("--workdir", required=True, help="fresh work directory for this run")
    r.add_argument("--out", default="runs", help="directory for run records")
    r.add_argument("--run-id", help="record name (default: timestamp)")
    _add_provider_option(r)
    r.add_argument(
        "--backend",
        default=os.environ.get("WORKFLOW_SECRETARY_BACKEND", "gemini"),
        choices=["gemini", "claude", "aws", "script"],
    )
    r.add_argument("--model", default=os.environ.get("WORKFLOW_SECRETARY_MODEL"))
    r.add_argument("--script", help="tool call script for --backend script")
    r.add_argument("--max-calls", type=int, default=100, help="tool call budget")
    r.add_argument(
        "--confirm", action="store_true", help="approve each action tool call"
    )
    for name in sorted(PROVIDERS):
        get_provider(name).add_arguments(r.add_argument_group(f"{name} options"))
    return p


def main(argv=None) -> int:
    """Entry point."""
    args = get_parser().parse_args(argv)
    if args.command == "prepare":
        return cmd_prepare(args)
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
