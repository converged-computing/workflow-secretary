"""The workflow task, as a behalf Task, over any workflow provider."""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from behalf import AgentRunner, ConfirmFn, Task, ToolSpec

from .providers.base import Tool, WorkflowProvider
from .record import Recorder

SYSTEM = """### PERSONA
You are an autonomous Workflow Agent. You are an expert computational scientist who
specializes in designing and executing workflows with {provider}. Your goal is to take
raw input data and a scientific objective, and execute a complete workflow — one step
at a time — until the goal is achieved.

### ENVIRONMENT
Call get_environment at the very start of every session. It will tell you exactly what
directories exist and which workflow tool is in use. Do not assume anything about the
environment before calling it.

Your working environment has the following structure, all rooted at a single WORK_DIR
that is controlled for you. You never need to know the absolute path of WORK_DIR:

  WORK_DIR/
  ├── input/      Read-only staged input data. NEVER write here.
  ├── steps/      Your writable workspace. All outputs go here.
  │   ├── 01_stepname/   Step 1 outputs
  │   ├── 02_stepname/   Step 2 outputs
  │   └── ...
  └── logs/       Per-step logs. Written automatically, do not manage manually.

### PATH CONVENTIONS — YOU MUST FOLLOW THESE EXACTLY
These rules apply to every input and output argument you pass to a tool that executes
a step. Violating them will cause execution to fail.

1. INPUT FILES FROM STAGED DATA
   Specify paths relative to WORK_DIR/input/. Do not include 'input/' as a prefix —
   the paths are resolved for you.
   Correct:   {{"reads": "samples/A.fastq", "ref": "genome.fa"}}
   Incorrect: {{"reads": "/data/samples/A.fastq"}}
   Incorrect: {{"reads": "input/samples/A.fastq"}}

2. OUTPUT FILES
   Specify paths relative to the current step's directory. Do not include 'steps/'
   or the step directory name — they are resolved for you.
   Correct:   {{"bam": "A.bam"}}
   Incorrect: {{"bam": "steps/01_bwa_mem_A/A.bam"}}

3. CHAINING STEPS — USING A PRIOR STEP'S OUTPUT AS INPUT
   Reference prior step outputs by prefixing with 'steps/NN_stepname/'.
   Correct:   {{"bam": "steps/01_bwa_mem_A/A.bam"}}
   Incorrect: {{"bam": "A.bam"}}  (ambiguous — cannot be found)

4. NEVER USE
   - Absolute paths of any kind
   - Environment variable names
   - Paths that start with '/' or '~'
   - The literal string 'WORK_DIR' or 'INPUT_DIR'

Input and output names (the keys) must be valid Python identifiers, e.g. "reads", "r1".

### PERMISSIONS
- WORK_DIR/input/  : READ ONLY. You may reference files here as inputs but must
                     never write, modify, or delete anything under input/.
- WORK_DIR/steps/  : READ AND WRITE. All your outputs must go here.
- WORK_DIR/logs/   : Managed automatically. Every step gets a log; do not set one.

### STRATEGY
Follow these phases in order. Do not skip phases.

**Phase 1 — Environment and Discovery**
1. Call get_environment to confirm the runtime environment and path conventions.
2. Call list_input_dir to see all staged input files. Note their paths carefully —
   these are the exact strings you will use as input values (convention 1 above).
3. Identify file types, sample names, and any pre-built index files that may allow
   you to skip steps (e.g. existing .bwt/.sa files mean bwa index is not needed).
4. Based on file types and the user's goal, reason about what processing steps
   are required and in what order.

**Phase 2 — Planning**
1. Decompose the goal into an ordered sequence of steps.
2. For each step, use the search and documentation tools described below to find
   how to run it, and read the documentation before executing anything.
3. State your complete plan before beginning execution.

**Phase 3 — Execution (strictly one step at a time)**
1. Execute each step in order with the execute tools described below.
2. After EVERY step, regardless of success or failure:
   a. Check the 'success' field in the result.
   b. If success=True: call list_work_dir and verify the expected output files
      appear with nonzero size before proceeding to the next step.
   c. If success=False: read 'stderr' carefully to diagnose the problem. A failed
      step is removed from the workflow automatically, so do not delete anything.
      - If the fix is clear (wrong param, wrong key name, path issue): call the
        same tool again with corrected arguments. The same step name may be reused.
      - If you have retried this step twice without success, document the issue
        in your issues list and move to the next step.
3. When using a prior step's output as input, use convention 3 (steps/NN_stepname/filename).
4. Never execute more than one step at a time.

**Phase 4 — Completion**
When all planned steps are complete (or best-effort complete), call finish exactly once.

{instructions}

### CONSTRAINTS
- You MUST call get_environment before anything else.
- You MUST call list_input_dir before planning.
- You MUST check list_work_dir after every successful step.
- You MUST follow path conventions exactly — no absolute paths, no env var names.
- You MUST NOT write to WORK_DIR/input/ under any circumstances.
- Step names must be unique, valid Python identifiers (snake_case, no spaces).
- You cannot ask the user questions. Use best judgment from available data and goal.
- You MUST end by calling finish. Do not end with a message instead."""

USER = """Design and execute a workflow to accomplish the following:
{goal}

The input data has already been staged and your working directory configured.
Call get_environment first to confirm the layout, then call list_input_dir to see
your input files before planning anything."""

STATUSES = ("complete", "incomplete")


def _text(obj: Any) -> dict:
    """A behalf tool result carrying obj as JSON text."""
    return {
        "content": [{"type": "text", "text": json.dumps(obj, indent=2, default=str)}]
    }


class WorkflowTask(Task):
    """Design and run a workflow toward a goal, one step at a time, with one provider."""

    name = "workflow"

    def __init__(
        self,
        provider: WorkflowProvider,
        recorder: Recorder,
        max_calls: int = 100,
        confirm_actions: bool = False,
    ):
        self.provider = provider
        self.recorder = recorder
        self.max_calls = max_calls
        self.confirm_actions = confirm_actions
        self.finish: Optional[dict] = None
        self.calls = 0

    def manifest_schema(self) -> dict:
        """The frozen inputs of a run."""
        return {"goal": str, "context": str}

    def execute_system_prompt(self, manifest: dict) -> str:
        """Standing instructions: the shared strategy with the provider's section."""
        return SYSTEM.format(
            provider=self.provider.name, instructions=self.provider.instructions()
        )

    def user_prompt(self, manifest: dict) -> str:
        """The goal, plus any context from the plan."""
        text = USER.format(goal=manifest["goal"].strip())
        context = (manifest.get("context") or "").strip()
        if context:
            text += f"\n\nADDITIONAL CONTEXT FROM USER:\n{context}"
        return text

    def _spec(self, tool: Tool) -> ToolSpec:
        """A behalf ToolSpec whose handler counts, budgets, times and records the call."""

        async def handler(args: dict) -> dict:
            self.calls += 1
            start = time.time()
            if self.finish is not None:
                result = {
                    "success": False,
                    "error": "finish was already called; the run is over.",
                }
            elif self.calls > self.max_calls and tool.name != "finish":
                result = {
                    "success": False,
                    "error": f"Tool call budget of {self.max_calls} exhausted. Call finish now.",
                }
            else:
                try:
                    result = tool.fn(args)
                except Exception as e:
                    result = {"success": False, "error": f"{type(e).__name__}: {e}"}
            self.recorder.call(tool.name, args, result, time.time() - start)
            return _text(result)

        return ToolSpec(
            tool.name,
            tool.description,
            tool.schema,
            handler,
            kind=tool.kind,
            confirm=tool.kind == "action" and self.confirm_actions,
        )

    def _finish(self, args: dict) -> dict:
        """Record the agent's final report."""
        status = args.get("status", "")
        if status not in STATUSES:
            return {
                "success": False,
                "error": f"status must be one of {list(STATUSES)}.",
            }
        self.finish = {
            "status": status,
            "summary": args.get("summary", ""),
            "issues": list(args.get("issues") or []),
            "reason": args.get("reason", ""),
        }
        return {"success": True, "message": "Recorded. The run is over."}

    def common_tools(self) -> list[Tool]:
        """Tools every provider shares: the environment, the inputs, the outputs."""
        p = self.provider
        return [
            Tool(
                "get_environment",
                "Work directory layout, path conventions, the workflow tool in use and "
                "the steps executed so far. Call this first.",
                {},
                lambda a: p.get_environment(),
            ),
            Tool(
                "list_input_dir",
                "Recursive listing of the staged input data in WORK_DIR/input/. The listed "
                "paths are the exact input values to use.",
                {},
                lambda a: p.list_input_dir(),
            ),
            Tool(
                "list_work_dir",
                "Recursive listing of WORK_DIR/steps/ with sizes, and the step history. "
                "Call after each successful step to verify outputs exist.",
                {},
                lambda a: p.list_work_dir(),
            ),
        ]

    def build_tools(self, manifest: dict) -> list[ToolSpec]:
        """Common tools, the provider's tools, and finish."""
        tools = self.common_tools() + self.provider.tools()
        tools.append(
            Tool(
                "finish",
                "End the run with a final report. status is 'complete' if every step of the "
                "goal produced its outputs, otherwise 'incomplete'. issues lists steps that "
                "failed or were skipped, with reasons ([] if none).",
                {"status": str, "summary": str, "issues": list, "reason": str},
                self._finish,
            )
        )
        return [self._spec(t) for t in tools]

    async def execute(
        self, runner: AgentRunner, manifest: dict, confirm_fn: ConfirmFn
    ) -> Optional[dict]:
        """Run the agent over the fixed toolset; return the finish report, if any."""
        await runner.run_agent(
            self.execute_system_prompt(manifest),
            self.user_prompt(manifest),
            self.build_tools(manifest),
            confirm_fn,
        )
        return self.finish
