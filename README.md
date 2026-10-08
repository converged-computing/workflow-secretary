# workflow-secretary

An agent that designs and runs a workflow toward a goal, one step at a time. It is built on [behalf](https://github.com/converged-computing/behalf), and the workflow tool that runs the steps is a **provider**: Snakemake is the first, and others plug in behind the same interface. It replaces the snakemake provider on the `add-snakemake` branch of resource-secretary (PR #5), which was driven through mcp-server and fractale.

Every run, whatever the provider, uses the same work directory:

```
WORK_DIR/
├── input/      input data, symlinked (read only for the agent)
├── steps/      one directory per executed step, NN_stepname/
└── logs/       one log per step
```

The agent reads the environment, lists the inputs, plans, and executes steps with the provider's tools. A step that fails is removed automatically. The run ends when the agent calls `finish`. The provider adds whatever its tool needs next to these directories; Snakemake keeps a `Snakefile` that is rebuilt after every step.

## Install

```bash
pip install -e ".[gemini,workflow]"   # or claude / aws
```

The `workflow` extra installs Snakemake. Running a Snakemake workflow also needs `git` and `conda` or `mamba` on PATH. [Miniforge](https://github.com/conda-forge/miniforge) provides both.

## Run

```bash
# once per provider: fetch and index what it needs. For snakemake, the wrappers
# at a release tag (cached in ~/.cache/workflow-secretary)
workflow-secretary prepare --provider snakemake --wrappers-version v9.19.0

# tutorial data for plans/variant-calling.yaml
curl -sSL -o tutorial.tar.gz https://github.com/snakemake/snakemake-tutorial-data/archive/v5.4.5.tar.gz
tar --wildcards -xf tutorial.tar.gz --strip 1 "*/data"

workflow-secretary run \
  --plan plans/variant-calling.yaml \
  --input data \
  --workdir work/run-01 \
  --provider snakemake \
  --backend gemini --model gemini-2.5-pro \
  --conda-prefix ~/conda-envs
```

The exit code is 0 when the agent finishes with `status: complete`, and 2 otherwise. Flags shared by every provider:

- `--provider`: the workflow tool (default `snakemake`, or `WORKFLOW_SECRETARY_PROVIDER`).
- `--goal`: give the goal text instead of a plan.
- `--max-calls`: the tool call budget (default 100).
- `--confirm`: approve each action tool call interactively.
- `--backend` and `--model`: the behalf runner (`gemini`, `claude`, `aws`) or `script`.

Each provider adds its own group to `workflow-secretary run --help`. Snakemake's:

- `--wrappers` and `--wrappers-version`: the catalog checkout (default: the user cache at the pinned version).
- `--cores`: cores given to snakemake.
- `--conda-prefix`: share conda environments across runs, so repeated runs don't rebuild them.
- `--conda-frontend`: only passed to snakemake when given. Snakemake 9 ignores it with a warning, which would otherwise appear in every result the agent reads.

`--backend script --script FILE` replays a fixed list of tool calls instead of calling a model. The tests use it, and it is useful for checking a workflow by hand. `tests/scripts/variant-calling.yaml` replays the whole variant-calling plan for samples A, B and C with a sample sheet:

```bash
workflow-secretary run \
  --plan plans/variant-calling.yaml \
  --input data \
  --workdir work/scripted-01 \
  --backend script --script tests/scripts/variant-calling.yaml \
  --conda-prefix ~/conda-envs
```

The exit code follows the script's `finish` call, so check `attempts` in the record to see whether every step actually ran.

## Tools

Every provider gets these from the task:

| Tool | Kind | Purpose |
|---|---|---|
| `get_environment` | read | layout, path conventions, provider details, history |
| `list_input_dir` | read | staged inputs |
| `list_work_dir` | read | step outputs and history |
| `finish` | read | final report: `complete` or `incomplete`, with summary, issues, and reason |

The Snakemake provider adds:

| Tool | Kind | Purpose |
|---|---|---|
| `search_wrappers` | read | keyword search of the pinned [snakemake-wrappers](https://github.com/snakemake/snakemake-wrappers) catalog |
| `get_wrapper_details` | read | README, packages, example rule |
| `view_snakefile` | read | current Snakefile |
| `create_sample_sheet` | action | `input/samples.csv`, plus `SAMPLES` and `samples_df` in the Snakefile |
| `execute_wrapper` | action | append and run a wrapper rule |
| `execute_rule` | action | append and run a custom shell rule, optionally with `conda_packages` |
| `delete_rule` | action | remove a successful rule by name |

Tool arguments follow behalf's flat schema, so every argument is required. An empty value means "not set": `{}` for params, `[]` for packages.

## Providers

A provider is a subclass of `WorkflowProvider` in `workflowsecretary/providers/base.py`, registered by name in `workflowsecretary/providers/__init__.py`. The base class owns everything the agent's view of a run has in common, so a provider only renders and runs its own steps:

- the layout: `setup()` stages inputs, `input_root`, `steps_dir`, `logs_dir`, `next_step_dir(name)`, `remove_step_dir(dir)`
- path handling: `resolve_inputs`, `resolve_outputs`, `check_names`, and `check_paths`, which refuses reads outside `WORK_DIR` and writes outside `steps/`
- execution: `run(cmd, **attempt)` runs a command in the work directory, appends to `attempts`, and returns the response every execute tool gives the agent (stdout and stderr tails, a listing of `steps/`)
- the common tools above

A provider implements:

| Member | Purpose |
|---|---|
| `name` | registry key, and the tool named in the prompt |
| `history` | ordered steps in the workflow: `[{step, name, step_dir}, ...]` |
| `tools()` | its `Tool`s: name, description, flat schema, a function of the arguments, and `kind` (`read` or `action`) |
| `instructions()` | its section of the system prompt: how its tools fit the shared strategy, naming rules, what the agent must know |
| `environment()` | extra fields for `get_environment`, e.g. a catalog version |
| `versions()` | versions of everything that can change a run, for the record |
| `artifacts()` | files to copy into the run record, e.g. `{"Snakefile": path}` |
| `add_arguments(parser)` and `from_args(args, workdir, input_dir)` | its `run` options and construction |
| `add_prepare_arguments(parser)` and `prepare(args)` | its `prepare` options and one-time setup |

The system prompt is shared: persona, layout, path conventions, permissions, the one-step-at-a-time strategy, and the constraints are the same for every provider, and the provider's `instructions()` fill one section. `tests/test_providers.py` has a complete provider in about sixty lines, one that runs each step as a shell command, and runs the task, the scripted runner and the recorder over it. That is the template for a new one.

## Run records

Each run writes `runs/<run-id>/` containing:

- `run.json`, with these fields:
  - `goal`, `plan` and `provider`
  - `versions`: workflow-secretary, behalf, python, runner and model, plus the provider's (for snakemake: snakemake, the wrappers tag and commit, and the conda frontend)
  - `status` and `finish`: the agent's report
  - `steps`: the steps in the final workflow
  - `attempts`: every execution, with return code and time
  - `outputs`: every file under `steps/`, with size and sha256
  - `calls`: every tool call, with arguments and result
- the provider's artifacts, for snakemake the `Snakefile` the agent built.

`status` is the agent's own report. `attempts` and `outputs` come from the provider, so a reported success can be checked against what actually ran.

## Plans

| Plan | Original runs | Data |
|---|---|---|
| `variant-calling` | 20 | Snakemake tutorial v5.4.5 |
| `qc-and-trim` | 20 | not published |
| `amplicon-batch` | 21 | not published |
| `amplicon-demultiplex-shell` | 3 | not published |

Goal texts are copied from the fractale-experiments plan and run logs.

## Changes from the resource-secretary provider

- **The workflow tool is a provider.** The task, recorder, CLI and scripted runner know nothing about Snakemake; the Snakemake provider is one implementation of the interface above.
- **Pinned wrappers.** Rules use `wrapper: 'bio/...'` with `--wrapper-prefix file://<checkout>/`, so the code that runs is the same checkout the agent reads docs from. Runs also work without network access to GitHub. The provider indexed a clone of `master` but executed `master/...` fetched at run time.
- **Failed rules are removed automatically, and `delete_rule` requires a name.** The provider's prompt told the agent to call `rollback_step`, which was never exposed as a tool. Its equivalent, `delete_rule` with no name, would have removed the previous successful step.
- **`execute_wrapper` and `execute_rule` have descriptions.** In the provider their docstrings were f-strings, which Python does not treat as docstrings. mcp-server therefore sent both tools with no description.
- **The run ends with a `finish` tool** instead of a JSON `stop` message. In 4 of the original runs the agent called `stop` as a tool.
- **`execute_rule` is shell only.** `run` and `script` were never used in the 44 logged runs.
- **Logs and cores are set by the engine.** Logs are automatic and cores come from `--cores`, so neither is an agent argument. A rule whose outputs use `{sample}` gets a `logs/rule.{sample}.log`, which snakemake requires; `rule all` expands only targets that use `{sample}`, so a joint step after per-sample steps works.
- **Values are quoted with `repr`.** A shell command containing quotes or `\t` stays valid in the Snakefile.
- **Sandbox checks cover inputs as well as outputs.** Input and output names must be valid identifiers.
- **Configuration comes from arguments** instead of module-level environment variables.

## Known behalf issues

- behalf 0.1.0 called ADK's `create_session` synchronously, which is a coroutine in current ADK (2.11), so `--backend gemini` failed. Fixed in behalf 0.1.1, which this package requires.
- Tool schemas cannot mark arguments optional, which is why every argument is required here.
- The Claude runner fixes `max_turns` at 40. The `--max-calls` budget is enforced in the tools, so it applies to every backend.
- Runners do not report token usage, so records carry tool calls and timing but not tokens.

## Tests

```bash
python -m pytest tests/test_unit.py tests/test_providers.py   # no snakemake or model needed

# needs snakemake, conda/mamba, git and the tutorial data
export WORKFLOW_SECRETARY_TEST_DATA=data
export WORKFLOW_SECRETARY_CONDA_PREFIX=~/conda-envs
python tests/test_integration.py
```

The unit tests use a fake snakemake and a two-wrapper catalog. One of them checks that Google ADK builds a complete schema for every tool, and is skipped when `google-adk` is not installed. The provider tests exercise the interface with the shell provider described above.

The integration tests run the variant-calling chain for sample A (bwa mem, samtools sort and index, bcftools mpileup and call) with real wrappers, then the whole plan for A, B and C through the CLI with `tests/scripts/variant-calling.yaml`. They also cover a real failure followed by a retry, a shell rule, and the record written by a scripted run. The first run builds the conda environments, which takes a few minutes; later runs with the same `--conda-prefix` take under a minute.

## License

HPCIC DevTools is distributed under the terms of the MIT license. All new contributions must be made under this license.

See [LICENSE](https://github.com/converged-computing/cloud-select/blob/main/LICENSE), [COPYRIGHT](https://github.com/converged-computing/cloud-select/blob/main/COPYRIGHT), and [NOTICE](https://github.com/converged-computing/cloud-select/blob/main/NOTICE) for details.

SPDX-License-Identifier: (MIT)

LLNL-CODE- 842614
