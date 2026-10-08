"""An AgentRunner that replays a fixed list of tool calls instead of calling a model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from behalf import AgentRunner, ConfirmFn, Task, ToolSpec


def load_script(path: Path) -> list[dict[str, Any]]:
    """Load a list of {tool, args} calls from YAML."""
    calls = yaml.safe_load(Path(path).read_text()) or []
    if not isinstance(calls, list) or any("tool" not in c for c in calls):
        raise ValueError(f"{path}: expected a list of {{tool, args}} entries.")
    return calls


class ScriptedRunner(AgentRunner):
    """Calls each scripted tool in order, applying the same action confirmation gate as behalf runners."""

    model = None

    def __init__(self, calls: list[dict[str, Any]]):
        self.calls = calls

    async def converse(self, task: Task) -> dict:
        """Scripted runs need a manifest up front."""
        raise RuntimeError("ScriptedRunner cannot elicit a manifest; pass one.")

    async def run_agent(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[ToolSpec],
        confirm_fn: ConfirmFn,
    ) -> Any:
        """Replay the script; unknown tools raise so a broken script fails loudly."""
        by_name = {t.name: t for t in tools}
        for call in self.calls:
            spec = by_name.get(call["tool"])
            if spec is None:
                raise KeyError(f"script calls unknown tool {call['tool']!r}")
            args = call.get("args") or {}
            if (
                spec.kind == "action"
                and spec.confirm
                and not confirm_fn(spec.name, args)
            ):
                continue
            await spec.handler(args)
        return None
