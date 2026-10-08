"""Workflow tool providers. Register a new one here to make it available to the CLI."""

from __future__ import annotations

from .base import PATH_CONVENTIONS, Tool, WorkflowProvider

PROVIDERS: dict[str, str] = {
    # name -> "module:Class", imported on demand so an unused tool's
    # dependencies are never loaded.
    "snakemake": "workflowsecretary.providers.snakemake:SnakemakeProvider",
}

DEFAULT_PROVIDER = "snakemake"


def get_provider(name: str) -> type[WorkflowProvider]:
    """The provider class registered under name."""
    try:
        target = PROVIDERS[name]
    except KeyError:
        raise SystemExit(
            f"unknown provider {name!r} (choose: {', '.join(sorted(PROVIDERS))})"
        )
    module, cls = target.split(":")
    import importlib

    return getattr(importlib.import_module(module), cls)


__all__ = [
    "DEFAULT_PROVIDER",
    "PATH_CONVENTIONS",
    "PROVIDERS",
    "Tool",
    "WorkflowProvider",
    "get_provider",
]
