"""The snakemake-wrappers catalog, pinned to one release."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

import yaml

WRAPPERS_URL = "https://github.com/snakemake/snakemake-wrappers"
DEFAULT_VERSION = "v9.19.0"


def _read_yaml(path: Path) -> dict:
    """Parse a YAML file, returning {} when it is empty or unreadable."""
    try:
        return yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}


def _read_text(path: Path) -> str:
    """Read a text file, returning "" when it is missing or unreadable."""
    try:
        return path.read_text()
    except Exception:
        return ""


class Catalog:
    """Index of every wrapper and meta-wrapper in a local wrappers checkout."""

    def __init__(self, path: Path, version: str):
        self.path = Path(path).resolve()
        self.version = version
        self.index: dict[str, dict[str, Any]] = {}
        self._build_index()

    @classmethod
    def ensure(
        cls, dest: Path, version: str = DEFAULT_VERSION, url: str = WRAPPERS_URL
    ) -> "Catalog":
        """Clone the wrappers at a release tag into dest if absent, then index them."""
        dest = Path(dest)
        if not dest.exists():
            git = shutil.which("git")
            if not git:
                raise RuntimeError("git is required to clone the wrappers catalog.")
            subprocess.run(
                [git, "clone", "--depth", "1", "--branch", version, url, str(dest)],
                check=True,
                capture_output=True,
                text=True,
            )
        return cls(dest, version)

    @property
    def commit(self) -> Optional[str]:
        """The checked out commit of the wrappers repository, if it is a git checkout."""
        try:
            out = subprocess.run(
                ["git", "-C", str(self.path), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
            return out.stdout.strip()
        except Exception:
            return None

    @property
    def prefix(self) -> str:
        """The --wrapper-prefix that makes snakemake execute this checkout."""
        return f"file://{self.path}/"

    def _build_index(self) -> None:
        """Index name, description, authors, conda packages, README and example per wrapper."""
        for meta_path in self.path.rglob("meta.yaml"):
            wrapper_dir = meta_path.parent
            wrapper_path = str(wrapper_dir.relative_to(self.path))
            meta = _read_yaml(meta_path)
            env = _read_yaml(wrapper_dir / "environment.yaml")
            readme = _read_text(wrapper_dir / "README.md") or _read_text(
                wrapper_dir / "readme.md"
            )
            self.index[wrapper_path] = {
                "path": wrapper_path,
                "category": wrapper_path.split("/")[0],
                "type": (
                    "meta_wrapper" if wrapper_path.startswith("meta/") else "wrapper"
                ),
                "name": meta.get("name", wrapper_path),
                "description": meta.get("description", ""),
                "authors": meta.get("authors", []),
                "conda_packages": env.get("dependencies", []),
                "readme": readme,
                "example_rule": _read_text(wrapper_dir / "test" / "Snakefile"),
            }

    def get(self, wrapper_path: str) -> Optional[dict[str, Any]]:
        """Return one wrapper entry, accepting an accidental version or 'master/' prefix."""
        for prefix in (f"{self.version}/", "master/"):
            if wrapper_path.startswith(prefix):
                wrapper_path = wrapper_path[len(prefix) :]
        return self.index.get(wrapper_path)

    def search(self, query: str) -> list[dict[str, Any]]:
        """Case-insensitive substring search over path, name, description, category and packages."""
        query = query.lower()
        matches = []
        for entry in self.index.values():
            searchable = " ".join(
                [
                    entry["path"],
                    str(entry["name"]),
                    str(entry["description"]),
                    entry["category"],
                    entry["type"],
                    " ".join(str(p) for p in entry["conda_packages"]),
                ]
            ).lower()
            if query in searchable:
                matches.append(
                    {
                        k: entry[k]
                        for k in ("path", "name", "type", "description", "category")
                    }
                )
        return matches
