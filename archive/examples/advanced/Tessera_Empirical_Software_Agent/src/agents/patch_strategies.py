"""Patch strategy helpers: apply natural-language patch plans to a working dir."""
import re, os, json, pathlib, textwrap
from typing import Dict

def apply_patch_plan(repo_dir: str, plan: str, files: Dict[str, str]) -> None:
    # Minimal reference: write whole-file replacements; extend with diffs/edits as needed.
    for rel, content in (files or {}).items():
        p = pathlib.Path(repo_dir) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
