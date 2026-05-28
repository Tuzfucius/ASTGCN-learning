from __future__ import annotations

import subprocess
from pathlib import Path

from build_context_pack import main as build_context_pack_main

ROOT = Path(__file__).resolve().parents[2]
GENERATED_FILES = [
    "docs/generated/FILE_INDEX.md",
    "docs/generated/PUBLIC_API.md",
    "docs/generated/CONTEXT_PACK.md",
]


def stage_generated_files() -> None:
    subprocess.run(["git", "add", *GENERATED_FILES], cwd=ROOT, check=True)


def main() -> None:
    build_context_pack_main()
    stage_generated_files()


if __name__ == "__main__":
    main()
