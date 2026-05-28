from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "generated" / "CONTEXT_PACK.md"

CONTEXT_FILES = [
    "README.md",
    "pyproject.toml",
    "pytest.ini",
    "configs/pems04.yaml",
    "src/README.md",
    "src/astgcn/README.md",
    "src/astgcn/baselines/README.md",
    "src/astgcn/data/README.md",
    "src/astgcn/engine/README.md",
    "src/astgcn/models/README.md",
    "scripts/README.md",
    "scripts/context/README.md",
    "docs/generated/FILE_INDEX.md",
    "docs/generated/PUBLIC_API.md",
]


def run_script(script: str) -> None:
    subprocess.run([sys.executable, str(ROOT / script)], check=True)


def fence_for(path: Path) -> str:
    if path.suffix == ".py":
        return "python"
    if path.suffix in {".yml", ".yaml"}:
        return "yaml"
    if path.suffix == ".ini":
        return "ini"
    if path.suffix == ".toml":
        return "toml"
    return "markdown"


def emit_file(lines: list[str], file: str) -> None:
    path = ROOT / file
    if not path.exists():
        return

    lang = fence_for(path)
    lines.append(f"## File: `{file}`")
    lines.append("")
    lines.append(f"```{lang}")
    lines.append(path.read_text(encoding="utf-8"))
    lines.append("```")
    lines.append("")


def main() -> None:
    run_script("scripts/context/generate_file_index.py")
    run_script("scripts/context/generate_public_api.py")

    lines = [
        "# Context Pack",
        "",
        "> This file is generated automatically for ChatGPT / Codex context.",
        "",
        "This pack is intended to be a single text blob that summarizes the repository structure and the public API surface, then embeds the most relevant project docs and config files.",
        "",
    ]

    for file in CONTEXT_FILES:
        emit_file(lines, file)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
