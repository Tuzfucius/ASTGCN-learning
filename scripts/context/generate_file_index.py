from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DIRS = ["src", "tests", "configs", "scripts"]
OUT = ROOT / "docs" / "generated" / "FILE_INDEX.md"
SKIP_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "runs",
    "outputs",
    "checkpoints",
    ".pytest_cache",
    "astgcn.egg-info",
}
SKIP_SUFFIXES = {".ipynb"}


def should_skip(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts) or path.suffix in SKIP_SUFFIXES


def main() -> None:
    lines = [
        "# File Index",
        "",
        "> This file is generated automatically. Do not edit manually.",
        "",
    ]

    for dirname in SRC_DIRS:
        base = ROOT / dirname
        if not base.exists():
            continue

        lines.append(f"## {dirname}")
        lines.append("")

        for path in sorted(base.rglob("*")):
            if path.is_file() and not should_skip(path):
                rel = path.relative_to(ROOT).as_posix()
                lines.append(f"- `{rel}`")

        lines.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
