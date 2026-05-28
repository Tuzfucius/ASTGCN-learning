from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
OUT = ROOT / "docs" / "generated" / "PUBLIC_API.md"


def format_args(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = []
    positional = list(node.args.posonlyargs) + list(node.args.args)
    defaults = [None] * (len(positional) - len(node.args.defaults)) + list(node.args.defaults)

    for arg, default in zip(positional, defaults):
        part = arg.arg
        if arg.annotation is not None:
            part += f": {ast.unparse(arg.annotation)}" if hasattr(ast, "unparse") else ""
        if default is not None:
            part += " = ..."
        args.append(part)

    if node.args.vararg is not None:
        part = f"*{node.args.vararg.arg}"
        if node.args.vararg.annotation is not None and hasattr(ast, "unparse"):
            part += f": {ast.unparse(node.args.vararg.annotation)}"
        args.append(part)

    if node.args.kwonlyargs:
        if node.args.vararg is None:
            args.append("*")
        for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
            part = arg.arg
            if arg.annotation is not None and hasattr(ast, "unparse"):
                part += f": {ast.unparse(arg.annotation)}"
            if default is not None:
                part += " = ..."
            args.append(part)

    if node.args.kwarg is not None:
        part = f"**{node.args.kwarg.arg}"
        if node.args.kwarg.annotation is not None and hasattr(ast, "unparse"):
            part += f": {ast.unparse(node.args.kwarg.annotation)}"
        args.append(part)

    return ", ".join(args)


def parse_python_file(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rel = path.relative_to(ROOT).as_posix()

    lines = [f"## `{rel}`", ""]

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            doc = ast.get_docstring(node)
            lines.append(f"### class `{node.name}`")
            if doc:
                lines.append("")
                lines.append(doc.splitlines()[0])
            lines.append("")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            signature = f"{node.name}({format_args(node)})"
            lines.append(f"### function `{signature}`")
            doc = ast.get_docstring(node)
            if doc:
                lines.append("")
                lines.append(doc.splitlines()[0])
            lines.append("")

    return lines


def main() -> None:
    lines = [
        "# Public API",
        "",
        "> This file is generated automatically. Do not edit manually.",
        "",
    ]

    for path in sorted(SRC.rglob("*.py")):
        if should_skip(path):
            continue
        section = parse_python_file(path)
        if len(section) > 2:
            lines.extend(section)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def should_skip(path: Path) -> bool:
    return any(part in {"__pycache__", ".venv", "venv"} for part in path.parts)


if __name__ == "__main__":
    main()
