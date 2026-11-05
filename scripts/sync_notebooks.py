"""Keep standalone notebooks in sync with maintainable Python code-cell sources.

Sources under notebooks/source are embedded in the published notebooks, so
Kaggle and Colab users do not need to clone the repository.
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("multimodal_embeddings_api_demo", "multimodal_embeddings_benchmark")
HEADER = "# Source for the standalone notebook code cells.\n"
CELL_MARKER = re.compile(r"(?m)^# %% NOTEBOOK CELL ([a-zA-Z0-9_-]+)\n")


def parse_source(path: Path) -> dict[str, str]:
    data = path.read_text(encoding="utf-8")
    if not data.startswith(HEADER):
        raise ValueError(f"{path}: missing source header")
    markers = list(CELL_MARKER.finditer(data))
    if not markers or data[len(HEADER):markers[0].start()].strip():
        raise ValueError(f"{path}: invalid first cell marker")
    result = {}
    for index, match in enumerate(markers):
        key = match.group(1)
        if key in result:
            raise ValueError(f"{path}: duplicate cell id {key}")
        end = markers[index + 1].start() if index + 1 < len(markers) else len(data)
        source = data[match.end():end]
        if not source or not source.endswith("\n"):
            raise ValueError(f"{path}: invalid or unterminated code cell {key}")
        result[key] = source
    return result


def synchronize(name: str, *, write: bool) -> None:
    notebook_path = ROOT / "notebooks" / f"{name}.ipynb"
    source_path = ROOT / "notebooks" / "source" / f"{name}.py"
    if not notebook_path.exists() and not source_path.exists():
        return
    if not notebook_path.is_file() or not source_path.is_file():
        raise ValueError(f"{name}: source and notebook must both be present")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    sources = parse_source(source_path)
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    ids = [cell.get("id") for cell in cells]
    if len(set(ids)) != len(ids) or set(ids) != set(sources):
        raise ValueError(f"{name}: code cell IDs do not match source")
    changes = []
    for cell in cells:
        source = sources[cell["id"]]
        compile(source, f"{source_path}#{cell['id']}", "exec")
        if "".join(cell["source"]) != source:
            changes.append(cell["id"])
            if write:
                cell["source"] = source.splitlines(keepends=True)
    if changes:
        if not write:
            raise ValueError(f"{name}: out-of-sync code cells: {', '.join(changes)}")
        notebook_path.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
    print(f"PASS {name}: {len(cells)} code cells; updated={len(changes)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="Validate notebook/source parity")
    group.add_argument("--write", action="store_true", help="Generate notebook code from source")
    args = parser.parse_args()
    try:
        for name in NAMES:
            synchronize(name, write=args.write)
    except (KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"FAIL notebook synchronization: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
