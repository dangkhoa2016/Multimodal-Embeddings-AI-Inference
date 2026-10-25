"""Offline checks for public documentation, translations and release hygiene."""

import re
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
PAIRS = (
    ("README.md", "README.vi.md"),
    ("docs/API_CONTRACT.md", "docs/API_CONTRACT.vi.md"),
    ("docs/OPERATIONS.md", "docs/OPERATIONS.vi.md"),
    ("docs/CLIENT_INTEGRATION.md", "docs/CLIENT_INTEGRATION.vi.md"),
    ("docs/MODEL_VOLUMES.md", "docs/MODEL_VOLUMES.vi.md"),
    ("docs/TROUBLESHOOTING.md", "docs/TROUBLESHOOTING.vi.md"),
    ("docs/VERIFICATION.md", "docs/VERIFICATION.vi.md"),
    ("docs/NOTEBOOK_GUIDE.md", "docs/NOTEBOOK_GUIDE.vi.md"),
)
GENERAL_PAIRS = (
    ("benchmarks/results/README.md", "benchmarks/results/README.vi.md"),
    ("docs/ACCEPTANCE_TEMPLATE.md", "docs/ACCEPTANCE_TEMPLATE.vi.md"),
    ("SECURITY.md", "SECURITY.vi.md"),
    ("LICENSE-NOTES.md", "LICENSE-NOTES.vi.md"),
    ("CONTRIBUTING.md", "CONTRIBUTING.vi.md"),
    ("CHANGELOG.md", "CHANGELOG.vi.md"),
)
MODEL_NAMES = ("Qwen/Qwen3-VL-Embedding-8B", "tencent/WeMM-Embedding-9B")
FORBIDDEN_DOCS = ("docs/PUBLISH_GITHUB.md", "docs/PUBLISH_GITHUB.vi.md")
REQUIRED_GITHUB = (
    ".github/CODEOWNERS",
    ".github/CODE_OF_CONDUCT.md",
    ".github/SUPPORT.md",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/ISSUE_TEMPLATE/config.yml",
    ".github/ISSUE_TEMPLATE/bug_report.yml",
    ".github/ISSUE_TEMPLATE/feature_request.yml",
    ".github/ISSUE_TEMPLATE/documentation.yml",
    ".github/ISSUE_TEMPLATE/question.yml",
    ".github/workflows/ci.yml",
    ".github/dependabot.yml",
)
# Patterns apply only to public-facing Markdown; source code may need
# implementation-specific resource names and paths.
EDITORIAL_PATTERNS = (
    r"\b935\s+(?:images|rows|ảnh|hàng)\b",
    r"CompactFaces[^\n]*Synthetic Face Dataset",
    r"previous development draft",
    r"master handoff",
    r"[a-z0-9-]+--[a-z0-9-]+\.us-east\.modal\.run",
    r"modal[-_]secret\.env",
    r"\.config/multimodal-embeddings-ai-inference",
    r"(?<![a-f0-9])[a-f0-9]{64}(?![a-f0-9])",
    r"(?:https?://)[^ \n'\"<>]+--[^ \n'\"<>]+\.modal\.run",
)
problems = []
# Enforce the repository-wide bilingual language switcher contract.
_LOCALE_PAIRS = (
    ('docs/ACCEPTANCE_TEMPLATE.md', 'docs/ACCEPTANCE_TEMPLATE.vi.md'),
    ('README.md', 'README.vi.md'),
    ('docs/API_CONTRACT.md', 'docs/API_CONTRACT.vi.md'),
    ('docs/CLIENT_INTEGRATION.md', 'docs/CLIENT_INTEGRATION.vi.md'),
    ('docs/NOTEBOOK_GUIDE.md', 'docs/NOTEBOOK_GUIDE.vi.md'),
    ('docs/OPERATIONS.md', 'docs/OPERATIONS.vi.md'),
    ('docs/MODEL_VOLUMES.md', 'docs/MODEL_VOLUMES.vi.md'),
    ('docs/TROUBLESHOOTING.md', 'docs/TROUBLESHOOTING.vi.md'),
    ('docs/VERIFICATION.md', 'docs/VERIFICATION.vi.md'),
    ('CHANGELOG.md', 'CHANGELOG.vi.md'),
    ('CONTRIBUTING.md', 'CONTRIBUTING.vi.md'),
    ('LICENSE-NOTES.md', 'LICENSE-NOTES.vi.md'),
    ('SECURITY.md', 'SECURITY.vi.md'),
    ('benchmarks/README.md', 'benchmarks/README.vi.md'),
    ('benchmarks/results/README.md', 'benchmarks/results/README.vi.md'),
)
for _english, _vietnamese in _LOCALE_PAIRS:
    for _path_name, _expected in (
        (_english,
         f'🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt]({Path(_vietnamese).name})'),
        (_vietnamese,
         f'🌐 Ngôn ngữ / Language: [English]({Path(_english).name}) | **Tiếng Việt**'),
    ):
        _content = (ROOT / _path_name).read_text(encoding='utf-8')
        if sum(_line == _expected for _line in _content.splitlines()) != 1:
            problems.append(f'incorrect bilingual switcher: {_path_name}')
for name in REQUIRED_GITHUB:
    if not (ROOT / name).is_file():
        problems.append(f"missing community file: {name}")

# Only the consolidated workflow should be shipped in a source release.
workflow_dir = ROOT / ".github" / "workflows"
workflow_files = sorted(
    str(path.relative_to(ROOT)) for path in workflow_dir.iterdir()
    if path.is_file() and path.suffix in {".yml", ".yaml"}
)
if workflow_files != [".github/workflows/ci.yml"]:
    problems.append(f"expected exactly one CI workflow: {workflow_files}")

version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
badge_fragments = (
    "actions/workflows/ci.yml/badge.svg?branch=main",
    "img.shields.io/badge/version-" + version,
    "img.shields.io/badge/license-MIT",
    "img.shields.io/badge/Python-3.11",
    "img.shields.io/badge/FastAPI-inference",
    "img.shields.io/badge/Modal-NVIDIA%20L40S",
    "img.shields.io/badge/input-image%20%2B%20text",
)
for name in ("README.md", "README.vi.md"):
    content = (ROOT / name).read_text()
    for badge in badge_fragments:
        if badge not in content:
            problems.append(f"missing badge in {name}: {badge}")
allowed_integration_docs = {"CLIENT_INTEGRATION.md", "CLIENT_INTEGRATION.vi.md"}
for path in (ROOT / "docs").glob("*_INTEGRATION*.md"):
    if path.name not in allowed_integration_docs:
        problems.append(f"obsolete framework integration document: {path.name}")
for name in FORBIDDEN_DOCS:
    if (ROOT / name).exists():
        problems.append(f"obsolete publication file: {name}")

def markdown_structure(lines: list[str]) -> tuple:
    """Compare bilingual structure without forcing identical translated line wraps."""
    headings, fences, table_columns = [], [], []
    in_fence = False
    for line in lines:
        stripped = line.strip()
        match = re.match(r"^(`{3,}|~{3,})(.*)$", stripped)
        if match:
            if not in_fence:
                fences.append(match.group(1)[0] + match.group(2).strip())
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        heading = re.match(r"^(#{1,6})\s+", stripped)
        if heading:
            headings.append(len(heading.group(1)))
        if stripped.startswith("|") and stripped.endswith("|"):
            table_columns.append(stripped.count("|") - 1)
    # This checks order/shape, not semantic translation quality.
    return tuple(headings), tuple(fences), tuple(table_columns)


for english, vietnamese in PAIRS + GENERAL_PAIRS:
    left, right = ROOT / english, ROOT / vietnamese
    if not left.exists() or not right.exists():
        problems.append(f"missing translated document: {english} / {vietnamese}")
        continue
    a, b = left.read_text().splitlines(), right.read_text().splitlines()
    if markdown_structure(a) != markdown_structure(b):
        problems.append(
            f"bilingual heading/fence/table structure mismatch: {english} / {vietnamese}"
        )
    if (english, vietnamese) in PAIRS:
        for model in MODEL_NAMES:
            if model not in left.read_text() or model not in right.read_text():
                problems.append(f"missing full checkpoint name in {english}: {model}")

for path in [ROOT / "README.md", ROOT / "README.vi.md", *(ROOT / "docs").glob("*.md"),
             ROOT / "LICENSE-NOTES.md"]:
    if not path.exists():
        continue
    content = path.read_text()
    for pattern in EDITORIAL_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            problems.append(f"environment/development text in {path.relative_to(ROOT)}")

def markdown_fence_errors(path: Path) -> list[str]:
    """Reject invalid closing fences and unfinished Markdown code blocks."""
    errors: list[str] = []
    opening: tuple[str, int, int] | None = None
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        marker_match = re.match(r"^ {0,3}(" + re.escape(chr(96)) + r"{3,}|~{3,})(.*)$", line)
        if marker_match is None:
            continue
        marker, tail = marker_match.groups()
        if opening is None:
            opening = (marker[0], len(marker), line_no)
        elif marker[0] == opening[0] and len(marker) >= opening[1]:
            if tail.strip():
                errors.append(
                    f"{path.name}:{line_no}: closing code fence must be alone on its line"
                )
            opening = None
    if opening is not None:
        errors.append(f"{path.name}:{opening[2]}: unterminated Markdown code fence")
    return errors


for markdown_path in sorted(ROOT.rglob("*.md")):
    if any(part in {".git", ".venv", "node_modules"} for part in markdown_path.parts):
        continue
    problems.extend(markdown_fence_errors(markdown_path))



# Check every local Markdown link and heading anchor (no network access required).
# This runs in GitHub Actions through the existing audit_release.py step.

def heading_anchors(source: str) -> set[str]:
    anchors: set[str] = set()
    used: set[str] = set()
    in_fence = False
    for line in source.splitlines():
        fence = re.match(r'^ {0,3}(`{3,}|~{3,})', line)
        if fence:
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = re.match(r'^ {0,3}#{1,6}\s+(.+?)\s*#*\s*$', line)
        if not match:
            continue
        heading = re.sub(r'!?\[([^\]]+)\]\([^)]+\)', r'\1', match.group(1))
        heading = re.sub(r'<[^>]*>', '', heading)
        heading = re.sub(r'[`*_~]', '', heading).lower()
        base = ''.join(ch for ch in heading if ch.isalnum() or ch in ' _-')
        base = re.sub(r'\s+', '-', base.strip())
        candidate, suffix = base, 1
        while candidate in used:
            candidate = f'{base}-{suffix}'
            suffix += 1
        used.add(candidate)
        anchors.add(candidate)
    return anchors


def local_links(source: str):
    in_fence = False
    for line_no, line in enumerate(source.splitlines(), 1):
        fence = re.match(r'^ {0,3}(`{3,}|~{3,})', line)
        if fence:
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in re.finditer(r'\]\(([^)\n]+)\)', line):
            raw = match.group(1).strip()
            # Markdown optional title and <angle-bracket> destinations.
            destination = raw.split(' "', 1)[0].strip('<>')
            if destination:
                yield line_no, destination


for markdown_path in sorted(ROOT.rglob('*.md')):
    if any(part in {'.git', '.venv', 'node_modules', '.pytest_cache', '.ruff_cache'}
           for part in markdown_path.relative_to(ROOT).parts):
        continue
    for line_no, href in local_links(markdown_path.read_text(encoding='utf-8')):
        parsed = urlsplit(href)
        if parsed.scheme or parsed.netloc or href.startswith('//'):
            continue
        target_path = (
            (markdown_path.parent / unquote(parsed.path))
            if parsed.path else markdown_path
        ).resolve()
        if not target_path.is_relative_to(ROOT.resolve()):
            problems.append(f'local link escapes repository: {markdown_path.relative_to(ROOT)}:'
                            f'{line_no}')
            continue
        if not target_path.exists() or (parsed.fragment and not target_path.is_file()):
            problems.append(f'broken Markdown file link: {markdown_path.relative_to(ROOT)}:'
                            f'{line_no}: {href}')
            continue
        if parsed.fragment and target_path.suffix.lower() == '.md':
            anchor = unquote(parsed.fragment).lower()
            if anchor not in heading_anchors(target_path.read_text(encoding='utf-8')):
                problems.append(f'broken Markdown heading link: {markdown_path.relative_to(ROOT)}:'
                                f'{line_no}: {href}')


if problems:
    for item in sorted(set(problems)):
        print(f"FAIL: {item}")
    raise SystemExit(1)

print("PASS bilingual documentation")
