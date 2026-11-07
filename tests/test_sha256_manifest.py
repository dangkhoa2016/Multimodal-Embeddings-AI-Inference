"""Check checksum generation uses the Git index, including newly staged paths."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_check_mode_is_non_mutating():
    tool = ROOT / "scripts" / "update_sha256_manifest.py"
    code = tool.read_text(encoding="utf-8")
    assert "git" in code and "ls-files" in code and "--cached" in code
    assert "--check" in code
    assert "SHA256SUMS.txt" in code


def test_manifest_generator_uses_staged_files_and_tracks_deletions(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True
        ).stdout

    git("init", "-q")
    (repo / "scripts").mkdir()
    tool = repo / "scripts" / "update_sha256_manifest.py"
    tool.write_bytes((ROOT / "scripts" / "update_sha256_manifest.py").read_bytes())
    a = repo / "a.txt"
    a.write_text("one\n", encoding="utf-8")
    readme = repo / "README.md"
    readme.write_text("# Public README\n", encoding="utf-8")
    vietnamese_readme = repo / "README.vi.md"
    vietnamese_readme.write_text("# README tiếng Việt\n", encoding="utf-8")
    git("add", "a.txt", "README.md", "README.vi.md", "scripts/update_sha256_manifest.py")

    def run(*args):
        return subprocess.run(
            ["python3", str(tool), *args], cwd=repo, capture_output=True, text=True
        )

    assert run().returncode == 0
    manifest = repo / "SHA256SUMS.txt"
    contents = manifest.read_text(encoding="utf-8")
    assert "  a.txt\n" in contents
    assert "  scripts/update_sha256_manifest.py\n" in contents
    assert "  SHA256SUMS.txt" not in contents
    assert "  README.md" not in contents
    assert "  README.vi.md" not in contents
    readme.write_text("# Updated public README\n", encoding="utf-8")
    git("add", "README.md")
    assert run("--check").returncode == 0
    assert run("--check").returncode == 0

    b = repo / "new.txt"
    b.write_text("two\n", encoding="utf-8")
    # Untracked files do not appear until staged, consistent with CI.
    assert run("--check").returncode == 0
    git("add", "new.txt")
    assert run("--check").returncode != 0
    assert run().returncode == 0
    assert "  new.txt\n" in manifest.read_text(encoding="utf-8")

    a.unlink()
    git("add", "-u")
    assert run("--check").returncode != 0
    assert run().returncode == 0
    assert "  a.txt" not in manifest.read_text(encoding="utf-8")
    assert run("--check").returncode == 0
