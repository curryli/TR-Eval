"""Check source files for common credential and private-location mistakes.

This is a release guard, not a substitute for reviewing files and Git history.
"""

from __future__ import annotations

import ipaddress
import re
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "__pycache__", "build", "dist", "data/results"}
URL = re.compile(r"https?://[^\s\"'<>]+")
PRIVATE_PATH = re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+/")
KEY_LITERAL = re.compile(
    r"(?i)\b(?:api[_-]?key|secret|password|private[_-]?key)\s*[:=]\s*"
    r"[\"']([^\"']{8,})[\"']"
)
KNOWN_KEY = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{20,})\b")


def findings(path: Path) -> list[tuple[int, str]]:
    output = []
    for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if PRIVATE_PATH.search(line):
            output.append((number, "absolute user path"))
        if KEY_LITERAL.search(line) or KNOWN_KEY.search(line):
            output.append((number, "possible credential literal"))
        if "-----BEGIN " + "PRIVATE KEY-----" in line:
            output.append((number, "private key block"))
        for match in URL.finditer(line):
            host = urlsplit(match.group(0).rstrip(".,)]")).hostname or ""
            if host.endswith((".internal", ".intra", ".corp", ".local")):
                output.append((number, "private hostname"))
            try:
                if ipaddress.ip_address(host).is_private:
                    output.append((number, "private IP URL"))
            except ValueError:
                pass
    return output


def main() -> int:
    problems = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in relative.parts):
            continue
        if relative.parts[:2] == ("data", "results"):
            continue
        if path.name == ".env" or path.name.startswith(".env.") or path.suffix in {".key", ".pem", ".p12", ".pfx"}:
            problems.append((relative, 0, "credential file present"))
            continue
        if b"\x00" in path.read_bytes():
            problems.append((relative, 0, "binary file requires review"))
            continue
        problems.extend((relative, line, reason) for line, reason in findings(path))
    for path, line, reason in problems:
        print(f"{path}:{line}: {reason}")
    if problems:
        return 1
    print("Release scan passed for current source files; review Git history separately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
