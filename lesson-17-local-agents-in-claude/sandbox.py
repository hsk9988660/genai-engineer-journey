"""File tools a model can be pointed at, kept inside one folder.

The lesson's `_safe_path` is a good start: it resolves the path (so `..` and symlinks that leave the folder are caught). What it leaves open, and this closes:
  * the root itself is never resolved, so a symlinked working directory can make every read look "outside"
  * absolute paths, drive letters and parent segments are not refused up front
  * there is no size limit, so one large file overflows a small model's context
  * a binary file raises an exception the loop does not catch
  * secrets inside the project are as readable as source: `.env`, private keys, credential files
  * nothing records that a result contained a secret, which matters the moment a conversation can reach a cloud model (see hybrid.py)

Every tool returns a `ToolOutput`. `sensitive=True` means the text looks like it contains a secret, so the conversation must stay local.
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

DENY_NAMES = frozenset({".env", ".env.local", ".env.production", "id_rsa", "id_ed25519", "id_ecdsa", ".netrc", ".npmrc", ".pypirc", "credentials", "credentials.json", "secrets.json"})
DENY_SUFFIXES = (".pem", ".key", ".pfx", ".p12", ".kdbx", ".keystore")
DENY_DIRS = frozenset({".git", ".ssh", ".aws", ".gnupg"})
_SENSITIVE = [re.compile(p, re.I) for p in (
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----", r"\bAKIA[0-9A-Z]{16}\b", r"\bsk-[A-Za-z0-9_-]{16,}", r"\bghp_[A-Za-z0-9]{20,}", r"\bxox[abp]-[A-Za-z0-9-]{10,}",
    r"\b(password|passwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token)\b\s*[=:]\s*\S{4,}", r"\bbearer\s+[A-Za-z0-9._-]{16,}")]
_PII = [re.compile(p, re.I) for p in (r"[\w.+-]+@[\w-]+\.[\w.-]+", r"\b\d{3}-\d{2}-\d{4}\b", r"\b(?:\d[ -]?){13,16}\b", r"\bconfidential\b")]


class SandboxError(Exception):
    pass


@dataclass(frozen=True)
class ToolOutput:
    text: str
    sensitive: bool = False
    truncated: bool = False


def looks_sensitive(text: str) -> bool:
    """A HEURISTIC list of secret shapes. It will miss secrets it has no pattern for. It exists so the router has something to act on, not to guarantee anything."""
    return any(p.search(text) for p in _SENSITIVE)


def looks_personal(text: str) -> bool:
    return any(p.search(text) for p in _PII)


class Sandbox:
    def __init__(self, root, max_read_bytes: int = 20_000, max_list: int = 200):
        self.root = Path(root).resolve()  # resolved ONCE, so comparisons use the same form as the resolved candidate
        if not self.root.is_dir():
            raise SandboxError("the sandbox root must be an existing directory")
        self.max_read_bytes, self.max_list = max_read_bytes, max_list

    # --- paths ----------------------------------------------------------------------------------------------------------------------
    def _denied(self, rel: Path) -> str | None:
        parts = rel.parts
        if any(p in DENY_DIRS for p in parts[:-1]) or (parts and parts[-1] in DENY_DIRS):
            return "that folder is not available"
        name = parts[-1].lower() if parts else ""
        if name in DENY_NAMES or name.endswith(DENY_SUFFIXES) or name.startswith(".env"):
            return "that file may hold secrets and is not available"
        return None

    def resolve(self, rel) -> Path:
        if not isinstance(rel, str) or not rel.strip():
            raise SandboxError("a file path is required")
        if len(rel) > 300 or any(ord(c) < 32 for c in rel):
            raise SandboxError("the path is too long or contains control characters")
        win, posix = PureWindowsPath(rel), PurePosixPath(rel)
        if win.drive or win.root or posix.is_absolute() or rel.startswith(("/", "\\", "~")):
            raise SandboxError("use a path relative to the project folder")
        if ".." in win.parts or ".." in posix.parts:
            raise SandboxError("parent folder segments ('..') are not allowed")
        full = (self.root / rel).resolve()  # follows symlinks: a link that leads outside resolves outside
        try:
            relative = full.relative_to(self.root)
        except ValueError:
            raise SandboxError("that path is outside the project folder") from None
        reason = self._denied(relative)
        if reason:
            raise SandboxError(reason)
        return full

    # --- tools ------------------------------------------------------------------------------------------------------------------------
    def read_text(self, rel) -> ToolOutput:
        full = self.resolve(rel)
        if not full.is_file():
            raise SandboxError(f"no such file: {rel}")
        size = full.stat().st_size
        with open(full, "rb") as f:
            data = f.read(self.max_read_bytes)
        if b"\x00" in data[:8192]:
            return ToolOutput("[binary file: contents not shown]")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = None
            if size > len(data):  # the cut may have split a multi-byte character (at most 3 bytes of one): drop the partial tail and try again
                for tail in (1, 2, 3):
                    try:
                        text = data[:-tail].decode("utf-8")
                        break
                    except UnicodeDecodeError:
                        continue
            if text is None:
                return ToolOutput("[file is not UTF-8 text: contents not shown]")
        truncated = size > self.max_read_bytes
        if truncated:
            text += f"\n[truncated: showing the first {self.max_read_bytes} of {size} bytes]"
        return ToolOutput(text, looks_sensitive(text), truncated)

    def _inside(self, p: Path) -> bool:
        try:
            p.resolve().relative_to(self.root)
            return True
        except (ValueError, OSError):
            return False

    def _walk(self):
        """Yields (relative Path, absolute Path) for every ordinary file inside the root. Hidden entries, denied names, links and anything whose real location is outside the root are skipped
        (checked by resolving, because a Windows junction is not always reported as a link)."""
        for dirpath, dirnames, filenames in os.walk(self.root, followlinks=False):
            here = Path(dirpath)
            dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d not in DENY_DIRS and not (here / d).is_symlink() and self._inside(here / d))
            for name in sorted(filenames):
                p = here / name
                if p.is_symlink() or name.startswith(".") or not self._inside(p):
                    continue
                rel = p.relative_to(self.root)
                if not self._denied(rel):
                    yield rel, p

    def list_files(self) -> ToolOutput:
        names = []
        for rel, _ in self._walk():
            names.append(rel.as_posix())
            if len(names) >= self.max_list:
                names.append(f"[list cut at {self.max_list} files]")
                break
        return ToolOutput(", ".join(names) if names else "(no files)")

    def analyze(self, rel) -> ToolOutput:
        out = self.read_text(rel)
        lines = out.text.splitlines()
        return ToolOutput(json.dumps({"path": rel, "lines": len(lines), "functions": sum(1 for ln in lines if ln.strip().startswith("def ")),
                                      "todos": sum(1 for ln in lines if "TODO" in ln or "FIXME" in ln), "truncated": out.truncated}), out.sensitive, out.truncated)

    def find_todos(self, max_results: int = 50) -> ToolOutput:
        hits, sensitive = [], False
        for rel, p in self._walk():
            try:
                with open(p, "rb") as f:
                    data = f.read(self.max_read_bytes)
            except OSError:
                continue
            if b"\x00" in data[:8192]:
                continue
            for n, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), start=1):
                if "TODO" in line or "FIXME" in line:
                    sensitive = sensitive or looks_sensitive(line)
                    hits.append(f"{rel.as_posix()}:{n}: {line.strip()[:120]}")
                    if len(hits) >= max_results:
                        return ToolOutput("\n".join(hits) + f"\n[cut at {max_results} results]", sensitive, True)
        return ToolOutput("\n".join(hits) if hits else "No TODO or FIXME comments found.", sensitive)
