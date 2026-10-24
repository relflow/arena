"""Fingerprint benchmark code and its independently installed RelFlow dependency."""

from __future__ import annotations

import hashlib
from pathlib import Path

import relflow as rf

__all__ = ["source_hash", "fingerprints"]


def source_hash() -> str:
    """Preserve checkout-era identities while locating code through the installed package."""
    package = Path(rf.__file__).resolve().parent
    files = sorted(package.rglob("*.py"))
    if not files:
        raise ValueError(f"installed RelFlow package has no Python sources at {package}")
    digest = hashlib.sha256()
    for path in files:
        digest.update((Path("src/relflow") / path.relative_to(package)).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def fingerprints(source: Path, additions: list[str] | None = None) -> tuple[str, str]:
    """Hash evaluator assets relative to the independent benchmark project root."""
    source = source.resolve()
    files = [
        path
        for path in (source / "benchmarks/relarena").rglob("*")
        if path.is_file() and path.suffix in {".py", ".yaml", ".sql", ".txt"}
    ]
    for name in additions or ():
        path = source / name
        if (
            not path.is_file()
            or not path.resolve().is_relative_to(source)
            or path.suffix not in {".py", ".yaml", ".sql", ".txt"}
        ):
            raise ValueError(
                f"benchmark evaluator addition must be an existing project code or specification file: {name!r}"
            )
        files.append(path)
    if not files:
        raise ValueError(f"benchmark evaluator assets are missing at {source / 'benchmarks/relarena'}")
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(path.read_bytes())
    return source_hash(), digest.hexdigest()
