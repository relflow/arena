"""Report files: content digests for provenance and atomic publication of a run's result."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

__all__ = ["hashes", "publish"]


def hashes(paths: dict[str, Path]) -> dict[str, str]:
    result = {}
    for name, path in paths.items():
        with path.open("rb") as stream:
            result[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def publish(report: dict[str, Any], path: Path, *, replace_existing: bool = False) -> None:
    """Claim a fresh report, then atomically update only this invocation's claim."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="smoke-report-", dir=path.parent) as temporary:
        pending = Path(temporary) / "report.json"
        pending.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if replace_existing:
            os.replace(pending, path)
        else:
            try:
                os.link(pending, path)
            except OSError as error:
                if error.errno not in {errno.EXDEV, errno.ENOTSUP, errno.EPERM}:
                    raise
                # Some object-backed volumes cannot link files. Fresh creation
                # remains exclusive; readers must await the completed receipt.
                destination = path.open("xb")
                try:
                    with destination, pending.open("rb") as source:
                        shutil.copyfileobj(source, destination)
                except BaseException:
                    path.unlink(missing_ok=True)
                    raise
