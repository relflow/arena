"""DuckDB budgets resolved from the benchmark process and its spill filesystem."""

from __future__ import annotations

import math
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path

MIB = 1024**2


def read(path: Path) -> str:
    """Missing or inaccessible kernel accounting is an optional constraint."""
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def cgroups(proc: Path = Path("/proc")) -> list[tuple[Path, Path, str]]:
    """Locate this process's v1/v2 hierarchies through its own mount namespace."""
    memberships = [line.split(":", 2) for line in read(proc / "self/cgroup").splitlines()]
    result = []
    for line in read(proc / "self/mountinfo").splitlines():
        fields, separator, filesystem = line.partition(" - ")
        if not separator:
            continue
        mounted, details = fields.split(), filesystem.split()
        if len(mounted) < 5 or len(details) < 3 or details[0] not in ("cgroup", "cgroup2"):
            continue
        root, mount = (
            Path(re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)) for value in mounted[3:5]
        )
        controllers = set(details[2].split(","))
        for _, names, member in memberships:
            kind = "v2" if details[0] == "cgroup2" and not names else ""
            if details[0] == "cgroup":
                kind = next(
                    (name for name in ("memory", "cpu") if name in names.split(",") and name in controllers), ""
                )
            if not kind:
                continue
            membership = Path(member)
            if membership.is_relative_to(root):
                current = mount / membership.relative_to(root)
            elif membership == Path("/"):
                current = mount
            else:
                continue
            while current.is_relative_to(mount):
                record = (current, mount, kind)
                if record not in result:
                    result.append(record)
                if current == mount:
                    break
                current = current.parent
    return result


def hardware(proc: Path = Path("/proc")) -> dict:
    """Respect affinity, CPU quotas, and every ancestor's remaining memory."""
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    available = total
    for line in read(proc / "meminfo").splitlines():
        if line.startswith("MemAvailable:"):
            available = int(line.split()[1]) * 1024
    constraints = []
    for path, _, kind in cgroups(proc):
        constraint = {"path": str(path), "kind": kind}
        if kind in ("v2", "memory"):
            limit = read(path / ("memory.max" if kind == "v2" else "memory.limit_in_bytes"))
            usage = read(path / ("memory.current" if kind == "v2" else "memory.usage_in_bytes"))
            if limit.isdigit():
                bound = int(limit)
                total = min(total, bound)
                statistics = dict(line.split() for line in read(path / "memory.stat").splitlines())
                inactive = statistics.get("inactive_file" if kind == "v2" else "total_inactive_file", "0")
                reclaimable = int(inactive) if inactive.isdigit() else 0
                remaining = max(0, bound - max(0, int(usage) - reclaimable)) if usage.isdigit() else bound
                available = min(available, remaining)
                constraint.update(
                    memory_limit_bytes=bound, memory_available_bytes=remaining, reclaimable_file_cache_bytes=reclaimable
                )
        if kind in ("v2", "cpu"):
            quota = (
                read(path / "cpu.max").split()
                if kind == "v2"
                else [read(path / "cpu.cfs_quota_us"), read(path / "cpu.cfs_period_us")]
            )
            if len(quota) == 2 and all(value.isdigit() for value in quota) and int(quota[1]) > 0:
                cpus = min(cpus, max(1, math.ceil(int(quota[0]) / int(quota[1]))))
                constraint["cpu_quota"] = int(quota[0]) / int(quota[1])
        if len(constraint) > 2:
            constraints.append(constraint)
    return {
        "cpus": cpus,
        "memory_total_bytes": total,
        "memory_available_bytes": min(total, available),
        "cgroups": constraints,
    }


def size(value: str, name: str) -> int:
    """Accept finite byte quantities, independently of DuckDB percentage defaults."""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(B|[KMGTPE]I?B)\s*", value.upper()) if isinstance(value, str) else None
    if match is None:
        raise ValueError(f"{name} must be a positive finite byte size such as '80GiB'; got {value!r}")
    unit = match[2]
    exponent = "BKMGTPE".index(unit[0]) if unit != "B" else 0
    amount = int(Decimal(match[1]) * (1024 if "I" in unit else 1000) ** exponent)
    if amount < 1:
        raise ValueError(f"{name} must be at least one byte; got {value!r}")
    return amount


def resolve(
    directory: Path,
    *,
    threads: int | None = None,
    memory_limit: str | None = None,
    max_temp_directory_size: str | None = None,
) -> dict:
    """Record explicit budgets with room for resident input frames and the OS."""
    detected = hardware()
    if threads is None:
        threads = detected["cpus"]
    if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= detected["cpus"]:
        raise ValueError(f"duckdb_threads must be between 1 and {detected['cpus']} available CPUs; got {threads!r}")
    memory = (
        size(memory_limit, "duckdb_memory_limit")
        if memory_limit is not None
        else int(min(detected["memory_total_bytes"] * 0.75, detected["memory_available_bytes"] * 0.8)) // MIB * MIB
    )
    if not 1 <= memory <= detected["memory_total_bytes"]:
        raise ValueError(f"DuckDB memory budget {memory} bytes exceeds available process capacity or is empty")
    directory = Path(directory).resolve()
    existing = directory
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    spill_bytes = (
        size(max_temp_directory_size, "duckdb_max_temp_directory_size")
        if max_temp_directory_size is not None
        else int(free * 0.8) // MIB * MIB
    )
    if not 1 <= spill_bytes <= free:
        raise ValueError(f"DuckDB spill budget {spill_bytes} bytes must fit within {free} free bytes at {existing}")
    return {
        "settings": {
            "threads": threads,
            "memory_limit": f"{memory}B",
            "temp_directory": str(directory),
            "max_temp_directory_size": f"{spill_bytes}B",
            "preserve_insertion_order": False,
        },
        "hardware": {**detected, "spill_filesystem_free_bytes": free, "spill_filesystem_path": str(existing)},
        "policy": "Auto: all affinity/quota CPUs; min(75% process RAM, 80% available RAM); 80% free spill disk. Explicit overrides recorded.",
    }


@contextmanager
def spill(settings: dict):
    """Own one connection's spill files without deleting another run's files."""
    directory = Path(settings["temp_directory"])
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="duckdb-", dir=directory) as temporary:
        yield {**settings, "temp_directory": temporary}
