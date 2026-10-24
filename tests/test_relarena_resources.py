"""Exercise process-aware budgets and owned spill lifetimes."""

import json
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pytest

from benchmarks.relarena import resources

GIB = 1024**3


def kernel(tmp_path, monkeypatch, *, total=125 * GIB, available=120 * GIB, cpus=32):
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "meminfo").write_text(f"MemAvailable: {available // 1024} kB\n")
    monkeypatch.setattr(resources.os, "sched_getaffinity", lambda _: set(range(cpus)), raising=False)
    monkeypatch.setattr(resources.os, "sysconf", lambda name: 4096 if name == "SC_PAGE_SIZE" else total // 4096)
    return proc


def test_auto_uses_lab_capacity_and_reserves_ram_and_disk(tmp_path, monkeypatch):
    proc = kernel(tmp_path, monkeypatch)
    original = resources.hardware
    monkeypatch.setattr(resources, "hardware", lambda: original(proc))
    monkeypatch.setattr(resources.shutil, "disk_usage", lambda _: SimpleNamespace(free=1000 * GIB))
    result = resources.resolve(tmp_path / "new-spill")
    assert result["settings"] == {
        "threads": 32,
        "memory_limit": f"{int(93.75 * GIB)}B",
        "max_temp_directory_size": f"{800 * GIB}B",
        "temp_directory": str(tmp_path / "new-spill"),
        "preserve_insertion_order": False,
    }
    assert result["hardware"]["memory_available_bytes"] == 120 * GIB
    assert not (tmp_path / "new-spill").exists()
    json.dumps(result)


def test_v2_walks_process_and_ancestor_limits(tmp_path, monkeypatch):
    proc = kernel(tmp_path, monkeypatch, cpus=16)
    mount = tmp_path / "cgroup"
    child = mount / "user.slice" / "trial.service"
    child.mkdir(parents=True)
    (proc / "self/cgroup").write_text("0::/user.slice/trial.service\n")
    (proc / "self/mountinfo").write_text(f"10 1 0:20 / {mount} rw - cgroup2 cgroup rw\n")
    for path, limit, usage, quota in (
        (mount, "max", 0, "max 100000"),
        (child.parent, 100 * GIB, 70 * GIB, "150000 100000"),
        (child, 110 * GIB, 20 * GIB, "max 100000"),
    ):
        (path / "memory.max").write_text(str(limit))
        (path / "memory.current").write_text(str(usage))
        (path / "cpu.max").write_text(quota)
    result = resources.hardware(proc)
    assert result["cpus"] == 2
    assert result["memory_total_bytes"] == 100 * GIB
    assert result["memory_available_bytes"] == 30 * GIB
    assert {item["path"] for item in result["cgroups"]} == {str(child), str(child.parent)}
    (child.parent / "memory.stat").write_text(f"inactive_file {50 * GIB}\n")
    with_cache = resources.hardware(proc)
    assert with_cache["memory_available_bytes"] == 80 * GIB
    assert with_cache["cgroups"][1]["reclaimable_file_cache_bytes"] == 50 * GIB


def test_v1_mount_root_maps_current_process_and_respects_both_controllers(tmp_path, monkeypatch):
    proc = kernel(tmp_path, monkeypatch, cpus=8)
    memory, cpu = tmp_path / "memory", tmp_path / "cpu"
    for mount in (memory, cpu):
        (mount / "job").mkdir(parents=True)
    (proc / "self/cgroup").write_text("5:memory:/tenant/job\n4:cpu,cpuacct:/tenant/job\n")
    (proc / "self/mountinfo").write_text(
        f"10 1 0:20 /tenant {memory} rw - cgroup cgroup rw,memory\n"
        f"11 1 0:21 /tenant {cpu} rw - cgroup cgroup rw,cpu,cpuacct\n"
    )
    (memory / "job/memory.limit_in_bytes").write_text(str(8 * GIB))
    (memory / "job/memory.usage_in_bytes").write_text(str(2 * GIB))
    (cpu / "job/cpu.cfs_quota_us").write_text("300000")
    (cpu / "job/cpu.cfs_period_us").write_text("100000")
    result = resources.hardware(proc)
    assert result["cpus"] == 3
    assert result["memory_total_bytes"] == 8 * GIB
    assert result["memory_available_bytes"] == 6 * GIB
    (memory / "job/memory.stat").write_text(f"inactive_file 0\ntotal_inactive_file {GIB}\n")
    assert resources.hardware(proc)["memory_available_bytes"] == 7 * GIB


@pytest.mark.parametrize("value", ["", "-1GB", "nanGB", "inf", "80%", "0B", "IB", "1", None])
def test_sizes_reject_unbounded_or_invalid_budgets(value):
    with pytest.raises(ValueError, match="positive finite|at least one"):
        resources.size(value, "test budget")


@pytest.mark.parametrize("value,expected", [("80GiB", 80 * GIB), ("1.5GB", 1_500_000_000), ("1024B", 1024)])
def test_sizes_resolve_to_explicit_bytes(value, expected):
    assert resources.size(value, "test") == expected


@pytest.mark.parametrize(
    "overrides",
    [{"threads": 0}, {"threads": True}, {"threads": 3}, {"memory_limit": "9GiB"}, {"max_temp_directory_size": "21GiB"}],
)
def test_overrides_cannot_exceed_process_or_filesystem_capacity(tmp_path, monkeypatch, overrides):
    monkeypatch.setattr(
        resources, "hardware", lambda: {"cpus": 2, "memory_total_bytes": 8 * GIB, "memory_available_bytes": 6 * GIB}
    )
    monkeypatch.setattr(resources.shutil, "disk_usage", lambda _: SimpleNamespace(free=20 * GIB))
    with pytest.raises(ValueError):
        resources.resolve(tmp_path, **overrides)


def test_spill_is_connection_owned_and_cleanup_preserves_other_runs(tmp_path):
    parent = tmp_path / "spill"
    parent.mkdir()
    unrelated = parent / "other-run"
    unrelated.write_text("owned elsewhere")
    settings = resources.resolve(parent, threads=1, memory_limit="128MiB", max_temp_directory_size="256MiB")["settings"]
    with pytest.raises(RuntimeError, match="job failed"):
        with resources.spill(settings) as config:
            owned = Path(config["temp_directory"])
            assert owned.parent == parent and owned != parent
            with duckdb.connect(config=config) as connection:
                actual = connection.execute(
                    "SELECT current_setting('threads'), current_setting('memory_limit'), current_setting('max_temp_directory_size')"
                ).fetchone()
                assert actual == (1, "128.0 MiB", "256.0 MiB")
            (owned / "temporary-data").write_text("remove me")
            raise RuntimeError("job failed")
    assert not owned.exists()
    assert unrelated.read_text() == "owned elsewhere"
