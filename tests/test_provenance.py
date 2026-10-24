"""Installed-package identity and benchmark assets are independent source contracts."""

import hashlib

import pytest

from benchmarks import provenance


def digest(files, root):
    value = hashlib.sha256()
    for path in sorted(files):
        value.update(path.relative_to(root).as_posix().encode())
        value.update(path.read_bytes())
    return value.hexdigest()


@pytest.fixture
def dependency(tmp_path, monkeypatch):
    checkout = tmp_path / "former-checkout"
    source = checkout / "src/relflow"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("version = 'test'\n")
    (source / "model.py").write_text("class Model: pass\n")
    (source / "nested").mkdir()
    (source / "nested/training.py").write_text("def fit(): pass\n")
    installed = tmp_path / "venv/site-packages/relflow"
    installed.mkdir(parents=True)
    for path in source.rglob("*.py"):
        destination = installed / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    monkeypatch.setattr(provenance.rf, "__file__", str(installed / "__init__.py"))
    return checkout, source, installed


def test_installed_dependency_hash_retains_original_checkout_identity(dependency):
    checkout, source, _ = dependency
    assert provenance.source_hash() == digest(source.rglob("*.py"), checkout)


def test_benchmark_project_does_not_need_a_relflow_checkout(tmp_path, dependency):
    project = tmp_path / "research"
    asset = project / "benchmarks/relarena/task.yaml"
    asset.parent.mkdir(parents=True)
    asset.write_text("task: driver-dnf\n")
    runner = project / "benchmarks/dnf.py"
    runner.write_text("def main(): pass\n")
    source, evaluator = provenance.fingerprints(project, ["benchmarks/dnf.py"])
    assert source == provenance.source_hash()
    assert evaluator == digest([asset, runner], project)
    assert not (project / "src/relflow").exists()


def test_dependency_changes_do_not_change_evaluator_hash(tmp_path, dependency):
    project = tmp_path / "research"
    asset = project / "benchmarks/relarena/run.py"
    asset.parent.mkdir(parents=True)
    asset.write_text("initial")
    before = provenance.fingerprints(project)
    installed = dependency[2]
    (installed / "model.py").write_text("changed")
    after = provenance.fingerprints(project)
    assert before[0] != after[0]
    assert before[1] == after[1]


def test_unrelated_checkout_changes_do_not_change_installed_dependency_identity(tmp_path, dependency):
    before = provenance.source_hash()
    (dependency[1] / "model.py").write_text("unrelated local checkout mutation")
    assert provenance.source_hash() == before


def test_assets_ignore_runtime_outputs_and_project_location(tmp_path, dependency):
    identities = []
    for directory in ("first", "second"):
        project = tmp_path / directory
        asset = project / "benchmarks/relarena/query.sql"
        asset.parent.mkdir(parents=True)
        asset.write_text("SELECT 1")
        (asset.parent / "result.json").write_text(directory)
        (asset.parent / "tensor.pt").write_bytes(directory.encode())
        identities.append(provenance.fingerprints(project))
    assert identities[0] == identities[1]


def test_additions_may_be_explicit_python_sql_yaml_or_text(tmp_path, dependency):
    project = tmp_path / "research"
    additions = []
    for extension in ("py", "sql", "yaml", "txt"):
        path = project / f"benchmarks/task.{extension}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(extension)
        additions.append(str(path.relative_to(project)))
    assert provenance.fingerprints(project, additions)[1] == digest([project / name for name in additions], project)


@pytest.mark.parametrize("name", ["missing.py", "../outside.py", "benchmarks/output.json"])
def test_evaluator_rejects_missing_external_or_noncode_additions(tmp_path, dependency, name):
    project = tmp_path / "research"
    asset = project / "benchmarks/relarena/task.yaml"
    asset.parent.mkdir(parents=True)
    asset.write_text("task: unit")
    (tmp_path / "outside.py").write_text("external")
    (project / "benchmarks/output.json").write_text("{}")
    with pytest.raises(ValueError, match="evaluator addition"):
        provenance.fingerprints(project, [name])


def test_empty_installed_package_and_empty_evaluator_are_errors(tmp_path, monkeypatch):
    package = tmp_path / "package"
    package.mkdir()
    monkeypatch.setattr(provenance.rf, "__file__", str(package / "__init__.py"))
    with pytest.raises(ValueError, match="no Python sources"):
        provenance.source_hash()
    with pytest.raises(ValueError, match="assets are missing"):
        provenance.fingerprints(tmp_path)
