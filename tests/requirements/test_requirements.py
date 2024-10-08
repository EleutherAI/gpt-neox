import pytest
import toml
from pathlib import Path

def parse_requirements(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

def parse_pyproject(file_path):
    with open(file_path, 'r') as f:
        pyproject_data = toml.load(f)
    return pyproject_data['tool']['poetry']['dependencies']

def normalize_version(version):
    return version.replace('>=', '').replace('==', '').replace('*', '')

def compare_dependencies(req_deps, pyproject_deps):
    for req in req_deps:
        name, _, version = req.partition('==')
        if not version:
            name, _, version = req.partition('>=')
        if not version:
            name, _, version = req.partition('>')
        name = name.lower()
        if name == 'python':
            continue  # Skip Python version comparison
        if name not in pyproject_deps:
            return False
        if version and normalize_version(version) != normalize_version(pyproject_deps[name]):
            return False
    return True

@pytest.mark.parametrize("req_file", Path("requirements").glob("requirements-*.txt"))
def test_pyproject_matches_requirements(req_file):
    pyproject_file = Path("requirements") / f"pyproject-{req_file.stem.split('-')[1]}.toml"
    assert pyproject_file.exists(), f"pyproject.toml file not found for {req_file}"

    req_deps = parse_requirements(req_file)
    pyproject_deps = parse_pyproject(pyproject_file)

    assert compare_dependencies(req_deps, pyproject_deps), \
        f"Dependencies in {req_file} do not match those in {pyproject_file}"