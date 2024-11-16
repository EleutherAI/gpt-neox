import pytest
import toml
from pathlib import Path
from typing import Dict, List, Optional
from packaging.version import parse as parse_version, Version
from dataclasses import dataclass


@dataclass
class Dependency:
    name: str
    version: Optional[str] = None

    @classmethod
    def from_requirement(cls, requirement: str) -> "Dependency":
        """Parse a requirement string into a Dependency object."""
        # Common version specifiers
        specifiers = ["==", ">=", ">", "<=", "<"]
        name = requirement
        version = None

        for spec in specifiers:
            if spec in requirement:
                name, version = requirement.split(spec, 1)
                version = version.strip()
                break

        return cls(name.lower().strip(), version)

    def matches_version(self, other_version: str) -> bool:
        """Check if this dependency's version matches another version string."""
        if not self.version or not other_version:
            return True

        try:
            # Convert versions to comparable objects
            our_version = parse_version(self.version)
            their_version = parse_version(other_version.replace("*", "0"))
            return our_version == their_version
        except ValueError:
            # If versions can't be parsed, fall back to string comparison
            return self.version.replace("*", "0") == other_version.replace("*", "0")


class DependencyValidator:
    def __init__(self, requirements_dir: Path):
        self.requirements_dir = requirements_dir

    def parse_requirements(self, file_path: Path) -> List[Dependency]:
        """Parse requirements.txt file into a list of Dependencies."""
        try:
            with open(file_path, "r") as f:
                lines = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.startswith("#")
                ]
                return [Dependency.from_requirement(line) for line in lines]
        except FileNotFoundError:
            raise FileNotFoundError(f"Requirements file not found: {file_path}")
        except Exception as e:
            raise ValueError(f"Error parsing requirements file {file_path}: {str(e)}")

    def parse_pyproject(self, file_path: Path) -> Dict[str, str]:
        """Parse pyproject.toml file and extract dependencies."""
        try:
            with open(file_path, "r") as f:
                pyproject_data = toml.load(f)
                return {
                    name.lower(): str(version)
                    for name, version in pyproject_data["tool"]["poetry"][
                        "dependencies"
                    ].items()
                    if name.lower() != "python"  # Exclude Python version
                }
        except FileNotFoundError:
            raise FileNotFoundError(f"pyproject.toml file not found: {file_path}")
        except Exception as e:
            raise ValueError(f"Error parsing pyproject.toml {file_path}: {str(e)}")

    def compare_dependencies(
        self, req_deps: List[Dependency], pyproject_deps: Dict[str, str]
    ) -> tuple[bool, List[str]]:
        """Compare dependencies between requirements.txt and pyproject.toml."""
        mismatches = []

        for req in req_deps:
            if req.name not in pyproject_deps:
                mismatches.append(
                    f"Dependency '{req.name}' not found in pyproject.toml"
                )
                continue

            if not req.matches_version(pyproject_deps[req.name]):
                mismatches.append(
                    f"Version mismatch for '{req.name}': "
                    f"requirements.txt={req.version}, "
                    f"pyproject.toml={pyproject_deps[req.name]}"
                )

        return len(mismatches) == 0, mismatches


def get_corresponding_pyproject(req_file: Path) -> Path:
    """Get the corresponding pyproject.toml file for a requirements file."""
    env_name = req_file.stem.split("-")[1]
    return req_file.parent / f"pyproject-{env_name}.toml"


@pytest.mark.parametrize("req_file", Path("requirements").glob("requirements-*.txt"))
def test_pyproject_matches_requirements(req_file: Path):
    """Test that requirements.txt dependencies match pyproject.toml dependencies."""
    validator = DependencyValidator(req_file.parent)
    pyproject_file = get_corresponding_pyproject(req_file)

    # Parse both dependency files
    req_deps = validator.parse_requirements(req_file)
    pyproject_deps = validator.parse_pyproject(pyproject_file)

    # Compare dependencies and get detailed mismatches
    is_match, mismatches = validator.compare_dependencies(req_deps, pyproject_deps)

    # Create detailed error message if there are mismatches
    if not is_match:
        error_msg = "\n".join(
            [
                f"\nDependency mismatches found between {req_file} and {pyproject_file}:",
                *[f"- {msg}" for msg in mismatches],
            ]
        )
        pytest.fail(error_msg)
