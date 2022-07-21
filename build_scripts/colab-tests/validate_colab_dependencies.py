import pathlib
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Set

from poetry.core.packages.dependency import Dependency
from poetry.core.packages.package import Package
from poetry.core.semver.version import Version
from poetry.factory import Factory


# If you add any new entries to this list, please make sure you test your changes in Colab manually first!
IGNORED_VIOLATIONS = {
    "Version check failed for 'grpcio'. Colab has 1.46.3, but 'grpcio-tools' requires >=1.47.0",
    "Version check failed for 'humanize'. Colab has 0.5.1, but 'layer' requires >=3.11.0",
    "Version check failed for 'protobuf'. Colab has 3.17.3, but 'layer-api' requires >=3.19.4",
    "Version check failed for 'psutil'. Colab has 5.4.8, but 'layer' requires >=5.9.1,<6.0.0",
    "Version check failed for 'pyarrow'. Colab has 6.0.1, but 'layer' requires 7.0.0",
    "Version check failed for 'pyyaml'. Colab has 3.13, but 'huggingface-hub' requires >=5.1",
    "Version check failed for 'pyyaml'. Colab has 3.13, but 'mlflow-skinny' requires >=5.1",
    "Version check failed for 'pyyaml'. Colab has 3.13, but 'transformers' requires >=5.1",
    "Version check failed for 'urllib3'. Colab has 1.24.3, but 'botocore' requires >=1.25.4,<1.27",
    "Version check failed for 'pillow'. Colab has 7.1.2, but 'imageio' requires >=8.3.2",
}


def parse_colab_packages(path: pathlib.Path) -> List[Package]:
    with open(path) as packages_file:
        lines = packages_file.readlines()
    matches = [re.search(r"^([^\s]*) +(\d+\.\d+.*)$", line) for line in lines]

    versions = {
        Package(match.group(1), Version.parse(match.group(2)))
        for match in matches
        if match is not None
    }

    return versions


@dataclass(frozen=True)
class PackageContraint:
    dependency: Dependency
    parent: str


class PackageConstraints:
    def __init__(self) -> None:
        self.constraints = defaultdict(list)

    def add_constraint(self, constraint: PackageContraint, package_name: str) -> None:
        self.constraints[package_name].append(constraint)

    def get_constraints(self, package_name: str) -> List[PackageContraint]:
        return self.constraints[package_name]


def get_package_constraints(path):
    poetry = Factory().create_poetry(path)

    locker = poetry.locker

    constraints = PackageConstraints()

    for dep in poetry.package.requires:
        constraints.add_constraint(PackageContraint(dep, "layer"), dep.name)

    transitive_dependencies = list(
        locker.get_project_dependency_packages(
            project_requires=poetry.package.requires, extras=[]
        )
    )

    for dependency in transitive_dependencies:
        for transitive_dependency in dependency.package.requires:
            constraints.add_constraint(
                PackageContraint(transitive_dependency, dependency.name),
                transitive_dependency.name,
            )

    return constraints


def find_violations(
    colab_packages: List[Package], package_constraints: PackageConstraints
) -> Set[str]:
    violations = set()
    for colab_package in colab_packages:
        pkg_constraints = package_constraints.get_constraints(colab_package.name)
        for pkg_constraint in pkg_constraints:
            if not pkg_constraint.dependency.constraint.allows(colab_package.version):
                violations.add(
                    f"Version check failed for '{colab_package.name}'. Colab has {colab_package.pretty_version}, but '{pkg_constraint.parent}' requires {pkg_constraint.dependency.constraint}"
                )
    return violations


if __name__ == "__main__":
    parent_dir = pathlib.Path(__file__).parent
    colab_packages = parse_colab_packages(parent_dir / "colab-packages.txt")
    package_constraints = get_package_constraints(
        parent_dir.parent.resolve().parent.parent
    )

    violations = (
        find_violations(colab_packages, package_constraints) - IGNORED_VIOLATIONS
    )

    if len(violations) > 0:
        print("Found violations:")
        print("\n".join(sorted(list(violations))))
        exit(1)
    print("Success")
