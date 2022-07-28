import os
import runpy
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from layer.executables.packager import get_function_package_info


class BaseFunctionRuntime:
    """Base skeleton of a function runtime."""

    def __init__(self, executable_path: Path) -> None:
        self._executable_path = executable_path

    def initialise(self) -> None:
        """Any initialisation required to run the function."""

    @property
    def executable_path(self) -> Path:
        return self._executable_path

    def install_packages(self, packages: Sequence[str]) -> None:
        """Installs packages required to run the function."""
        _run_pip_install(packages)

    def __call__(self, func: Callable[..., Any]) -> Any:
        """Called from the executable to run the function."""
        return func()

    def run_executable(self) -> Any:
        """Runs the packaged function."""
        runpy.run_path(
            str(self.executable_path),
            run_name="__main__",
            init_globals={"__function_runtime": self},
        )

    @classmethod
    def execute(cls, executable_path: Path, *args: Any, **kwargs: Any) -> None:
        """Initialises the environment, installs packages and runs the executable."""
        _validate_executable_path(executable_path)
        package_info = get_function_package_info(executable_path)
        runtime = cls(executable_path, *args, **kwargs)
        runtime.initialise()
        runtime.install_packages(packages=package_info.pip_dependencies)
        runtime.run_executable()

    @classmethod
    def main(
        cls, add_cli_args: Optional[Callable[[ArgumentParser], None]] = None
    ) -> None:
        parser = ArgumentParser(description="Function runtime")

        parser.add_argument(
            "executable_path",
            type=Path,
            help="the local file path of the executable",
        )

        if add_cli_args is not None:
            add_cli_args(parser)

        args = parser.parse_args()

        cls.execute(**vars(args))


def _validate_executable_path(executable_path: Path) -> None:
    if not executable_path:
        raise FunctionRuntimeError("executable path is required")
    if not executable_path.exists():
        raise FunctionRuntimeError(f"executable path does not exist: {executable_path}")
    if not os.path.isfile(str(executable_path)):
        raise FunctionRuntimeError(f"executable path is not a file: {executable_path}")


class FunctionRuntimeError(Exception):
    pass


def _run_pip_install(packages: Sequence[str]) -> None:
    if len(packages) == 0:
        return

    pip_install = [
        sys.executable,
        "-m",
        "pip",
        "--quiet",
        "--disable-pip-version-check",
        "--no-color",
        "install",
    ] + list(packages)

    import subprocess  # nosec

    result = subprocess.run(
        pip_install,
        shell=False,  # nosec
        text=True,
        check=False,
        capture_output=True,
    )

    if result.returncode != 0:
        raise FunctionRuntimeError(f"package instalation failed:\n{result.stderr}")


if __name__ == "__main__":
    BaseFunctionRuntime.main()
