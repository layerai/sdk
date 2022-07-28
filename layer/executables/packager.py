import glob
import inspect
import json
import os
import pickle  # nosec
import shutil  # nosec
import sys
import tempfile
import zipapp
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Optional, Sequence, Tuple

from .. import cloudpickle


FUNCTION_SERIALIZER_NAME = cloudpickle.__name__
FUNCTION_SERIALIZER_VERSION = cloudpickle.__version__  # type: ignore


def package_function(
    function: Callable[..., Any],
    resources: Optional[Sequence[Path]] = None,
    pip_dependencies: Optional[Sequence[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """Packages layer function as a Python executable."""

    if not inspect.isfunction(function) or function.__name__ == "<lambda>":
        raise ValueError("function must be a function")

    with tempfile.TemporaryDirectory() as source_dir:
        source = Path(source_dir)

        # include cloudpickle itself in the executable
        _copy_cloudpickle_package(source)

        requirements_path = source / "requirements.txt"
        with open(requirements_path, mode="w", encoding="utf8") as requirements:
            requirements.write("\n".join(pip_dependencies or []))

        main_path = source / "__main__.py"
        with open(main_path, mode="w", encoding="utf8") as main:
            main.write(_loader_source())

        if resources:
            _package_resources(source, resources)

        function_path = source / "function"
        with open(function_path, mode="wb") as function_:
            # register to pickle by value to ensure unpickling works anywhere, even if a module is not accessible for the runtime
            cloudpickle.register_pickle_by_value(sys.modules[function.__module__])  # type: ignore
            cloudpickle.dump(function, function_, protocol=pickle.DEFAULT_PROTOCOL)  # type: ignore

        metadata_path = source / "metadata.json"
        with open(metadata_path, mode="w", encoding="utf8") as metadata_:
            json.dump(metadata or {}, metadata_, separators=(",", ":"))

        target = (output_dir or Path(".")) / function.__name__

        # create the executable
        zipapp.create_archive(
            source, target, interpreter="/usr/bin/env python", compressed=True
        )

        # ensure the archive is executable
        target.chmod(0o744)

        return target


@dataclass(frozen=True)
class FunctionPackageInfo:
    pip_dependencies: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)


def get_function_package_info(package_path: Path) -> FunctionPackageInfo:
    """Returns package info from a function package file."""
    pip_dependencies: Tuple[str, ...] = ()
    metadata = {}
    with zipfile.ZipFile(package_path) as package:
        with package.open("requirements.txt", "r") as requirements:
            pip_dependencies = tuple(requirements.read().decode("utf8").splitlines())
        with package.open("metadata.json", "r") as metadata_:
            metadata = json.loads(metadata_.read().decode("utf8"))
    return FunctionPackageInfo(pip_dependencies=pip_dependencies, metadata=metadata)


def _copy_cloudpickle_package(target_path: Path) -> None:
    # source path to copy cloudpickle package from
    source_path = Path(cloudpickle.__file__).resolve().parent

    def _package_contents() -> Generator[str, None, None]:
        for module_file in glob.iglob(f"{source_path}{os.sep}*.py"):
            yield module_file
        yield str(source_path / "LICENSE")

    cloudpickle_dir = target_path / "cloudpickle"
    cloudpickle_dir.mkdir(parents=True, exist_ok=True)
    for file in _package_contents():
        file_path = Path(file)
        shutil.copyfile(file_path, cloudpickle_dir / file_path.name)


def _package_resources(source: Path, resources: Sequence[Path]) -> None:
    resource_root = source / "resources"
    resource_root.mkdir(parents=True, exist_ok=True)

    def _copy_resource_file(resource_file: Path) -> None:
        target_path = resource_root / resource_file
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(resource_file, target_path)

    def _walk_dir(dir: Path) -> None:
        for root, dirs, files in os.walk(dir):
            root_path = Path(root)
            for file in files:
                _copy_resource_file(root_path / file)
            for dir_ in dirs:
                _walk_dir(root_path / dir_)

    for resource in resources:
        if resource.is_file():
            _copy_resource_file(resource)
        elif resource.is_dir():
            _walk_dir(resource)


def _loader_source() -> str:
    """Returns the source code of the layer function loader."""

    def _entrypoint() -> None:
        """Entrypoint that is inlined in the executable's __main__.py to load the function
        and communicate with the runtime."""

        # pylint: disable=W0404
        import os
        import shutil
        import tempfile
        import zipfile
        from pathlib import Path

        # cloudpickle is included in the executable
        import cloudpickle

        def _get_function_runtime():  # type: ignore
            """Load the function runtime from the globals dictionary.
            If none is provided, load the default runtime, which calls the function directly."""

            # check for the runtime being provided from the outside
            if "__function_runtime" in globals():
                return __function_runtime  # type: ignore # noqa: F821 # pylint: disable=undefined-variable
            else:

                # return the default runtime, which calls the function directly
                def _execute_function(function):  # type: ignore
                    return function()

                return _execute_function

        def _extract_resources(exec):  # type: ignore
            """Extracts resources from the executable."""
            for entry in exec.infolist():
                if entry.filename.startswith("resources/"):
                    with exec.open(entry, mode="r") as resource:
                        target_path = Path(".") / entry.filename[len("resources/") :]
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        if not entry.is_dir():
                            with target_path.open(mode="wb") as f:
                                shutil.copyfileobj(resource, f)

        def _extract_function(exec, function_path):  # type: ignore
            """Extracts the cloudpickled function from the executable."""
            with exec.open("function", mode="r") as function:
                with open(function_path, mode="wb") as f:
                    shutil.copyfileobj(function, f)

        def _load_function(function_path):  # type: ignore
            """Loads the cloudpickled function."""
            with open(function_path, mode="rb") as function:
                return cloudpickle.load(function)  # type: ignore

        def _execute():  # type: ignore
            """Extracts the contents of the executable and runs the function in the provided runtime."""
            # get the function runtime
            runtime_exec = _get_function_runtime()  # type: ignore

            with tempfile.TemporaryDirectory() as work_dir:
                function_path = os.path.join(work_dir, "function")

                with zipfile.ZipFile(os.path.dirname(__file__)) as exec:
                    _extract_resources(exec)  # type: ignore
                    _extract_function(exec, function_path)  # type: ignore

                function = _load_function(function_path)  # type: ignore
                runtime_exec(function)

        _execute()  # type: ignore

    return f"""
if __name__ == "__main__":
{inspect.getsource(_entrypoint)}
    {_entrypoint.__name__}()
"""
