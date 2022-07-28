import logging
import uuid
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Callable, Optional

from layerapi.api.ids_pb2 import DatasetBuildId, ProjectId

import layer
from layer.clients.data_catalog import DataCatalogClient
from layer.clients.project_service import ProjectServiceClient
from layer.config.config_manager import ConfigManager
from layer.contracts.asset import AssetPath, AssetType
from layer.contracts.fabrics import Fabric
from layer.executables.runtime import BaseFunctionRuntime
from layer.global_context import current_project_full_name, set_has_shown_update_message


_ProjectId = uuid.UUID


class LayerFunctionRuntime(BaseFunctionRuntime):
    def __init__(self, executable_path: Path, project: Optional[str] = None) -> None:
        super().__init__(executable_path)
        self._project = project or _get_current_project_name()
        self._project_id = None
        self._client_config = None
        self._logger = logging.getLogger(__name__)

    def initialise(self) -> None:
        if self._project is None:
            raise LayerFunctionRuntimeError(
                "project not specified and could not be resolved"
            )

        # required to ensure project exists
        self._layer_init()

        self._client_config = ConfigManager().load().client
        self._project_id = self._get_project_id()

    def __call__(self, func: Callable[..., Any]) -> Any:
        self._create_dataset("test", func)

    def _layer_init(self) -> None:
        set_has_shown_update_message(True)
        layer.init(project_name=self._project)

    def _get_project_id(self) -> _ProjectId:
        project_name = current_project_full_name()
        client = ProjectServiceClient.create(self._client_config)
        project = client.get_project(project_name)
        return project.id

    def _create_dataset(self, name: str, func: Callable[..., Any]) -> None:
        client = DataCatalogClient.create(self._client_config, self._logger)
        fabric = Fabric.F_LOCAL.value
        asset_path = AssetPath(name, AssetType.DATASET)
        client.add_dataset(
            project_id=self._project_id,
            asset_path=asset_path,
            description="",
            fabric=fabric,
            func_source="",
            entrypoint="",
            environment="",
        )
        build_response = client.initiate_build(
            ProjectId(value=str(self._project_id)), name, fabric
        )
        client.store_dataset(func(), uuid.UUID(build_response.id.value))
        client.complete_build(build_response.id, name, fabric.value)


def _add_cli_args(parser: ArgumentParser) -> None:
    parser.add_argument("--project", help="project name", required=False)


def _get_current_project_name() -> Optional[str]:
    # try get the project from the global context
    project = current_project_full_name()
    if project is not None:
        return project.project_name

    return None


class LayerFunctionRuntimeError(Exception):
    pass


if __name__ == "__main__":
    LayerFunctionRuntime.main(add_cli_args=_add_cli_args)
