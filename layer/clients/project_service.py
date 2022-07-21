import uuid
from contextlib import contextmanager
from logging import Logger
from typing import Iterator, Optional
from uuid import UUID

from layerapi.api.entity.project_pb2 import Project as ProjectMessage
from layerapi.api.entity.project_view_pb2 import ProjectView
from layerapi.api.ids_pb2 import ProjectId
from layerapi.api.service.flowmanager.project_api_pb2 import (
    CreateProjectRequest,
    GetProjectByPathRequest,
    GetProjectByPathResponse,
    GetProjectViewByIdRequest,
    GetProjectViewByIdResponse,
    RemoveProjectByIdRequest,
    UpdateProjectRequest,
)
from layerapi.api.service.flowmanager.project_api_pb2_grpc import ProjectAPIStub

from layer.config import ClientConfig
from layer.contracts.accounts import Account
from layer.contracts.project_full_name import ProjectFullName
from layer.contracts.projects import Project
from layer.exceptions.exceptions import (
    LayerClientResourceAlreadyExistsException,
    LayerClientResourceNotFoundException,
)
from layer.utils.grpc import create_grpc_channel, generate_client_error_from_grpc_error


class ProjectServiceClient:
    _service: ProjectAPIStub

    def __init__(
        self,
        config: ClientConfig,
        logger: Logger,
        service: Optional[ProjectAPIStub] = None,
    ):
        self._grpc_gateway_address = config.grpc_gateway_address
        self._logger = logger
        self._access_token = config.access_token
        self._do_verify_ssl = config.grpc_do_verify_ssl
        self._logs_file_path = config.logs_file_path
        if service is not None:
            self._service = service

    @contextmanager
    def init(self) -> Iterator["ProjectServiceClient"]:
        with create_grpc_channel(
            self._grpc_gateway_address,
            self._access_token,
            do_verify_ssl=self._do_verify_ssl,
            logs_file_path=self._logs_file_path,
        ) as channel:
            self._service = ProjectAPIStub(channel=channel)
            yield self

    @staticmethod
    def _map_project_message_to_project_contract(
        full_name: ProjectFullName, project_msg: ProjectMessage
    ) -> Project:
        project_id = UUID(project_msg.id.value)
        account_id = UUID(project_msg.account_id.value)
        return Project(
            name=full_name.project_name,
            id=project_id,
            account=Account(
                name=full_name.account_name,
                id=account_id,
            ),
        )

    @staticmethod
    def _map_project_view_message_to_project_contract(
        project_view: ProjectView,
    ) -> Project:
        project_id = UUID(project_view.id.value)
        account_id = UUID(project_view.account.id.value)
        return Project(
            name=project_view.name,
            id=project_id,
            account=Account(
                name=project_view.account.name,
                id=account_id,
            ),
        )

    def get_project_by_id(self, project_id: UUID) -> Optional[Project]:
        try:
            resp: GetProjectViewByIdResponse = self._service.GetProjectViewById(
                GetProjectViewByIdRequest(project_id=ProjectId(value=str(project_id)))
            )
            if resp.project is not None:
                return self._map_project_view_message_to_project_contract(resp.project)
        except LayerClientResourceNotFoundException:
            pass
        except Exception as err:
            raise generate_client_error_from_grpc_error(err, "internal")
        return None

    def get_project(self, full_name: ProjectFullName) -> Optional[Project]:
        try:
            resp: GetProjectByPathResponse = self._service.GetProjectByPath(
                GetProjectByPathRequest(path=full_name.path)
            )
            if resp.project is not None:
                return self._map_project_message_to_project_contract(
                    full_name, resp.project
                )
        except LayerClientResourceNotFoundException:
            pass
        except Exception as err:
            raise generate_client_error_from_grpc_error(err, "internal")
        return None

    def remove_project(self, project_id: uuid.UUID) -> None:
        self._service.RemoveProjectById(
            RemoveProjectByIdRequest(project_id=ProjectId(value=str(project_id)))
        )

    def create_project(self, full_name: ProjectFullName) -> Project:
        try:
            resp = self._service.CreateProject(
                CreateProjectRequest(
                    project_full_name=full_name.path,
                    visibility=ProjectMessage.VISIBILITY_PRIVATE,
                )
            )
            return self._map_project_message_to_project_contract(
                full_name, resp.project
            )
        except LayerClientResourceAlreadyExistsException as e:
            raise e
        except Exception as err:
            raise generate_client_error_from_grpc_error(err, "internal")

    def update_project_readme(
        self, project_full_name: ProjectFullName, readme: str
    ) -> None:
        try:
            self._service.UpdateProject(
                UpdateProjectRequest(
                    project_full_name=project_full_name.path, readme=readme
                )
            )
        except LayerClientResourceNotFoundException as e:
            raise e
        except Exception as err:
            raise generate_client_error_from_grpc_error(err, "internal")

    def set_project_visibility(
        self, project_full_name: ProjectFullName, *, is_public: bool
    ) -> None:
        visibility = (
            ProjectMessage.VISIBILITY_PUBLIC
            if is_public
            else ProjectMessage.VISIBILITY_PRIVATE
        )
        try:
            self._service.UpdateProject(
                UpdateProjectRequest(
                    project_full_name=project_full_name.path, visibility=visibility
                )
            )
        except LayerClientResourceNotFoundException as e:
            raise e
        except Exception as err:
            raise generate_client_error_from_grpc_error(err, "internal")
