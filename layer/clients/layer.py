from contextlib import ExitStack, contextmanager
from logging import Logger
from typing import Iterator, Optional

from layer.config import ClientConfig
from layer.utils.grpc.channel import get_grpc_channel

from .account_service import AccountServiceClient
from .data_catalog import DataCatalogClient
from .executor_service import ExecutorClient
from .flow_manager import FlowManagerClient
from .logged_data_service import LoggedDataClient
from .model_catalog import ModelCatalogClient
from .model_training_service import ModelTrainingClient
from .project_service import ProjectServiceClient
from .user_logs_service import UserLogsClient


class LayerClient:
    def __init__(self, config: ClientConfig, logger: Logger):
        self._config = config
        self._logger = logger
        self._data_catalog: Optional[DataCatalogClient] = None
        self._model_catalog = ModelCatalogClient(config, logger)
        self._model_training = ModelTrainingClient(config, logger)
        self._account = AccountServiceClient(config, logger)
        self._flow_manager = FlowManagerClient(config, logger)
        self._user_logs = UserLogsClient(config, logger)
        self._project_service_client = ProjectServiceClient(config, logger)
        self._logged_data_client = LoggedDataClient(config, logger)
        self._executor_client = ExecutorClient(config, logger)

    @contextmanager
    def init(self) -> Iterator["LayerClient"]:
        with ExitStack() as exit_stack:
            exit_stack.enter_context(self._model_catalog.init())
            exit_stack.enter_context(self._model_training.init())
            exit_stack.enter_context(self._account.init())
            exit_stack.enter_context(self._flow_manager.init())
            exit_stack.enter_context(self._user_logs.init())
            exit_stack.enter_context(self.project_service_client.init())
            exit_stack.enter_context(self._executor_client.init())
            exit_stack.enter_context(self._logged_data_client.init())
            yield self

    @property
    def data_catalog(self) -> DataCatalogClient:
        if self._data_catalog is None:
            self._data_catalog = DataCatalogClient.create(self._config, self._logger)
        return self._data_catalog

    @property
    def model_catalog(self) -> ModelCatalogClient:
        return self._model_catalog

    @property
    def model_training(self) -> ModelTrainingClient:
        return self._model_training

    @property
    def account(self) -> AccountServiceClient:
        return self._account

    @property
    def flow_manager(self) -> FlowManagerClient:
        return self._flow_manager

    @property
    def user_logs(self) -> UserLogsClient:
        return self._user_logs

    @property
    def project_service_client(self) -> ProjectServiceClient:
        return self._project_service_client

    @property
    def logged_data_service_client(self) -> LoggedDataClient:
        return self._logged_data_client

    @property
    def executor_service_client(self) -> ExecutorClient:
        return self._executor_client

    def close(self) -> None:
        channel = get_grpc_channel(self._config, closing=True)
        if channel is not None:
            channel.close()
