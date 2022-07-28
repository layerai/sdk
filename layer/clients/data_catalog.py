import sys
import tempfile
import uuid
from logging import Logger
from pathlib import Path
from typing import Any, Callable, Generator, List, Optional, Tuple

import pandas
import pyarrow
from layerapi.api.entity.dataset_build_pb2 import DatasetBuild as PBDatasetBuild
from layerapi.api.entity.dataset_pb2 import Dataset as PBDataset
from layerapi.api.entity.dataset_version_pb2 import DatasetVersion as PBDatasetVersion
from layerapi.api.ids_pb2 import DatasetBuildId, DatasetId, DatasetVersionId, ProjectId
from layerapi.api.service.datacatalog.data_catalog_api_pb2 import (
    CompleteBuildRequest,
    CompleteBuildResponse,
    GetBuildRequest,
    GetDatasetRequest,
    GetLatestBuildRequest,
    GetPythonDatasetAccessCredentialsRequest,
    GetPythonDatasetAccessCredentialsResponse,
    GetResourcePathsRequest,
    GetVersionRequest,
    InitiateBuildRequest,
    InitiateBuildResponse,
    RegisterDatasetRequest,
    UpdateResourcePathsIndexRequest,
)
from layerapi.api.service.datacatalog.data_catalog_api_pb2_grpc import (
    DataCatalogAPIStub,
)
from layerapi.api.service.dataset.dataset_api_pb2 import (
    Command,
    DatasetQuery,
    DatasetSnapshot,
)
from layerapi.api.value.language_version_pb2 import LanguageVersion
from layerapi.api.value.python_dataset_pb2 import PythonDataset as PBPythonDataset
from layerapi.api.value.python_source_pb2 import PythonSource
from layerapi.api.value.s3_path_pb2 import S3Path
from layerapi.api.value.storage_location_pb2 import StorageLocation
from layerapi.api.value.ticket_pb2 import DatasetPathTicket, DataTicket

from layer.config import ClientConfig
from layer.contracts.assets import AssetPath, AssetType
from layer.contracts.datasets import Dataset, DatasetBuild, DatasetBuildStatus
from layer.contracts.project_full_name import ProjectFullName
from layer.exceptions.exceptions import LayerClientException
from layer.pandas_extensions import _infer_custom_types
from layer.utils.file_utils import tar_directory
from layer.utils.grpc import generate_client_error_from_grpc_error
from layer.utils.grpc.channel import get_grpc_channel
from layer.utils.s3 import S3Util

from .dataset_service import DatasetClient, DatasetClientError


class DataCatalogClient:
    _service: DataCatalogAPIStub

    def __init__(
        self,
        config: ClientConfig,
        logger: Logger,
        dataset_client: Optional["DatasetClient"] = None,
    ):
        self._logger = logger
        self._call_metadata = [("authorization", f"Bearer {config.access_token}")]
        self._s3_endpoint_url = config.s3.endpoint_url
        self._dataset_client = (
            DatasetClient(
                address_and_port=config.grpc_gateway_address,
                access_token=config.access_token,
            )
            if dataset_client is None
            else dataset_client
        )

    @staticmethod
    def create(config: ClientConfig, logger: Logger) -> "DataCatalogClient":
        client = DataCatalogClient(config=config, logger=logger)
        channel = get_grpc_channel(config)
        client._service = DataCatalogAPIStub(  # pylint: disable=protected-access
            channel
        )
        return client

    def _get_python_dataset_access_credentials(
        self, dataset_path: AssetPath
    ) -> GetPythonDatasetAccessCredentialsResponse:
        return self._service.GetPythonDatasetAccessCredentials(
            GetPythonDatasetAccessCredentialsRequest(dataset_path=dataset_path.path()),
        )

    def fetch_dataset(
        self, asset_path: AssetPath, no_cache: bool = False
    ) -> "pandas.DataFrame":
        data_ticket = DataTicket(
            dataset_path_ticket=DatasetPathTicket(path=asset_path.path()),
        )
        command = Command(dataset_query=DatasetQuery(ticket=data_ticket))
        all_partition_data = []
        try:
            for partition_metadata in self._dataset_client.get_partitions_metadata(
                command
            ):
                partition = self._dataset_client.fetch_partition(
                    partition_metadata, no_cache=no_cache
                )
                all_partition_data.append(partition.to_pandas())
        except DatasetClientError as e:
            raise LayerClientException(str(e))

        if len(all_partition_data) > 0:
            df = pandas.concat(all_partition_data, ignore_index=True)
        else:
            df = pandas.DataFrame()

        return df

    def _get_dataset_writer(self, build_id: uuid.UUID, schema: Any) -> Any:
        dataset_snapshot = DatasetSnapshot(build_id=DatasetBuildId(value=str(build_id)))
        return self._dataset_client.get_dataset_writer(
            Command(dataset_snapshot=dataset_snapshot), schema
        )

    def store_dataset(
        self,
        data: "pandas.DataFrame",
        build_id: uuid.UUID,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> None:
        """
        Store a dataset.

        :param name: the name of the dataset
            If empty, the data is written to internal storage
        :param data: dataset data
        :param build_id: dataset build id
        :param progress_callback: progress callback
        """

        # Creates a Record batch from the pandas dataframe
        batch = pyarrow.RecordBatch.from_pandas(
            _infer_custom_types(data), preserve_index=False
        )
        try:
            writer, _ = self._get_dataset_writer(build_id, batch.schema)
            try:
                total_rows = 0  # total rows written
                for chunk in _get_batch_chunks(batch):
                    writer.write_batch(chunk)
                    total_rows += chunk.num_rows
                    if progress_callback:
                        progress_callback(batch.num_rows - total_rows)
            finally:
                writer.close()
        except Exception as err:
            raise generate_client_error_from_grpc_error(
                err, "internal dataset store error"
            )

    def initiate_build(
        self,
        project_id: ProjectId,
        asset_name: str,
        fabric: str,
    ) -> InitiateBuildResponse:
        self._logger.debug("Initiating build for the dataset %r", asset_name)

        resp = self._service.InitiateBuild(
            InitiateBuildRequest(
                dataset_name=asset_name,
                format="python",
                build_entity_type=PBDatasetBuild.BUILD_ENTITY_TYPE_DATASET,
                project_id=project_id,
                fabric=fabric,
            )
        )

        return resp

    def complete_build(
        self,
        dataset_build_id: DatasetBuildId,
        asset_name: str,
        dataset_uri: str,
        error: Optional[Exception] = None,
    ) -> CompleteBuildResponse:
        self._logger.debug("Completing build for the dataset %r", asset_name)

        if error:
            max_error_length = 99_999
            placeholder = "..."
            raw_text = str.format("Dataset build failed with {}", error)
            status = DatasetBuildStatus.FAILED
            success = None
            failure = CompleteBuildRequest.BuildFailed(
                info=(raw_text[: (max_error_length - len(placeholder))] + placeholder)
                if len(raw_text) > max_error_length
                else raw_text
            )
        else:
            status = DatasetBuildStatus.COMPLETED
            success = CompleteBuildRequest.BuildSuccess(
                location=StorageLocation(uri=dataset_uri), schema="{}"
            )
            failure = None

        resp = self._service.CompleteBuild(
            CompleteBuildRequest(
                id=dataset_build_id,
                status=status,
                success=success,
                failure=failure,
            )
        )

        return resp

    def add_dataset(
        self,
        asset_path: AssetPath,
        project_id: uuid.UUID,
        description: str,
        fabric: str,
        func_source: str,
        entrypoint: str,
        environment: str,
        function_home_dir: Optional[Path] = None,
    ) -> str:
        self._logger.debug(
            "Adding or updating a dataset with name %r",
            asset_path.asset_name,
        )
        resp = self._service.RegisterDataset(
            RegisterDatasetRequest(
                name=asset_path.asset_name,
                description=description,
                python_dataset=self._get_pb_python_dataset(
                    asset_path,
                    function_home_dir,
                    fabric,
                    func_source,
                    entrypoint,
                    environment,
                ),
                project_id=ProjectId(value=str(project_id)),
            ),
        )
        return resp.dataset_id.value

    def _get_pb_python_dataset(
        self,
        asset_path: AssetPath,
        fabric: str,
        func_source: str,
        entrypoint: str,
        environment: str,
        function_home_dir: Optional[Path] = None,
    ) -> PBPythonDataset:
        s3_path = (
            self._upload_dataset_source(asset_path, function_home_dir)
            if function_home_dir
            else None
        )
        language_version = _language_version()
        return PBPythonDataset(
            s3_path=s3_path,
            python_source=PythonSource(
                content=func_source,
                entrypoint=entrypoint,
                environment=environment,
                language_version=LanguageVersion(
                    major=language_version[0],
                    minor=language_version[1],
                    micro=language_version[2],
                ),
            ),
            fabric=fabric,
        )

    def _upload_dataset_source(
        self, asset_path: AssetPath, function_home_dir: Path
    ) -> S3Path:
        response = self._get_python_dataset_access_credentials(asset_path)
        archive_name = f"{asset_path.asset_name}.tgz"

        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_path = f"{tmp_dir}/{archive_name}"
            tar_directory(archive_path, function_home_dir)
            S3Util.upload_dir(
                Path(tmp_dir),
                response.credentials,
                response.s3_path,
                endpoint_url=self._s3_endpoint_url,
            )
        return S3Path(
            bucket=response.s3_path.bucket,
            key=f"{response.s3_path.key}{archive_name}",
        )

    def get_resource_paths(
        self, project_full_name: ProjectFullName, function_name: str, path: str = ""
    ) -> List[str]:
        request = GetResourcePathsRequest(
            project_full_name=project_full_name.path,
            function_name=function_name,
            path=path,
        )
        response = self._service.GetResourcePaths(request)
        return response.paths

    def update_resource_paths_index(
        self, project_full_name: ProjectFullName, function_name: str, paths: List[str]
    ) -> None:
        request = UpdateResourcePathsIndexRequest(
            project_full_name=project_full_name.path,
            function_name=function_name,
            paths=paths,
        )
        self._service.UpdateResourcePathsIndex(request)

    def get_dataset_by_name(
        self, project_id: uuid.UUID, name: str, version_name: str = ""
    ) -> Dataset:
        build = self._get_build_by_name(project_id, name, version_name)
        version = self._get_version_by_id(build.dataset_version_id.value)
        dataset = self._get_dataset_by_id(version.dataset_id.value)
        return self._create_dataset(dataset, version, build)

    def get_dataset_by_build_id(self, id_: uuid.UUID) -> Dataset:
        build = self._get_build_by_id(str(id_))
        version = self._get_version_by_id(build.dataset_version_id.value)
        dataset = self._get_dataset_by_id(version.dataset_id.value)
        return self._create_dataset(dataset, version, build)

    def _create_dataset(
        self,
        dataset: PBDataset,
        version: PBDatasetVersion,
        build: PBDatasetBuild,
    ) -> Dataset:

        asset_path = AssetPath(
            asset_type=AssetType.DATASET,
            asset_name=dataset.name,
            asset_version=version.name,
            project_name=dataset.project_name,
        )
        return Dataset(
            id=uuid.UUID(dataset.id.value),
            asset_path=asset_path,
            description=dataset.description,
            schema=version.schema,
            version=version.name,
            uri=build.location.uri,
            metadata=dict(build.location.metadata.value),
            build=DatasetBuild(
                id=build.id.value,
                status=DatasetBuildStatus(build.build_info.status),
                info=build.build_info.info,
                index=str(build.index),
            ),
        )

    def _get_dataset_by_id(self, id_: str) -> PBDataset:
        return self._service.GetDataset(
            GetDatasetRequest(dataset_id=DatasetId(value=id_)),
        ).dataset

    def _get_version_by_id(self, id_: str) -> PBDatasetVersion:
        return self._service.GetVersion(
            GetVersionRequest(version_id=DatasetVersionId(value=id_)),
        ).version

    def _get_build_by_id(self, id_: str) -> PBDatasetBuild:
        return self._service.GetBuild(
            GetBuildRequest(build_id=DatasetBuildId(value=id_)),
        ).build

    def _get_build_by_name(
        self, project_id: uuid.UUID, name: str, version: str = ""
    ) -> PBDatasetBuild:
        return self._service.GetLatestBuild(
            GetLatestBuildRequest(
                dataset_name=name,
                dataset_version=version,
                project_id=ProjectId(value=str(project_id)),
            ),
        ).build


def _language_version() -> Tuple[int, int, int]:
    return sys.version_info.major, sys.version_info.minor, sys.version_info.micro


def _get_batch_chunks(
    batch: pyarrow.RecordBatch, max_chunk_size_bytes: int = 4_000_000
) -> Generator[pyarrow.RecordBatch, None, None]:
    """
    Slice the batch into chunks, based on average row size,
    but not exceeding the maximum chunk size.
    """

    bytes_per_row = batch.nbytes / batch.num_rows  # average row size
    if bytes_per_row > max_chunk_size_bytes:
        raise ValueError(
            f"the average size of a single row of {int(bytes_per_row)} byte(s) exceeds max chunk size of {max_chunk_size_bytes} byte(s)"
        )
    max_num_rows = int(
        max_chunk_size_bytes / bytes_per_row
    )  # how many rows fit into the chunk
    start = 0
    while start < batch.num_rows:
        i = max_num_rows
        # in case chunk size is exceeded due to large individual rows, slim down until it fits
        while i > 0:
            chunk = batch.slice(start, i)
            if chunk.nbytes <= max_chunk_size_bytes:
                yield chunk
                break
            i -= 1
        if i == 0:
            raise ValueError(
                f"single row in the batch at index {start} exceeds max chunk size of {max_chunk_size_bytes} byte(s)"
            )
        start += i
