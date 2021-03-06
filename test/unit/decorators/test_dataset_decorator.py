import pickle
import uuid
from typing import Any, Callable
from unittest.mock import MagicMock, patch
from uuid import UUID

import pandas as pd
import pytest
from layerapi.api.ids_pb2 import DatasetBuildId
from layerapi.api.service.datacatalog.data_catalog_api_pb2 import InitiateBuildResponse

from layer.clients.data_catalog import DataCatalogClient
from layer.contracts.assets import AssetType
from layer.contracts.datasets import Dataset
from layer.contracts.fabrics import Fabric
from layer.contracts.models import Model
from layer.contracts.project_full_name import ProjectFullName
from layer.decorators import dataset, fabric, pip_requirements
from layer.exceptions.exceptions import (
    ConfigError,
    LayerClientResourceNotFoundException,
    ProjectInitializationException,
)
from layer.global_context import reset_to, set_default_fabric
from test.unit.decorators.util import project_client_mock


def _make_test_dataset_function(name: str) -> Callable[..., Any]:
    @dataset(
        name, dependencies=["datasets/bar", "models/foo", Dataset("baz"), Model("zoo")]
    )
    @pip_requirements(packages=["sklearn==0.0"])
    def func() -> pd.DataFrame:
        return pd.DataFrame()

    return func


class TestDatasetDecorator:
    def test_dataset_decorator_assigns_attributes_to_function_before_calling_function(
        self,
    ) -> None:
        func = _make_test_dataset_function("foo")

        assert func.layer.get_asset_name() == "foo"
        assert func.layer.get_asset_type() == AssetType.DATASET

    def test_dataset_decorator_assigns_attributes_to_function_before_calling_bound_function(
        self,
    ) -> None:
        class MyClass:
            @dataset("foo")
            def create_my_dataset(self) -> pd.DataFrame:
                return pd.DataFrame()

        assert MyClass.create_my_dataset.layer.get_asset_name() == "foo"
        assert MyClass.create_my_dataset.layer.get_asset_type() == AssetType.DATASET

        my_class = MyClass()
        assert my_class.create_my_dataset.layer.get_asset_name() == "foo"
        assert my_class.create_my_dataset.layer.get_asset_type() == AssetType.DATASET

    def test_dataset_decorator_given_no_current_project_set_raise_exception(
        self,
    ) -> None:
        reset_to(None)

        @dataset("foo1")
        def create_my_dataset() -> pd.DataFrame:
            return pd.DataFrame()

        with pytest.raises(
            ProjectInitializationException,
            match="Please specify the current project name globally with",
        ):
            create_my_dataset()

    def test_dataset_decorator_given_set_project_does_not_exist_raise_exception(
        self,
    ) -> None:
        mock_project_api = MagicMock()
        mock_project_api.GetProjectByPath.side_effect = (
            LayerClientResourceNotFoundException()
        )

        with project_client_mock(project_api_stub=mock_project_api):

            @dataset("foo2")
            def create_my_dataset() -> pd.DataFrame:
                return pd.DataFrame()

            with pytest.raises(
                ProjectInitializationException,
                match="Project 'acc-name/foo-test' does not exist.",
            ):
                reset_to("acc-name/foo-test")
                create_my_dataset()

    @pytest.mark.parametrize(("name",), [("foo1",), ("foo2",)])
    def test_dataset_definition_created_correctly(
        self, name: str, test_project_name: str
    ) -> None:

        func = _make_test_dataset_function(name)

        with project_client_mock(), patch(
            "layer.decorators.dataset_decorator._build_locally_update_remotely",
            return_value=("", UUID(int=0x12345678123456781234567812345678)),
        ), patch(
            "layer.decorators.dataset_decorator.register_function",
        ) as mock_register_datasets:

            func()

            _, kwargs = mock_register_datasets.call_args
            dataset = kwargs["func"]

            assert dataset
            assert dataset.asset_name == name
            assert dataset.project_name == test_project_name
            assert [
                (
                    dep.asset_name,
                    dep.asset_type,
                )
                for dep in dataset.asset_dependencies
            ] == [
                ("bar", AssetType.DATASET),
                ("foo", AssetType.MODEL),
                ("baz", AssetType.DATASET),
                ("zoo", AssetType.MODEL),
            ]

            assert dataset.environment_path.exists()
            assert dataset.environment_path.read_text() == "sklearn==0.0"
            assert dataset.pickle_path.exists()
            loaded = pickle.load(open(dataset.pickle_path, "rb"))
            assert loaded.layer.get_asset_name() == name
            assert loaded.layer.get_asset_type() == AssetType.DATASET
            assert loaded.layer.get_pip_packages() == ["sklearn==0.0"]

    def test_should_complete_remote_build_when_failed(self) -> None:
        data_catalog_client = MagicMock(spec=DataCatalogClient)
        data_catalog_client.initiate_build.return_value = InitiateBuildResponse(
            id=DatasetBuildId(value=str(uuid.uuid4()))
        )

        mock_dataset_function = MagicMock()
        mock_dataset_function.project_full_name = ProjectFullName(
            account_name="test-account",
            project_name="project-name",
        )

        with patch(
            "layer.decorators.dataset_decorator.register_function",
            return_value=mock_dataset_function,
        ), project_client_mock(data_catalog_client=data_catalog_client):

            @dataset("foo")
            def create_my_dataset() -> None:
                raise RuntimeError()

            with pytest.raises(RuntimeError):
                create_my_dataset()

            data_catalog_client.initiate_build.assert_called_once()
            data_catalog_client.complete_build.assert_called_once()

    def test_given_fabric_override_uses_it_over_default(self) -> None:
        set_default_fabric(Fabric.F_SMALL)

        with project_client_mock(), patch(
            "layer.decorators.dataset_decorator._build_dataset_locally_and_store_remotely"
        ) as mock_build_locally:

            @dataset("test")
            @fabric(Fabric.F_MEDIUM.value)
            def create_my_dataset() -> pd.DataFrame:
                return pd.DataFrame()

            @dataset("test-2")
            def create_another_dataset() -> pd.DataFrame:
                return pd.DataFrame()

            create_my_dataset()
            create_another_dataset()

            (
                unused_func,
                settings,
                unused_ds,
                unused_tracker,
                unused_client,
            ) = mock_build_locally.call_args_list[0][0]
            assert settings.get_fabric() == Fabric.F_MEDIUM

            (
                unused_func,
                settings,
                unused_ds,
                unused_tracker,
                unused_client,
            ) = mock_build_locally.call_args_list[1][0]
            assert settings.get_fabric() == Fabric.F_SMALL

    def test_not_named_dataset_cannot_be_run_even_locally(self) -> None:
        @dataset("")
        def func() -> None:
            pass

        with pytest.raises(
            ConfigError,
            match="^Your @dataset and @model must be named. Pass an asset name as a first argument to your decorators.$",
        ):
            func()
