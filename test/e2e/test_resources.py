import filecmp
import os.path
import tempfile
import uuid

import pandas as pd

import layer
from layer.contracts.projects import Project
from layer.contracts.runs import ModelFunctionDefinition, Run
from layer.decorators import dataset, model, pip_requirements, resources
from layer.resource_manager import ResourceManager
from layer.tracker.progress_tracker import RunProgressTracker
from test.e2e.assertion_utils import E2ETestAsserter


def test_resource_manager(initialized_project: Project, asserter: E2ETestAsserter):
    @model("model")
    @resources("test/e2e/assets/data/test.csv")
    def func() -> None:
        return None

    definition = ModelFunctionDefinition(func, initialized_project.name)
    run = Run(uuid.uuid4(), initialized_project.name).with_definitions([definition])
    resource_manager = ResourceManager(asserter.client)

    resource_manager.wait_resource_upload(run, RunProgressTracker())
    with tempfile.TemporaryDirectory(prefix="test_resource_manager") as resource_dir:
        resource_manager.wait_resource_download(
            definition.project_name, definition.func_name, target_dir=resource_dir
        )
        assert filecmp.cmp(
            "test/e2e/assets/data/test.csv",
            os.path.join(resource_dir, "test", "e2e", "assets", "data", "test.csv"),
        )


def test_resources_simple(initialized_project: Project, asserter: E2ETestAsserter):
    @dataset("test_resources")
    @resources("test/e2e/assets/data")
    def resources_func():
        return pd.read_csv("test/e2e/assets/data/test.csv")

    run = layer.run([resources_func])
    asserter.assert_run_succeeded(run.run_id)


def test_titanic_resources(initialized_project: Project, asserter: E2ETestAsserter):
    @dataset("titanic")
    @resources("test/e2e/assets/titanic/titanic.csv")
    def titanic():
        return pd.read_csv("test/e2e/assets/titanic/titanic.csv")

    run = layer.run([titanic])
    asserter.assert_run_succeeded(run.run_id)


def test_model_resources(initialized_project: Project, asserter: E2ETestAsserter):
    @model("foo-model")
    @pip_requirements(packages=["scikit-learn==0.23.2"])
    @resources("test/e2e/assets/titanic/titanic.csv")
    def train_model():
        from sklearn.svm import SVC

        svc = SVC(kernel="linear")
        pd.read_csv("test/e2e/assets/titanic/titanic.csv")
        return svc.fit([[1], [2], [3]], [[1], [2], [3]])

    run = layer.run([train_model])
    asserter.assert_run_succeeded(run.run_id)


def test_local_model_train_with_resources(
    initialized_project: Project, asserter: E2ETestAsserter
):
    model_name = "foo-model-local-built-with-resources"

    @model(model_name)
    @pip_requirements(packages=["scikit-learn==0.23.2"])
    @resources("test/e2e/assets/titanic/titanic.csv")
    def train_model():
        from sklearn.svm import SVC

        svc = SVC(kernel="linear")
        pd.read_csv("test/e2e/assets/titanic/titanic.csv")
        return svc.fit([[1], [2], [3]], [[1], [2], [3]])

    train_model()

    retrieved_model = layer.get_model(f"{initialized_project.name}/models/{model_name}")
    assert retrieved_model.name == model_name


def test_no_resource_decorator(initialized_project: Project, asserter: E2ETestAsserter):
    @dataset("titanic_no_resources")
    def titanic_no_resources():
        return pd.DataFrame(data={"col_a": ["a"], "col_b": ["b"]})

    run = layer.run([titanic_no_resources])
    asserter.assert_run_succeeded(run.run_id)


def test_add_remove_resource_decorator(
    initialized_project: Project, asserter: E2ETestAsserter
):
    resource_file = "test/e2e/assets/titanic/titanic.csv"

    def first_run():
        @dataset("titanic")
        @resources(resource_file)
        def titanic():
            return pd.read_csv(resource_file)

        run = layer.run([titanic])
        asserter.assert_run_succeeded(run.run_id)

    def second_run():  # with resources removed
        @dataset("titanic")
        def titanic():
            if os.path.exists(resource_file):
                raise AssertionError(f"File ${resource_file} should not exist anymore")
            return pd.DataFrame(data={"col_a": ["a"], "col_b": ["b"]})

        run = layer.run([titanic])
        asserter.assert_run_succeeded(run.run_id)

    first_run()
    second_run()


def test_resource_path_contains_spaces(
    initialized_project: Project, asserter: E2ETestAsserter
):
    resource_path = "test/e2e/assets/titanic/ti ta nic.csv"

    @dataset("titanic-with-spaces-in-the-resource-path")
    @resources(resource_path)
    def titanic():
        return pd.read_csv(resource_path)

    run = layer.run([titanic])
    asserter.assert_run_succeeded(run.run_id)
