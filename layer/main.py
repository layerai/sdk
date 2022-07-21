import json
import logging
import pathlib
import re
import urllib.request
import warnings
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Union,
)
from uuid import UUID

from yarl import URL

from layer import Image, Markdown
from layer.cache.cache import Cache
from layer.cache.utils import is_cached
from layer.clients.layer import LayerClient
from layer.config import DEFAULT_PATH, DEFAULT_URL, ConfigManager
from layer.config.config import Config
from layer.context import Context
from layer.contracts.assets import AssetPath, AssetType
from layer.contracts.datasets import Dataset
from layer.contracts.fabrics import Fabric
from layer.contracts.models import Model
from layer.contracts.project_full_name import ProjectFullName
from layer.contracts.projects import Project
from layer.contracts.runs import Run
from layer.contracts.tracker import ResourceTransferState
from layer.exceptions.exceptions import (
    ConfigError,
    ProjectInitializationException,
    UserConfigurationError,
    UserNotLoggedInException,
    UserWithoutAccountError,
)
from layer.global_context import (
    current_account_name,
    get_active_context,
    has_shown_python_version_message,
    has_shown_update_message,
    reset_active_context,
    reset_to,
    set_active_context,
    set_has_shown_python_version_message,
    set_has_shown_update_message,
)
from layer.logged_data.log_data_runner import LogDataRunner
from layer.projects.init_project_runner import InitProjectRunner
from layer.projects.project_runner import ProjectRunner
from layer.projects.utils import get_current_project_full_name, validate_project_name
from layer.settings import LayerSettings
from layer.tracker.utils import get_progress_tracker
from layer.training.train import Train
from layer.utils.async_utils import asyncio_run_in_thread


if TYPE_CHECKING:
    import matplotlib.axes._subplots  # type: ignore
    import matplotlib.figure  # type: ignore
    import pandas
    import PIL.Image


logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    "`should_run_async` will not call `transform_cell` automatically in the future.",
    DeprecationWarning,
)


def login(url: Union[URL, str] = DEFAULT_URL) -> None:
    """
    :param url: Not used.
    :raise UserNotLoggedInException: If the user is not logged in.
    :raise UserConfigurationError: If Layer user is not configured correctly.

    Logs user in to Layer. You might be prompted to enter an access token from a web page that is provided.

    .. code-block:: python

        layer.login()

    """
    _check_latest_version()

    async def _login(manager: ConfigManager, login_url: URL) -> None:
        await manager.login_headless(login_url)

    _refresh_and_login_if_needed(url, _login)


def login_with_access_token(
    access_token: str, url: Union[URL, str] = DEFAULT_URL
) -> None:
    """
    :param access_token: A valid access token retrieved from Layer.
    :param url: Not used.
    :raise UserNotLoggedInException: If the user is not logged in.
    :raise UserConfigurationError: If Layer user is not configured correctly.

    Log in with an access token. You might be prompted to enter an access token from a web page that is provided.

    .. code-block:: python

        layer.login_with_access_token(TOKEN)
    """

    async def _login(manager: ConfigManager, login_url: URL) -> None:
        await manager.login_with_access_token(login_url, access_token)

    _refresh_and_login_if_needed(url, _login)


def _refresh_and_login_if_needed(
    url: Union[URL, str], login_func: Callable[[ConfigManager, URL], Awaitable[None]]
) -> None:
    async def _login(login_url: URL) -> None:
        manager = ConfigManager(DEFAULT_PATH)
        try:
            config = await manager.refresh()
            if not config.is_guest and config.url == login_url:
                # no need to re-login
                return
        except (UserNotLoggedInException, UserConfigurationError):
            pass

        try:
            await login_func(manager, login_url)
        except UserWithoutAccountError as ex:
            raise SystemExit(ex)

    if isinstance(url, str):
        url = URL(url)

    asyncio_run_in_thread(_login(url))


def login_as_guest(url: Union[URL, str] = DEFAULT_URL) -> None:
    """
    :param url: (optional) The target platform where the user wants to log in.

    Logs user in to Layer as a guest user. Guest users can view public projects, but they cannot add, edit, or delete entities.

    .. code-block:: python

        layer.login_as_guest() # Uses default target platform URL

    """

    async def _login_as_guest(url: URL) -> None:
        manager = ConfigManager(DEFAULT_PATH)
        await manager.login_as_guest(url)

    if isinstance(url, str):
        url = URL(url)

    asyncio_run_in_thread(_login_as_guest(url))


def login_with_api_key(api_key: str, url: Union[URL, str] = DEFAULT_URL) -> None:
    """
    :param access_token: A valid API key retrieved from Layer UI.
    :param url: Not used.
    :raise AuthException: If the API key is invalid.
    :raise UserConfigurationError: If Layer user is not configured correctly.

    Log in with an API key.

    .. code-block:: python

        layer.login_with_api_key(API_KEY)
    """

    async def _login(url: URL) -> None:
        manager = ConfigManager(DEFAULT_PATH)
        await manager.login_with_api_key(url, api_key)

    if isinstance(url, str):
        url = URL(url)

    asyncio_run_in_thread(_login(url))


def logout() -> None:
    """
    Log out of Layer.

    .. code-block:: python

        layer.logout()

    """
    asyncio_run_in_thread(ConfigManager(DEFAULT_PATH).logout())


def show_api_key() -> None:
    config = ConfigManager(DEFAULT_PATH).load()
    print(config.credentials.refresh_token)


def get_dataset(name: str, no_cache: bool = False) -> Dataset:
    """
    :param name: Name or path of the dataset.
    :param no_cache: if True, force dataset fetch from the remote location.
    :return: A dataset object defined in a Layer project.

    Retrieves a Layer dataset object from the Discover > Datasets tab.

    Guest users can use this function to access public datasets without logging in to Layer.

    By default, this function caches dataset contents locally when possible.

    .. code-block:: python

        # The simplest application of this method just returns a list of dataset information.
        layer.get_dataset("titanic")

        # This example gets the titanic dataset, turns it into a Pandas dataframe, and then displays it.
        dataset = layer.get_dataset("titanic")
        df = dataset.to_pandas()
        df.head()

        # You can also get datasets from other projects you have access to, including public projects.
        layer.get_dataset("the-project/datasets/titanic")
    """
    config: Config = asyncio_run_in_thread(ConfigManager().refresh(allow_guest=True))
    asset_path = AssetPath.parse(name, expected_asset_type=AssetType.DATASET)
    asset_path = _ensure_asset_path_is_absolute(asset_path)

    def fetch_dataset() -> "pandas.DataFrame":
        context = get_active_context()
        with LayerClient(config.client, logger).init() as client:
            within_run = (
                True if context else False
            )  # if layer.get_dataset is called within an @dataset decorated func or not
            callback = lambda: client.data_catalog.fetch_dataset(  # noqa: E731
                asset_path, no_cache=no_cache
            )
            if not within_run:
                try:
                    with Context() as context:
                        set_active_context(context)
                        context.with_asset_name(asset_path.asset_name)
                        context.with_asset_type(AssetType.DATASET)
                        tracker = get_progress_tracker(
                            url=config.url,
                            account_name=asset_path.must_org_name(),
                            project_name=asset_path.must_project_name(),
                        )
                        context.with_tracker(tracker)
                        with tracker.track():
                            dataset = _ui_progress_with_tracker(
                                callback,
                                asset_path.asset_name,
                                False,  # Datasets are fetched per partition, no good way to show caching per partition
                                within_run,
                                context,
                                AssetType.DATASET,
                            )
                finally:
                    reset_active_context()  # Reset only if outside layer func, as the layer func logic will reset it
            else:
                assert context
                dataset = _ui_progress_with_tracker(
                    callback,
                    asset_path.asset_name,
                    False,  # Datasets are fetched per partition, no good way to show caching per partition
                    within_run,
                    context,
                    AssetType.DATASET,
                )

            return dataset

    return Dataset(
        asset_path=asset_path,
        _pandas_df_factory=fetch_dataset,
    )


def get_model(name: str, no_cache: bool = False) -> Model:
    """
    :param name: Name or path of the model. You can pass additional parameters in the name to retrieve a specific version of the model with format: ``model_name:major_version.minor_version``
    :param no_cache: if True, force model fetch from the remote location.
    :return: The model object.

    Retrieves a Layer model object by its name.

    Guest users can use this function to access public models without logging in to Layer.

    By default, this function caches models locally when possible.

    .. code-block:: python

        # Loads the default version of the model.
        layer.get_model("churn_model")

        # Loads the latest train of version 2.
        layer.get_model("churn_model:2")

        # Loads a specific train of the model version 2.
        layer.get_model("churn_model:2.12")

        # Load a model from a project you aren't logged in to.
        layer.get_model("the-project/models/churn_model")
    """
    config = asyncio_run_in_thread(ConfigManager().refresh(allow_guest=True))
    asset_path = AssetPath.parse(name, expected_asset_type=AssetType.MODEL)
    asset_path = _ensure_asset_path_is_absolute(asset_path)
    context = get_active_context()

    with LayerClient(config.client, logger).init() as client:
        within_run = (
            True if context else False
        )  # if layer.get_model is called within a @model decorated func of not
        model = client.model_catalog.load_model_by_path(path=asset_path.path())
        from_cache = not no_cache and is_cached(model)
        state = ResourceTransferState(model.name)

        def callback() -> Model:
            return _load_model_runtime_objects(client, model, state, no_cache)

        if not within_run:
            try:
                with Context() as context:
                    set_active_context(context)
                    context.with_asset_name(asset_path.asset_name)
                    context.with_asset_type(AssetType.MODEL)
                    tracker = get_progress_tracker(
                        url=config.url,
                        account_name=asset_path.must_org_name(),
                        project_name=asset_path.must_project_name(),
                    )
                    context.with_tracker(tracker)
                    with tracker.track():
                        model = _ui_progress_with_tracker(
                            callback,
                            asset_path.asset_name,
                            from_cache,
                            within_run,
                            context,
                            AssetType.MODEL,
                            state,
                        )
            finally:
                reset_active_context()  # Reset only if outside layer func, as the layer func logic will reset it
        else:
            assert context
            model = _ui_progress_with_tracker(
                callback,
                asset_path.asset_name,
                from_cache,
                within_run,
                context,
                AssetType.MODEL,
                state,
            )

        return model


def _load_model_runtime_objects(
    client: LayerClient,
    model: Model,
    state: ResourceTransferState,
    no_cache: bool,
) -> Model:
    model_runtime_objects = client.model_catalog.load_model_runtime_objects(
        model,
        state=state,
        no_cache=no_cache,
    )
    model.set_model_runtime_objects(model_runtime_objects)
    return model


def _ui_progress_with_tracker(
    callback: Callable[[], Any],
    getting_asset_name: str,
    from_cache: bool,
    within_run: bool,
    context: Context,
    getting_asset_type: AssetType,
    state: Optional[ResourceTransferState] = None,
) -> Any:
    asset_name = context.asset_name()
    assert asset_name
    tracker = context.tracker()
    assert tracker
    asset_type = context.asset_type()
    assert asset_type
    if asset_type == AssetType.MODEL:
        if getting_asset_type == AssetType.DATASET:
            tracker.mark_model_getting_dataset(
                asset_name, getting_asset_name, from_cache
            )
        elif getting_asset_type == AssetType.MODEL:
            tracker.mark_model_getting_model(
                asset_name, getting_asset_name, state, from_cache
            )
    elif asset_type == AssetType.DATASET:
        if getting_asset_type == AssetType.DATASET:
            tracker.mark_dataset_getting_dataset(
                asset_name, getting_asset_name, from_cache
            )
        elif getting_asset_type == AssetType.MODEL:
            tracker.mark_dataset_getting_model(
                asset_name, getting_asset_name, state, from_cache
            )
    result = callback()
    if within_run:
        if asset_type == AssetType.MODEL:
            tracker.mark_model_training(asset_name)
        elif asset_type == AssetType.DATASET:
            tracker.mark_dataset_building(asset_name)
    elif asset_type == AssetType.MODEL:
        tracker.mark_model_loaded(asset_name)
    elif asset_type == AssetType.DATASET:
        tracker.mark_dataset_loaded(asset_name)
    return result


def _ensure_asset_path_is_absolute(
    path: AssetPath,
) -> AssetPath:
    if not path.is_relative():
        return path
    project_name = (
        path.project_name
        if path.has_project()
        else get_current_project_full_name().project_name
    )
    account_name = (
        path.org_name if path.org_name is not None else current_account_name()
    )

    if not project_name or not account_name:
        raise ProjectInitializationException(
            "Please specify the project full name globally with layer.init('account-name/project-name')"
            "or have it in the asset full name like 'the-account/the-project/models/the-model-name'"
        )

    path = path.with_project_full_name(
        ProjectFullName(account_name=account_name, project_name=project_name)
    )
    return path


@contextmanager
def start_train(
    name: str, version: str, train_id: Optional[UUID] = None
) -> Iterator[Train]:
    """
    :param name: name of the model
    :param version: version number of the model
    :param train_id: id to start the training with

    Initiates a model training, generating and retaining model metadata on the backend

    Example usage:
    import layer

    with layer.start_train(name="model_name", version=2) as train:
        train.register_input(x_train)
        train.register_output(y_train)
        trained_model = .....
        train.save_model(trained_model)

    """
    project_full_name = get_current_project_full_name()
    config = asyncio_run_in_thread(ConfigManager().refresh())
    with LayerClient(config.client, logger).init() as client:
        train = Train(
            layer_client=client,
            name=name,
            version=version,
            train_id=train_id,
            project_full_name=project_full_name,
        )
        with train:
            yield train


def init(
    project_name: str,
    fabric: Optional[str] = None,
    pip_packages: Optional[List[str]] = None,
    pip_requirements_file: Optional[str] = None,
) -> Project:
    """
    :param project_name: Name of the project to be initialized.
    :param fabric: Default fabric to use for current project when running code in Layer's backend.
    :param pip_packages: List of packages to install in Layer backend when running code for this project.
    :param pip_requirements_file: File name with list of packages to install in Layer backend when running code for this project.
    :return: Project object.

    Initializes a project with given name both locally and remotely.

    If the project does not exist in the Layer backend, then it will be automatically created.
    Additional settings can be passed in as parameters: namely fabric and packages to install.
    These settings are only applied to runs initiated from the current local environment.

    Other users can set different settings for the same project in their own local environments
    via ``layer.init``, and those will not impact your runs.

    Note that only one of ``pip_packages`` or ``pip_requirements_file`` should be provided.

    .. code-block:: python

        # Initialize new project and create it in Layer backend if it does not exist
        project = layer.init("my_project_name", fabric="x-small")
    """
    _check_latest_version()

    validate_project_name(project_name)

    if pip_packages and pip_requirements_file:
        raise ValueError(
            "either pip_requirements_file or pip_packages should be provided, not both"
        )
    layer_config = asyncio_run_in_thread(ConfigManager().refresh())

    project_full_name = _get_project_full_name(layer_config, project_name)

    reset_to(project_full_name)

    init_project_runner = InitProjectRunner(project_full_name, logger=logger)
    fabric_to_set = Fabric(fabric) if fabric else None
    return init_project_runner.setup_project(
        fabric=fabric_to_set,
        pip_packages=pip_packages,
        pip_requirements_file=pip_requirements_file,
    )


def run(functions: List[Any], debug: bool = False) -> Run:
    """
    :param functions: List of decorated functions to run in the Layer backend.
    :param debug: Stream logs to console from infra executing the project remotely.
    :return: An object with information on the run triggered, unless there is an initialization error.

    Runs the specified functions on Layer infrastructure.

    Running the project does the following:
    - Adds new versions of decorated functions to the Layer web UI.
    - Runs any logs and stores them.

    .. code-block:: python

        import pandas as pd
        from layer
        from layer.decorators import dataset

        # Define a new function for dataset generation
        @dataset("my_dataset_name")
        def create_my_dataset():
            data = [[1, "product1", 15], [2, "product2", 20], [3, "product3", 10]]
            dataframe = pd.DataFrame(data, columns=["Id", "Product", "Price"])
            return dataframe

        # Initialize current project locally and remotely (if it doesn't exist)
        project = layer.init("my_project_name", fabric="x-small")

        # Run the dataset generation code using Layer's backend compute resources
        run = layer.run([create_my_dataset])
        # run = layer.run([create_my_dataset], debug=True)  # Stream logs to console
    """
    _check_python_version()
    _ensure_all_functions_are_decorated(functions)

    layer_config: Config = asyncio_run_in_thread(ConfigManager().refresh())

    project_full_name = get_current_project_full_name()
    project_runner = ProjectRunner(
        config=layer_config,
        project_full_name=project_full_name,
        functions=functions,
    )
    run = project_runner.run(debug=debug)

    _make_notebook_links_open_in_new_tab()

    return run


def _get_project_full_name(
    layer_config: Config, user_input_project_name: str
) -> ProjectFullName:
    """
    Will first try to extract account_name/project_name from :user_input_project_name

    If no account name can be extracted, will try to get it from global context.

    If that too fails, will fetch the personal account.
    """
    parts = user_input_project_name.split("/")
    account_name: Optional[str]
    project_name: str
    if len(parts) == 1:
        project_name = user_input_project_name
        with LayerClient(layer_config.client, logger).init() as client:
            account_name = client.account.get_my_account().name
    else:
        account_name = parts[0]
        project_name = parts[1]

    return ProjectFullName(
        project_name=project_name,
        account_name=account_name,
    )


# Normally, Colab/IPython opens links as an IFrame. One can open them as new tabs through the right-click menu or using shift+click.
# However, we would like to change the default behavior to always open on a new page.
def _make_notebook_links_open_in_new_tab() -> None:
    def is_notebook() -> bool:
        try:
            shell = get_ipython().__class__.__name__  # type: ignore
            if shell == "ZMQInteractiveShell":
                return True  # Jupyter notebook or qtconsole
            elif shell == "TerminalInteractiveShell":
                return False  # Terminal running IPython
            else:
                return False  # Other type (?)
        except NameError:
            return False  # Probably standard Python interpreter

    if is_notebook():
        from IPython.display import Javascript, display  # type: ignore

        # Colab
        display(
            Javascript(
                """
document.querySelectorAll("#output-area a").forEach(function(a){
    a.setAttribute('target', '_blank');
})
        """
            )
        )

        # Jupyter
        display(
            Javascript(
                """
document.querySelectorAll(".output a").forEach(function(a){
    a.setAttribute('target', '_blank');
})
        """
            )
        )


def _get_latest_version() -> str:
    pypi_url = "https://pypi.org/pypi/layer/json"
    response = urllib.request.urlopen(pypi_url).read().decode()  # nosec urllib_urlopen
    data = json.loads(response)

    return data["info"]["version"]


def _check_latest_version() -> None:
    if has_shown_update_message():
        return

    latest_version = _get_latest_version()
    current_version = get_version()
    if current_version != latest_version:
        print(
            f"You are using the version {current_version} but the latest version is {latest_version}, please upgrade with 'pip install --upgrade layer'"
        )
    set_has_shown_update_message(True)


def _check_python_version() -> None:
    if has_shown_python_version_message():
        return

    import platform

    major, minor, _ = platform.python_version_tuple()

    if major != "3" or minor not in ["7", "8"]:
        print(
            f"You are using the Python version {platform.python_version()} but layer requires Python 3.7.x or 3.8.x"
        )
    set_has_shown_python_version_message(True)


def _ensure_all_functions_are_decorated(functions: List[Callable[..., Any]]) -> None:
    for f in functions:
        if not hasattr(f, "layer"):
            raise ConfigError(
                'Either @dataset(name="...") or @model(name="...") is required for each function. '
                "Add @dataset or @model decorator to your functions to run them in Layer."
            )
        settings: LayerSettings = f.layer  # type: ignore
        settings.validate()


def log(
    data: Dict[
        str,
        Union[
            str,
            float,
            bool,
            int,
            Dict[str, Any],
            "pandas.DataFrame",
            "PIL.Image.Image",
            "matplotlib.figure.Figure",
            "matplotlib.axes._subplots.AxesSubplot",
            Image,
            ModuleType,
            Path,
            Markdown,
        ],
    ],
    step: Optional[int] = None,
) -> None:
    """
    :param data: A dictionary in which each key is a string tag (i.e. name/id). The value can have different types. See examples below for more details.
    :param step: An optional non-negative integer that associates data with a particular step (epoch). This only takes effect if the logged data is to be associated with a model train (and *not* with a dataset build), and the data is either a number or an image.
    :return: None

    Logs arbitrary data associated with a model train or a dataset build into Layer backend.

    This function can only be run inside functions decorated with ``@model`` or ``@dataset``. The logged data can then be discovered and analyzed through the Layer UI.

    We support Python primitives, images, tables and plots to enable better experiment tracking and version comparison.

    **Python Primitives**

    You can log Python primitive types for your model train parameters, metrics or KPIs

    Accepted Types:
    ``str``,
    ``float``,
    ``bool``,
    ``int``

    **Markdown**

    You can put markdown syntax and it will be rendered in the web UI accordingly.

    Accepted Types:
    ``layer.Markdown``

    **Images**

    You can log images to track inputs, outputs, detections, activations and more. We support GIF, JPEG, PNG formats.

    Accepted Types:
    ``PIL.Image.Image``,
    ``path.Path``
    ``layer.Image``

    **Videos**

    We support MP4, WebM, Ogg file formats or ``pytorch.Tensor`` (with NTCHW shape) through ``layer.Video``.

    Accepted Types:
    ``path.Path``
    ``layer.Video``

    **Charts**

    You can track your metrics in detail with charts

    Accepted Types:
    ``matplotlib.figure.Figure``,
    ``matplotlib.pyplot``,
    ``matplotlib.axes._subplots.AxesSubplot``,
    ``ModuleType`` (only for the matplotlib module, for convenience)

    **Tables**

    You can log dataframes to display and analyze your tabular data.

    Accepted Types:
    ``pandas.DataFrame``
    ``dict`` (the key should be a string. The value either needs to be a primitive type or it will be converted to str)

    .. code-block:: python

        import layer
        import matplotlib.pyplot as plt
        import pandas as pd
        from layer.decorators import dataset, model

        # Define a new function for dataset generation
        @dataset("my_dataset_name")
        def create_my_dataset():
            data = [[1, "product1", 15], [2, "product2", 20], [3, "product3", 10]]
            dataframe = pd.DataFrame(data, columns=["Id", "Product", "Price"])

            t = np.arange(0.0, 2.0, 0.01)
            s = 1 + np.sin(2 * np.pi * t)
            fig, ax = plt.subplots()

            layer.log({
                "my-str-tag": "any text",
                "my-int-tag": 123, # any number
                "foo-bool": True,
                "some-sample-dataframe-tag": ..., # Pandas data frame
                "some-local-image-file": Path.home() / "images/foo.png",
                "some-matplot-lib-figure": fig, # You could alternatively just passed plt as well, and Layer would just get the current/active figure.
            })

            layer.log({
                "another-tag": "you can call layer.log multiple times"
            })

            return dataframe

        @model("my_model")
        def create_my_model():
            # everything that can be logged for a dataset (as shown above), can also be logged for a model too.

            # In addition to those, if you have a multi-step (i.e. multi-epoch) algorithm, you can associate a metric with the step.
            # These will be rendered on a graph inside the Layer UI.
            for i in range(1000):
                some_result, accuracy = some_algo(some_result)
                layer.log({
                    "Model accuracy": accuracy,
                }, step=i)

        create_my_dataset()
        create_my_model()
    """
    active_context = get_active_context()
    if not active_context:
        raise RuntimeError(
            "Data logging only allowed inside functions decorated with @model or @dataset"
        )
    train = active_context.train()
    train_id = train.get_id() if train is not None else None
    dataset_build = active_context.dataset_build()
    dataset_build_id = dataset_build.id if dataset_build is not None else None
    layer_config = asyncio_run_in_thread(ConfigManager().refresh())
    with LayerClient(layer_config.client, logger).init() as client:
        log_data_runner = LogDataRunner(
            client=client,
            train_id=train_id,
            dataset_build_id=dataset_build_id,
            logger=logger,
        )
        log_data_runner.log(data=data, epoch=step)


def clear_cache() -> None:
    """
    Clear all cached objects fetched by the Layer SDK on this machine.
    The Layer SDK locally stores all datasets and models by default on your computer.
    When you fetch a dataset with :func:``layer.get_dataset``, or load the model with ``layer.get_model``,
    the first call will fetch the artifact from remote storage,
    but subsequent calls will read it from the local disk.
    """
    Cache().clear()


def get_version() -> str:
    with open(pathlib.Path(__file__).parent.parent / "pyproject.toml") as pyproject:
        text = pyproject.read()
        # Use a simple regex to avoid a dependency on toml
        version_match = re.search(r'version = "(\d+\.\d+\.\d+)"', text)

    if version_match is None:
        raise RuntimeError("Failed to parse version")
    return version_match.group(1)
