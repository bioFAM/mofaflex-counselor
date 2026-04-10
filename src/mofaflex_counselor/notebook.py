import json
import traceback
from collections.abc import Mapping, Sequence
from importlib import resources
from io import StringIO
from typing import Annotated, Literal

from jupyter_ai_jupyternaut.jupyternaut.toolkits import notebook, utils
from jupyter_client.asynchronous.client import AsyncKernelClient
from langchain.messages import AIMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field
from pydantic_core import ValidationError

from .utils import DEBUG

_NOTEBOOK_ANALYSIS_SYSTEM_PROMPT = (
    resources.files(__package__) / "prompts/notebook_analyzer_system_prompt.txt"
).read_text()
_DATA_ANALYSIS_FUNCTION = (resources.files(__package__) / "analyze_data.py").read_text()
_FEATURESETS_FILTER_OPTIMIZE_FUNCTION = (resources.files(__package__) / "analyze_features.py").read_text()
_MOFAFLEX_API_ANALYSIS_FUNCTION = (resources.files(__package__) / "analyze_mofaflex_api.py").read_text()


class _NotebookAnalysisResult(BaseModel):
    data_variable_name: Annotated[
        str | None,
        Field(description="The name of the variable holding the most recently modified AnnData or MuData object."),
    ]
    data_type: Annotated[
        Literal["anndata", "mudata"] | None, Field(description="Whether the object is an AnnData or a MuData object.")
    ]
    data_path: Annotated[str | None, Field(description="Path that the object was loaded from.")]
    featuresets_variable_name: Annotated[
        str | None, Field(description="The name of the variable holding the most recently modified FeatureSets object.")
    ]


class _LayerProperties(BaseModel):
    name: str
    nonnegative: bool


class DataAnalysisResult(BaseModel):
    data_var_name: str
    type: Literal["MuData", "AnnData"]
    n_obs: int
    n_views: int
    n_vars: int | Mapping[str, int]
    X_nonnegative: bool
    layers: Sequence[_LayerProperties]
    grouping_cols: Sequence[str]
    covariates_obs_cols: Sequence[str]
    covariates_obsm_keys: Sequence[str]
    annotations_varm_keys: Sequence[str]
    featuresets_var_name: str | None


class FeatureSetsOptimizationResult(BaseModel):
    min_fraction: float
    min_count: int
    max_count: int


async def _get_active_notebook_code(nb_path: str) -> str | None:
    file_id = await utils.get_file_id(nb_path)
    if not file_id:
        return
    ydoc = await utils.get_jupyter_ydoc(file_id)
    active_cell_id = notebook._get_active_cell_id_from_ydoc(ydoc)
    code = StringIO()
    try:
        for cellidx in range(ydoc.cell_number):
            cell = ydoc.get_cell(cellidx)
            if cell.get("cell_type") == "code":
                source = cell["source"]
                code.write(source)
                if len(source) > 0 and source[-1] != "\n":
                    code.write("\n")
            if cell["id"] == active_cell_id:
                break

        return code.getvalue()
    except Exception as e:  # noqa: BLE001
        if DEBUG:
            traceback.print_exception(e)
        return None


async def _get_active_notebook_kernel_client(nb_path: str) -> AsyncKernelClient:
    session_manager = utils.get_serverapp().session_manager
    session = await session_manager.get_session(path=nb_path)
    return session_manager.get_kernel_client(session["kernel"]["id"])


async def _analyze_active_notebook_code(model: BaseChatModel, nb_path: str) -> _NotebookAnalysisResult:
    code = await _get_active_notebook_code(nb_path)
    if code is None:
        return None

    response = await model.with_structured_output(_NotebookAnalysisResult).ainvoke(
        [AIMessage(_NOTEBOOK_ANALYSIS_SYSTEM_PROMPT), HumanMessage(code)]
    )
    if DEBUG:
        print(response.model_dump_json())
    return response


async def _run_in_active_kernel(nb_path: str, setup: str, execute: str, teardown: str) -> str | None:
    kclient = await _get_active_notebook_kernel_client(nb_path)
    await kclient.stop_listening()
    await kclient.execute(setup, silent=True, reply=True)
    execute_result = await kclient.execute(execute, store_history=False, reply=True)
    msg_id = execute_result["parent_header"]["msg_id"]
    if execute_result["content"]["status"] != "ok":
        ret = None
        if DEBUG:
            print(execute_result)
    else:
        while True:
            result = await kclient.get_iopub_msg()
            if result["parent_header"].get("msg_id") == msg_id and result["msg_type"] == "stream":
                break
        if result["content"]["name"] != "stdout":
            ret = None
        else:
            ret = result["content"]["text"]
    kclient.execute(teardown, silent=True)
    await kclient.start_listening()
    return ret


async def _analyze_notebook_data(data: _NotebookAnalysisResult, nb_path: str) -> DataAnalysisResult | None:
    ret = None
    try:
        result = await _run_in_active_kernel(
            nb_path,
            _DATA_ANALYSIS_FUNCTION,
            f"print(____analyze_data({data.data_variable_name!r}, {data.data_type!r}, {data.data_path!r}, {data.featuresets_variable_name!r}))",
            "del ____analyze_data",
        )
    except Exception as e:  # noqa: BLE001
        if DEBUG:
            print(e)
        return None
    if result is not None:
        try:
            ret = DataAnalysisResult.model_validate_json(result)
        except ValidationError:
            ret = None
        except Exception as e:  # noqa: BLE001
            if DEBUG:
                print(e)
    return ret


async def analyze_active_notebook(model: BaseChatModel) -> DataAnalysisResult | None:
    nb_path = await notebook.get_active_notebook()
    if not nb_path:
        yield None
        return
    yield "Analyzing active notebook...\n\n"
    res = await _analyze_active_notebook_code(model, nb_path)
    res = await _analyze_notebook_data(res, nb_path)
    if res is None:
        yield "Notebook analysis failed, falling back to defaults...\n\n"
    yield res
    return


async def analyze_mofaflex_api() -> str | None:
    nb_path = await notebook.get_active_notebook()
    result = await _run_in_active_kernel(
        nb_path, _MOFAFLEX_API_ANALYSIS_FUNCTION, "print(____analyze_mofaflex_api())", "del ____analyze_mofaflex_api"
    )
    return json.loads(result) if result is not None else None


async def optimize_featuresets_filtering(
    data: DataAnalysisResult | None,
) -> FeatureSetsOptimizationResult | Literal[False] | None:
    if data is None or data.featuresets_var_name is None:
        return None
    nb_path = await notebook.get_active_notebook()
    if not nb_path:
        return None
    try:
        result = await _run_in_active_kernel(
            nb_path,
            _FEATURESETS_FILTER_OPTIMIZE_FUNCTION,
            f"print(____optimize_features_filtering({data.data_var_name!r}, {data.featuresets_var_name!r}))",
            "del ____optimize_features_filtering",
        )
    except Exception as e:  # noqa: BLE001
        if DEBUG:
            print(e)
        return None
    ret = None
    if result is not None:
        if result != "null":
            try:
                ret = FeatureSetsOptimizationResult.model_validate_json(result)
            except ValidationError:
                ret = None
            except Exception as e:  # noqa: BLE001
                if DEBUG:
                    print(e)
        else:
            ret = False
    return ret
