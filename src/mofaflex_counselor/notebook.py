from importlib import resources
from io import StringIO
from typing import Annotated, Literal

from jupyter_ai_jupyternaut.jupyternaut.toolkits import notebook, utils
from jupyter_client.asynchronous.client import AsyncKernelClient
from langchain.messages import AIMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

NOTEBOOK_ANALYSIS_SYSTEM_PROMPT = (
    resources.files(__package__) / "prompts/notebook_analyzer_system_prompt.txt"
).read_text()


async def get_active_notebook_code() -> str | None:
    nb_path = await notebook.get_active_notebook()
    if not nb_path:
        return
    file_id = await utils.get_file_id(nb_path)
    if not file_id:
        return
    ydoc = await utils.get_jupyter_ydoc(file_id)
    code = StringIO()
    for cellidx in range(ydoc.cell_number):
        cell = ydoc.get_cell(cellidx)
        if cell.get("cell_type") == "code":
            source = cell["source"]
            code.write(source)
            if len(source) > 0 and source[-1] != "\n":
                code.write("\n")

    return code.getvalue()


async def get_active_notebook_kernel_client() -> AsyncKernelClient:
    nb_path = await notebook.get_active_notebook()
    session_manager = utils.get_serverapp().session_manager
    session = await session_manager.get_session(path=nb_path)
    return session_manager.get_kernel_client(session["kernel"]["id"])


class NotebookAnalysisResult(BaseModel):
    variable_name: Annotated[
        str | None,
        Field(description="The name of the variable holding the most recently modified AnnData or MuData object."),
    ]
    type: Annotated[
        Literal["anndata", "mudata"] | None, Field(description="Whether the object is an AnnData or a MuData object.")
    ]
    path: Annotated[str | None, Field(description="Path that the object was loaded from.")]


async def analyze_active_notebook(model: BaseChatModel) -> NotebookAnalysisResult:
    code = await get_active_notebook_code()
    if code is None:
        return None

    response = await model.with_structured_output(NotebookAnalysisResult).ainvoke(
        [AIMessage(NOTEBOOK_ANALYSIS_SYSTEM_PROMPT), HumanMessage(code)]
    )
    return response


async def analyze_notebook_data(data: NotebookAnalysisResult):
    kclient = await get_active_notebook_kernel_client()
    await kclient.stop_listening()
    ret = await kclient.execute(rf"print({data.variable_name}.obs.columns)", silent=True, reply=True, timeout=1)
    await kclient.start_listening()
    return ret
