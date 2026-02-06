import os
from collections.abc import Sequence
from importlib import resources
from typing import Annotated, Literal

import aiosqlite
from jupyter_ai_jupyternaut.jupyternaut.jupyternaut import JupyternautPersona
from jupyter_ai_persona_manager import BasePersona, PersonaDefaults
from jupyter_core.paths import jupyter_data_dir
from jupyterlab_chat.models import Message
from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_litellm import ChatLiteLLM
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, Field, PositiveInt

MEMORY_STORE_PATH = os.path.join(jupyter_data_dir(), "jupyter_ai", "memory.sqlite")
SYSTEM_PROMPT = resources.files(__package__).joinpath("prompts/chat_system_prompt.txt").read_text()


class FinalizeConfigurationSchema(BaseModel):
    n_factors: Annotated[
        PositiveInt,
        Field(
            default=10,
            description="Number of latent factors. More factors increase training time and memory requirements. Downstream analysis typically focuses on the 5-10 most important factors. Highly complex datasets may require a large number of factors.",
        ),
    ]
    layer: Annotated[
        str | None,
        Field(
            default=None,
            description="Name of the layer in the input file to use. If omitted the X matrix will be used.",
        ),
    ]
    group_by: Annotated[
        str | Sequence[str] | None,
        Field(
            default=None,
            description="Columns of .obs in the MuData to group the data by. Only useful if the input is a MuData file. Ignored for AnnData files.",
        ),
    ]
    factor_prior: Annotated[
        Literal["Normal", "Laplace", "Horseshoe", "SnS"],
        Field(
            default="Normal",
            description="Prior distribution to use for the factors. Laplace, Horseshoe and SnS are sparsity-inducing distributions that can yield better interpretability of the results. SnS typically yields the most sparse results. The horseshoe does not achieve exact zeros, but it also does not shrink values that are far from zero. The Laplace shrinks all values.",
        ),
    ]
    weight_prior: Annotated[
        Literal["Normal", "Laplace", "Horseshoe", "SnS"],
        Field(
            default="Normal",
            description="Prior distribution to use for the weights. Laplace, Horseshoe and SnS are sparsity-inducing distributions that can yield better interpretability of the results. SnS typically yields the most sparse results. The horseshoe does not achieve exact zeros, but it also does not shrink values that are far from zero. The Laplace shrinks all values.",
        ),
    ]
    nonnegative_factors: Annotated[
        bool,
        Field(
            default=False,
            description="Constrain the factors to be nonnegative. This may improve interpretability in some cases.",
        ),
    ]
    nonnegative_weights: Annotated[
        bool,
        Field(
            default=False,
            description="Constrain the weights to be nonnegative. This may improve interpretability in some cases.",
        ),
    ]


class MofaFlexCounselor(BasePersona):
    @property
    def defaults(self):
        return PersonaDefaults(
            name="MOFA-FLEX counselor",
            description="An agent to help you configure MOFA-FLEX.",
            avatar_path="",
            system_prompt="...",
        )

    async def get_memory_store(self):
        if not hasattr(self, "_memory_store"):
            conn = await aiosqlite.connect(MEMORY_STORE_PATH, check_same_thread=False)
            self._memory_store = AsyncSqliteSaver(conn)
        return self._memory_store

    async def get_agent(self, model_id: str, model_args, system_prompt: str, tools: list | None = None):
        model = ChatLiteLLM(**model_args, model=model_id, streaming=True)
        memory_store = await self.get_memory_store()

        return create_agent(model, system_prompt=system_prompt, checkpointer=memory_store, tools=tools)

    async def process_message(self, message: Message) -> None:
        if not JupyternautPersona.config_manager.chat_model:
            self.send_message(
                "No chat model is configured.\n\n"
                "You must set one first in the Jupyter AI settings, found in 'Settings > AI Settings' from the menu bar."
            )
            return

        model_id = JupyternautPersona.config_manager.chat_model
        model_args = JupyternautPersona.config_manager.chat_model_args
        agent = await self.get_agent(
            model_id=model_id, model_args=model_args, system_prompt=SYSTEM_PROMPT, tools=[self.finalize_configuration]
        )

        context = {"thread_id": self.ychat.get_id(), "username": message.sender}

        async def create_aiter():
            async for token, metadata in agent.astream(
                {"messages": [{"role": "user", "content": message.body}]},
                {"configurable": context},
                context={"model_id": model_id, "model_args": model_args, "self": self},
                stream_mode="messages",
            ):
                node = metadata["langgraph_node"]
                content_blocks = token.content_blocks
                if node == "model" and content_blocks:
                    if token.text:
                        yield token.text

        response_aiter = create_aiter()
        await self.stream_message(response_aiter)

    @tool(args_schema=FinalizeConfigurationSchema)
    async def finalize_configuration(
        runtime: ToolRuntime,
        n_factors: int,
        layer: str | None,
        group_by: str | Sequence[str] | None,
        factor_prior: str,
        weight_prior: str,
        nonnegative_factors: bool,
        nonnegative_weights: bool,
    ):
        """Finalize the MOFA-FLEX configuration and generate runnable code. Only call this tool once you have collected all relevant information from the user."""
        runtime.context["self"].send_message(
            "finalize_configuration called\n\n"
            f"n_factors: {n_factors}\n\n"
            f"layer: {layer}\n\n"
            f"group_by: {group_by}\n\n"
            f"factor_prior: {factor_prior}\n\n"
            f"weight_prior: {weight_prior}\n\n"
            f"nonnegative_factors: {nonnegative_factors}\n\n"
            f"nonnegative_weights: {nonnegative_weights}"
        )
