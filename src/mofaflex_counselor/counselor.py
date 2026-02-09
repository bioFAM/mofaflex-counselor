import asyncio
import os
from collections.abc import Callable, Sequence
from importlib import resources
from io import StringIO
from pathlib import Path
from typing import Annotated, Literal

import aiosqlite
from jupyter_ai_jupyternaut.jupyternaut.jupyternaut import JupyternautPersona
from jupyter_ai_persona_manager import BasePersona, PersonaDefaults
from jupyter_core.paths import jupyter_data_dir
from jupyterlab_chat.models import Message
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_litellm import ChatLiteLLM
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from pydantic import BaseModel, Field, PositiveInt

MEMORY_STORE_PATH = os.path.join(jupyter_data_dir(), "jupyter_ai", "memory.sqlite")
AVATAR_PATH = str((Path(__file__).parent / "logo.svg").absolute())
SYSTEM_PROMPT = (resources.files(__package__) / "prompts/chat_system_prompt.txt").read_text()
DEBATER_SYSTEM_PROMPT = (resources.files(__package__) / "prompts/debater_system_prompt.txt").read_text()
MODERATOR_SYSTEM_PROMPT = (resources.files(__package__) / "prompts/moderator_system_prompt.txt").read_text()
MODERATOR_MESSAGE_PROMPT = (resources.files(__package__) / "prompts/moderator_msg_prompt.txt").read_text()
JUDGE_MESSAGE_PROMPT = (resources.files(__package__) / "prompts/judge_msg_prompt.txt").read_text()

DEBUG = os.environ.get("MOFAFLEX_DEBUG")


class MofaFlexParameters(BaseModel):
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


class FinalizeConfigurationSchema(BaseModel):
    parameters: MofaFlexParameters


class ModeratorAgentState(AgentState):
    finished: bool
    judging: bool


class MofaFlexCounselor(BasePersona):
    @property
    def defaults(self):
        return PersonaDefaults(
            name="MOFA-FLEX counselor",
            description="An agent to help you configure MOFA-FLEX.",
            avatar_path=AVATAR_PATH,
            system_prompt="...",
        )

    async def get_memory_store(self):
        if not hasattr(self, "_memory_store"):
            conn = await aiosqlite.connect(MEMORY_STORE_PATH, check_same_thread=False)
            self._memory_store = AsyncSqliteSaver(conn)
        return self._memory_store

    async def get_agent(self, model_id: str, model_args, system_prompt: str, tools: list | None = None):
        self._model = ChatLiteLLM(**model_args, model=model_id, streaming=True)
        memory_store = await self.get_memory_store()

        return create_agent(
            self._model, system_prompt=system_prompt, checkpointer=memory_store, tools=tools, name="user_facing"
        )

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
            async for stream_mode, data in agent.astream(
                {"messages": [{"role": "user", "content": message.body}]},
                {"configurable": context},
                context={"self": self},
                stream_mode=["messages", "custom"],
            ):
                if stream_mode == "messages":
                    token, metadata = data
                    if metadata.get("lc_agent_name") == "user_facing":
                        node = metadata["langgraph_node"]
                        content_blocks = token.content_blocks
                        if node == "model" and content_blocks:
                            if token.text:
                                yield token.text
                elif stream_mode == "custom":
                    yield data

        response_aiter = create_aiter()
        await self.stream_message(response_aiter)

    @staticmethod
    async def moderator(runtime, debaters, moderator, debater_responses, round):
        debater_msgs = [
            f"Debater {i + 1} argued:\n-----------------\n{response['messages'][-1].text}\n\n"
            for i, response in enumerate(debater_responses)
        ]
        moderator_msg = MODERATOR_MESSAGE_PROMPT.format(round=round + 1, debater_responses="".join(debater_msgs))
        if DEBUG:
            runtime.stream_writer(moderator_msg)
        moderator_response = await moderator.ainvoke(
            {"messages": [HumanMessage(moderator_msg)], "finished": False, "judging": False}
        )

        return moderator_response, debater_msgs

    @wrap_model_call
    async def force_finalize(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]):
        if request.state["judging"]:
            request = request.override(tool_choice="finalize_configuration")
        return await handler(request)

    @tool(args_schema=FinalizeConfigurationSchema)
    async def finalize_configuration(runtime: ToolRuntime, parameters: MofaFlexParameters):
        """Finalize the MOFA-FLEX configuration and generate runnable code. Only call this tool once you have collected all relevant information from the user."""
        ndebaters = 2
        nrounds = 3
        self = runtime.context["self"]
        if DEBUG:
            runtime.stream_writer(parameters.model_dump_json())

        transcript = StringIO()
        for message in runtime.state["messages"]:
            if isinstance(message, HumanMessage):
                transcript.writelines(("USER\n", "----\n", message.text, "\n"))
            elif isinstance(message, AIMessage) and message.text:
                transcript.writelines(("ASSISTANT\n", "---------\n", message.text, "\n"))
        debater_system_prompt = DEBATER_SYSTEM_PROMPT.format(
            parameters=MofaFlexParameters.model_json_schema(), transcript=transcript.getvalue()
        )
        moderator_system_prompt = MODERATOR_SYSTEM_PROMPT.format(
            selected_params=parameters.model_dump_json(exclude_unset=True)
        )

        debaters_checkpointer = InMemorySaver()
        debaters = [
            create_agent(
                self._model,
                system_prompt=debater_system_prompt,
                checkpointer=debaters_checkpointer,
                name=f"debater_{i}",
            )
            for i in range(ndebaters)
        ]
        moderator = create_agent(
            self._model,
            system_prompt=moderator_system_prompt,
            tools=[self.finalize_configuration_after_debate],
            state_schema=ModeratorAgentState,
            middleware=[self.force_finalize],
            name="moderator",
        )

        if DEBUG:
            runtime.stream_writer("Thinking...\nround 0")
        else:
            runtime.stream_writer("Thinking...")
        debater_responses = await asyncio.gather(
            *(
                debater.ainvoke(
                    {
                        "messages": [
                            HumanMessage(
                                f"These parameters were selected:\n{parameters.model_dump_json(exclude_unset=True)}"
                            )
                        ]
                    },
                    {"configurable": {"thread_id": i}},
                )
                for i, debater in enumerate(debaters)
            )
        )
        for round in range(nrounds - 1):
            moderator_response, debater_msgs = await self.moderator(
                runtime, debaters, moderator, debater_responses, round
            )
            if DEBUG:
                runtime.stream_writer(moderator_response["messages"][-1].text)

            if moderator_response["finished"]:
                for msg in reversed(moderator_response["messages"]):
                    if isinstance(msg, ToolMessage):
                        return msg.text

            if DEBUG:
                runtime.stream_writer(f"Thinking...\nround {round + 1}")
            else:
                runtime.stream_writer("Thinking...")
            debater_responses = await asyncio.gather(
                *(
                    debater.ainvoke(
                        {
                            "messages": [
                                HumanMessage(debater_msg) for j, debater_msg in enumerate(debater_msgs) if i != j
                            ]
                        },
                        {"configurable": {"thread_id": i}},
                    )
                    for i, debater in enumerate(debaters)
                )
            )

        moderator_response, debater_msgs = await self.moderator(runtime, debaters, moderator, debater_responses)
        if DEBUG:
            runtime.stream_writer(moderator_response["messages"][-1].text)

        if not moderator_response["finished"]:
            judge_msg = JUDGE_MESSAGE_PROMPT.format(debater_responses="".join(debater_msgs))
            moderator_response = await moderator.ainvoke(
                {"messages": [HumanMessage(judge_msg)], "finished": False, "judging": True}
            )
        if not moderator_response["finished"]:
            self.send_message("Something went horribly wrong.")
        for msg in reversed(moderator_response["messages"]):
            if isinstance(msg, ToolMessage):
                return msg.text

    @tool("finalize_configuration", args_schema=FinalizeConfigurationSchema)
    async def finalize_configuration_after_debate(runtime: ToolRuntime, parameters: MofaFlexParameters):
        """Finalize the MOFA-FLEX configuration and generate runnable code. Only call this tool once you have arrived at a final answer."""
        if DEBUG:
            runtime.stream_writer(parameters.model_dump_json())
        return Command(
            update={
                "finished": True,
                "messages": [ToolMessage(parameters.model_dump_json(), tool_call_id=runtime.tool_call_id)],
            }
        )
