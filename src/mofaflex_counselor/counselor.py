import asyncio
import os
from collections.abc import Callable
from importlib import resources
from io import StringIO
from pathlib import Path
from typing import Annotated

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
from pydantic import Field, create_model

from .notebook import DataAnalysisResult, analyze_active_notebook
from .parametersmodel import make_mofaflex_parameters_model
from .utils import DEBUG

MEMORY_STORE_PATH = os.path.join(jupyter_data_dir(), "jupyter_ai", "memory.sqlite")
AVATAR_PATH = str((Path(__file__).parent / "logo.svg").absolute())
SYSTEM_PROMPT = (resources.files(__package__) / "prompts/chat_system_prompt.txt").read_text()
DEBATER_SYSTEM_PROMPT = (resources.files(__package__) / "prompts/debater_system_prompt.txt").read_text()
MODERATOR_SYSTEM_PROMPT = (resources.files(__package__) / "prompts/moderator_system_prompt.txt").read_text()
MODERATOR_MESSAGE_PROMPT = (resources.files(__package__) / "prompts/moderator_msg_prompt.txt").read_text()
JUDGE_MESSAGE_PROMPT = (resources.files(__package__) / "prompts/judge_msg_prompt.txt").read_text()
DATA_PROPERTIES_PROMPT = (resources.files(__package__) / "prompts/data_properties.txt").read_text()


class ModeratorAgentState(AgentState):
    finished: bool
    judging: bool
    notebook_var_name: str | None
    notebook_var_type: str | None


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

    async def process_message(self, message: Message) -> None:
        if not JupyternautPersona.config_manager.chat_model:
            self.send_message(
                "No chat model is configured.\n\n"
                "You must set one first in the Jupyter AI settings, found in 'Settings > AI Settings' from the menu bar."
            )
            return

        model_id = JupyternautPersona.config_manager.chat_model
        model_args = JupyternautPersona.config_manager.chat_model_args
        self._model = ChatLiteLLM(**model_args, model=model_id, streaming=True)
        memory_store = await self.get_memory_store()

        configurable = {"configurable": {"thread_id": self.ychat.get_id(), "username": message.sender}}

        async def create_aiter():
            try:
                store = await self.get_memory_store()
                checkpoint = await anext(store.alist(configurable, limit=1))
                analyzed = configurable["data_analysis_result"] = checkpoint.metadata.get("data_analysis_result")
                data_prompt = configurable["data_prompt"] = checkpoint.metadata.get("data_prompt")
                if analyzed is not None:
                    analyzed = DataAnalysisResult.model_validate_json(analyzed)
            except StopAsyncIteration:  # new thread
                async for analyzed in analyze_active_notebook(self._model):
                    if isinstance(analyzed, str):
                        yield analyzed
            if DEBUG:
                yield f"`{analyzed}`"
            parameters_model = make_mofaflex_parameters_model(analyzed)
            finalize_configuration_schema = create_model("FinalizeConfigurationSchema", parameters=parameters_model)

            system_prompt = SYSTEM_PROMPT
            data_prompt = None
            if analyzed is not None:
                configurable["data_analysis_result"] = analyzed.model_dump_json()
                data_prompt = configurable["data_prompt"] = DATA_PROPERTIES_PROMPT.format(
                    type=analyzed.type, n_views=analyzed.n_views, n_obs=analyzed.n_obs, n_vars=analyzed.n_vars
                )
                system_prompt += data_prompt
                var_name = analyzed.var_name
                var_type = analyzed.type
            else:
                var_name = None
                var_type = None

            # can't use functools.partial or lambdas because langchain needs type hints
            async def finalize_configuration_tool(runtime: ToolRuntime, parameters):
                return await self.finalize_configuration(
                    parameters_model, data_prompt, var_name, var_type, runtime, parameters
                )

            agent = create_agent(
                self._model,
                system_prompt=system_prompt,
                checkpointer=memory_store,
                tools=[
                    tool(
                        "finalize_configuration",
                        description=self.finalize_configuration.__doc__,
                        args_schema=finalize_configuration_schema,
                    )(finalize_configuration_tool)
                ],
                name="user_facing",
            )

            async for stream_mode, data in agent.astream(
                {"messages": [{"role": "user", "content": message.body}]},
                configurable,
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
    async def moderator(runtime, debaters, moderator, debater_responses, round, var_name, var_type, judging=False):
        if not isinstance(debater_responses, str):
            debater_responses = [
                f"Debater {i + 1} argued:\n-----------------\n{response['messages'][-1].text}\n\n"
                for i, response in enumerate(debater_responses)
            ]
        if not judging:
            moderator_msg = MODERATOR_MESSAGE_PROMPT.format(
                round=round + 1, debater_responses="".join(debater_responses)
            )
        else:
            moderator_msg = JUDGE_MESSAGE_PROMPT.format(debater_responses="".join(debater_responses))
        if DEBUG:
            runtime.stream_writer(moderator_msg)
        moderator_response = await moderator.ainvoke(
            {
                "messages": [HumanMessage(moderator_msg)],
                "finished": False,
                "judging": judging,
                "notebook_var_name": var_name,
                "notebook_var_type": var_type,
            }
        )

        return moderator_response, debater_responses

    @wrap_model_call
    async def force_finalize(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]):
        if request.state["judging"]:
            request = request.override(tool_choice="finalize_configuration")
        return await handler(request)

    async def finalize_configuration(
        self, parameters_model, data_prompt, var_name, var_type, runtime: ToolRuntime, parameters
    ):
        """Finalize the MOFA-FLEX configuration and generate runnable Python code.

        Only call this tool once you have collected all relevant information from the user.
        Provide the Python code to the user in a markdown code block.
        Do not edit or change the code in any way.
        The tool additionally outputs a summary of the reasoning leading to the final parameter choice.
        Provide the summary to the user also.
        """
        ndebaters = 2
        nrounds = 3
        if DEBUG:
            runtime.stream_writer(parameters.model_dump_json())

        transcript = StringIO()
        for message in runtime.state["messages"]:
            if isinstance(message, HumanMessage):
                transcript.writelines(("USER\n", "----\n", message.text, "\n"))
            elif isinstance(message, AIMessage) and message.text:
                transcript.writelines(("ASSISTANT\n", "---------\n", message.text, "\n"))
        debater_system_prompt = DEBATER_SYSTEM_PROMPT.format(
            parameters=parameters_model.model_json_schema(), transcript=transcript.getvalue()
        )
        moderator_system_prompt = MODERATOR_SYSTEM_PROMPT.format(
            selected_params=parameters.model_dump_json(exclude_unset=True)
        )

        if data_prompt:
            debater_system_prompt += data_prompt
            moderator_system_prompt += data_prompt

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
        finalize_configuration_schema = create_model(
            "FinalizeConfigurationSchema",
            parameters=parameters_model,
            summary=Annotated[str, Field(description="Summary of the reasoning for the final parameter choice.")],
        )
        moderator = create_agent(
            self._model,
            system_prompt=moderator_system_prompt,
            tools=[
                tool("finalize_configuration", args_schema=finalize_configuration_schema)(
                    self.finalize_configuration_after_debate
                )
            ],
            state_schema=ModeratorAgentState,
            middleware=[self.force_finalize],
            name="moderator",
        )

        if DEBUG:
            runtime.stream_writer("Thinking...\n\nround 0\n\n")
        else:
            runtime.stream_writer("Thinking...\n\n")
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
                runtime, debaters, moderator, debater_responses, round, var_name, var_type
            )
            if DEBUG:
                runtime.stream_writer(moderator_response["messages"][-1].text)

            if moderator_response["finished"]:
                for msg in reversed(moderator_response["messages"]):
                    if isinstance(msg, ToolMessage):
                        return msg.text

            if DEBUG:
                runtime.stream_writer(f"Thinking...\n\nround {round + 1}\n\n")
            else:
                runtime.stream_writer("Thinking...\n\n")
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

        moderator_response, debater_msgs = await self.moderator(
            runtime, debaters, moderator, debater_responses, round, var_name
        )
        if DEBUG:
            runtime.stream_writer(moderator_response["messages"][-1].text)

        if not moderator_response["finished"]:
            moderator_response, _ = await self.moderator(
                runtime, debaters, moderator, debater_msgs, round, var_name, judging=True
            )
        if not moderator_response["finished"]:
            self.send_message("Something went horribly wrong.")
        for msg in reversed(moderator_response["messages"]):
            if isinstance(msg, ToolMessage):
                return msg.text

    async def finalize_configuration_after_debate(self, runtime: ToolRuntime, parameters, summary):
        """Finalize the MOFA-FLEX configuration and generate runnable Python code.

        Only call this tool once you have arrived at a final answer.
        """
        if DEBUG:
            runtime.stream_writer(parameters.model_dump_json())
        data = runtime.state["notebook_var_name"] or "data"
        if (
            var_type := runtime.state["notebook_var_type"] or getattr(parameters, "data_type", None)
        ) and var_type == "AnnData":
            data = f"{{'group_1': {{'view_1': {data}}}}}"
        code = f"""```python
import mofaflex as mfl
model = mfl.MOFAFLEX({data},
                     mfl.ModelOptions(n_factors={parameters.n_factors},
                                      factor_prior={parameters.factor_prior!r},
                                      weight_prior={parameters.weight_prior!r},
                                      nonnegative_factors={parameters.nonnegative_factors},
                                      nonnegative_weights={parameters.nonnegative_weights},
                                     ),
                     mfl.DataOptions(layer={getattr(parameters, "layer", None)!r},
                                     group_by={getattr(parameters, "group_by", None)!r},
                                     annotations_varm_key={getattr(parameters, "annotattions_varm_key", None)!r},
                                    ),
                    )
```"""
        return Command(
            update={
                "finished": True,
                "messages": [
                    ToolMessage(
                        content_blocks=[{"type": "text", "text": code}, {"type": "text", "text": summary}],
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    def shutdown(self):
        if hasattr(self, "_memory_store"):
            self.parent.event_loop.create_task(self._memory_store.conn.close())
        super().shutdown()
