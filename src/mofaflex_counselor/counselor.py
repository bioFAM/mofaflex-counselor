import os

import aiosqlite
from jupyter_ai_jupyternaut.jupyternaut.jupyternaut import JupyternautPersona
from jupyter_ai_persona_manager import BasePersona, PersonaDefaults
from jupyter_core.paths import jupyter_data_dir
from jupyterlab_chat.models import Message
from langchain.agents import create_agent
from langchain_litellm import ChatLiteLLM
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

MEMORY_STORE_PATH = os.path.join(jupyter_data_dir(), "jupyter_ai", "memory.sqlite")


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

    async def get_agent(self, model_id: str, model_args, system_prompt: str):
        model = ChatLiteLLM(**model_args, model=model_id, streaming=True)
        memory_store = await self.get_memory_store()

        return create_agent(model, system_prompt=system_prompt, checkpointer=memory_store)

    async def process_message(self, message: Message) -> None:
        if not JupyternautPersona.config_manager.chat_model:
            self.send_message(
                "No chat model is configured.\n\n"
                "You must set one first in the Jupyter AI settings, found in 'Settings > AI Settings' from the menu bar."
            )
            return

        model_id = JupyternautPersona.config_manager.chat_model
        model_args = JupyternautPersona.config_manager.chat_model_args
        system_prompt = JupyternautPersona.get_system_prompt(self, model_id=model_id, message=message)
        agent = await self.get_agent(model_id=model_id, model_args=model_args, system_prompt=system_prompt)

        context = {"thread_id": self.ychat.get_id(), "username": message.sender}

        async def create_aiter():
            async for token, metadata in agent.astream(
                {"messages": [{"role": "user", "content": message.body}]},
                {"configurable": context},
                stream_mode="messages",
            ):
                node = metadata["langgraph_node"]
                content_blocks = token.content_blocks
                if node == "model" and content_blocks:
                    if token.text:
                        yield token.text

        response_aiter = create_aiter()
        await self.stream_message(response_aiter)
