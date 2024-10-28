import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from llama_index.core.agent.runner.base import AgentRunner
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.schema import ImageDocument

from swarmzero.database.database import DatabaseManager
from swarmzero.filestore import BASE_DIR, FileStore

file_store = FileStore(BASE_DIR)


class ChatManager:

    def __init__(self, llm: AgentRunner, user_id: str, session_id: str, enable_multi_modal: bool = False):
        self.allowed_image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}
        self.llm = llm
        self.user_id = user_id
        self.session_id = session_id
        self.chat_store_key = f"{user_id}_{session_id}"
        self.enable_multi_modal = enable_multi_modal

    def is_valid_image(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in self.allowed_image_extensions

    async def add_message(self, db_manager: DatabaseManager, role: str, content: Any | None):
        data = {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "message": content,
            "role": role,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if "AGENT_ID" in os.environ:
            data["agent_id"] = os.getenv("AGENT_ID", "")
        if "SWARM_ID" in os.environ:
            data["swarm_id"] = os.getenv("SWARM_ID", "")

        await db_manager.insert_data(
            table_name="chats",
            data=data,
        )

    async def get_messages(self, db_manager: DatabaseManager):
        filters = {"user_id": [self.user_id], "session_id": [self.session_id]}
        if "AGENT_ID" in os.environ:
            filters["agent_id"] = os.getenv("AGENT_ID", "")
        if "SWARM_ID" in os.environ:
            filters["swarm_id"] = os.getenv("SWARM_ID", "")

        db_chat_history = await db_manager.read_data("chats", filters)
        chat_history = [ChatMessage(role=chat["role"], content=chat["message"]) for chat in db_chat_history]
        return chat_history

    async def get_all_chats_for_user(self, db_manager: DatabaseManager):
        filters = {"user_id": [self.user_id]}
        if "AGENT_ID" in os.environ:
            filters["agent_id"] = os.getenv("AGENT_ID", "")
        if "SWARM_ID" in os.environ:
            filters["swarm_id"] = os.getenv("SWARM_ID", "")

        db_chat_history = await db_manager.read_data("chats", filters)

        chats_by_session: dict[str, list] = {}
        for chat in db_chat_history:
            session_id = chat["session_id"]
            if session_id not in chats_by_session:
                chats_by_session[session_id] = []
            chats_by_session[session_id].append(
                {
                    "message": chat["message"],
                    "role": chat["role"],
                    "timestamp": chat["timestamp"],
                }
            )

        return chats_by_session

    async def generate_response(
        self,
        db_manager: Optional[DatabaseManager],
        last_message: ChatMessage,
        files: Optional[List[str]] = [],
    ) -> str:
        chat_history = []

        if db_manager is not None:
            chat_history = await self.get_messages(db_manager)
            await self.add_message(db_manager, last_message.role.value, last_message.content)

        if self.enable_multi_modal:
            image_documents = (
                [
                    ImageDocument(image=file_store.get_file(image_path))
                    for image_path in files
                    if self.is_valid_image(image_path)
                ]
                if files is not None and len(files) > 0
                else []
            )

            assistant_message = await self._handle_openai_multimodal(last_message, chat_history, image_documents)
        else:
            assistant_message = await self._handle_openai_agent(last_message, chat_history)

        if db_manager is not None:
            await self.add_message(db_manager, MessageRole.ASSISTANT, assistant_message)

        return assistant_message

    async def _handle_openai_multimodal(
        self,
        last_message: ChatMessage,
        chat_history: List[ChatMessage],
        image_documents: List[ImageDocument],
    ) -> str:

        self.llm.memory = ChatMemoryBuffer.from_defaults(chat_history=chat_history)
        task = self.llm.create_task(str(last_message.content), extra_state={"image_docs": image_documents})
        return await self._execute_task(task.task_id)

    async def _handle_openai_agent(
        self,
        last_message: ChatMessage,
        chat_history: List[ChatMessage],
    ) -> str:
        response_stream = await self.llm.astream_chat(last_message.content, chat_history=chat_history)
        return "".join([token async for token in response_stream.async_response_gen()])

    async def _execute_task(self, task_id: str) -> str:
        while True:
            try:
                response = await self.llm._arun_step(task_id)
                if response.is_last:
                    return str(self.llm.finalize_response(task_id))
            except Exception as e:
                return f"error during step execution: {str(e)}"
