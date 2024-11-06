import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, List, Optional

from llama_index.core.agent.runner.base import AgentRunner
from llama_index.core.chat_engine.types import StreamingAgentChatResponse
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.schema import ImageDocument

from swarmzero.core.services.chat.suggest_questions import NextQuestionSuggestion
from swarmzero.core.services.database import DatabaseManager
from swarmzero.core.services.filestore import BASE_DIR, FileStore
from swarmzero.utils.callback import EventCallbackHandler

file_store = FileStore(BASE_DIR)


class ChatManager:

    def __init__(self, llm: AgentRunner, user_id: str, session_id: str, enable_multi_modal: bool = False):
        self.allowed_image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}
        self.llm = llm
        self.user_id = user_id
        self.session_id = session_id
        self.chat_store_key = f"{user_id}_{session_id}"
        self.enable_multi_modal = enable_multi_modal
        self.next_question_suggestion = NextQuestionSuggestion()

    def is_valid_image(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in self.allowed_image_extensions

    async def add_message(self, db_manager: DatabaseManager, role: str, content: Any | None, event: Any | None):
        data = {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "message": content,
            "role": role,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
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
            filters["agent_id"] = [os.getenv("AGENT_ID", "")]
        if "SWARM_ID" in os.environ:
            filters["swarm_id"] = [os.getenv("SWARM_ID", "")]

        db_chat_history = await db_manager.read_data("chats", filters)
        chat_history = [ChatMessage(role=chat["role"], content=chat["message"]) for chat in db_chat_history]
        return chat_history

    async def get_all_chats_for_user(self, db_manager: DatabaseManager):
        filters = {"user_id": [self.user_id]}
        if "AGENT_ID" in os.environ:
            filters["agent_id"] = [os.getenv("AGENT_ID", "")]
        if "SWARM_ID" in os.environ:
            filters["swarm_id"] = [os.getenv("SWARM_ID", "")]

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
        event_handler: EventCallbackHandler = None,
        stream_mode: bool = False,
    ) -> AsyncGenerator[str, None]:
        chat_history = []
        collected_event_chunks = []
        collected_message_chunks = []

        if db_manager is not None:
            chat_history = await self.get_messages(db_manager)
            await self.add_message(db_manager, last_message.role.value, last_message.content, [])

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

            async for chunk in self._handle_multimodal_task(
                last_message, chat_history, image_documents, event_handler, stream_mode
            ):
                if db_manager is not None and not stream_mode:
                    if isinstance(chunk, dict):
                        collected_event_chunks.append(chunk)
                    else:
                        if chunk != "END_OF_STREAM":
                            await self.add_message(db_manager, MessageRole.ASSISTANT, chunk, collected_event_chunks)
                            collected_event_chunks = []

                if db_manager is not None and stream_mode:
                    if isinstance(chunk, dict):
                        collected_event_chunks.append(chunk)
                    else:
                        collected_message_chunks.append(chunk)
                    if chunk == "END_OF_STREAM":
                        await self.add_message(
                            db_manager,
                            MessageRole.ASSISTANT,
                            ''.join(collected_message_chunks[:-1]),
                            collected_event_chunks,
                        )
                        collected_event_chunks = []
                        collected_message_chunks = []

                yield chunk

            question_data = await self.next_question_suggestion.suggest_next_questions(
                chat_history, ''.join(collected_message_chunks[:-1]), self.llm
            )
            print(question_data)
            yield question_data

        else:
            async for chunk in self._handle_task(last_message, chat_history, event_handler, stream_mode):
                if db_manager is not None and not stream_mode:
                    if isinstance(chunk, dict):
                        collected_event_chunks.append(chunk)
                    else:
                        if chunk != "END_OF_STREAM":
                            await self.add_message(db_manager, MessageRole.ASSISTANT, chunk, collected_event_chunks)
                            collected_event_chunks = []

                if db_manager is not None and stream_mode:
                    if isinstance(chunk, dict):
                        collected_event_chunks.append(chunk)
                    else:
                        collected_message_chunks.append(chunk)
                    if chunk == "END_OF_STREAM":
                        await self.add_message(
                            db_manager,
                            MessageRole.ASSISTANT,
                            ''.join(collected_message_chunks[:-1]),
                            collected_event_chunks,
                        )
                        collected_event_chunks = []
                        collected_message_chunks = []

                yield chunk

            question_data = await self.next_question_suggestion.suggest_next_questions(
                chat_history, ''.join(collected_message_chunks[:-1]), self.llm
            )
            print(question_data)
            yield question_data

    async def _handle_multimodal_task(
        self,
        last_message: ChatMessage,
        chat_history: List[ChatMessage],
        image_documents: List[ImageDocument],
        event_handler: EventCallbackHandler,
        stream_mode: bool = False,
    ) -> AsyncGenerator[str, None]:
        self.llm.memory = ChatMemoryBuffer.from_defaults(chat_history=chat_history)
        task = self.llm.create_task(str(last_message.content), extra_state={"image_docs": image_documents})
        async for chunk in self._execute_task(task.task_id, event_handler, stream_mode):
            yield chunk

    async def _handle_task(
        self,
        last_message: ChatMessage,
        chat_history: List[ChatMessage],
        event_handler: EventCallbackHandler,
        stream_mode: bool = False,
    ) -> AsyncGenerator[str, None]:
        task = self.llm.create_task(last_message.content, chat_history=chat_history)
        async for chunk in self._execute_task(task.task_id, event_handler, stream_mode):
            yield chunk

    async def _execute_task(
        self, task_id: str, event_handler: EventCallbackHandler, stream_mode: bool = False
    ) -> AsyncGenerator[str, None]:

        while True:
            try:
                response = await self.llm.astream_step(task_id) if stream_mode else await self.llm._arun_step(task_id)

                if isinstance(response.output, StreamingAgentChatResponse):
                    async for token in response.output.async_response_gen():
                        yield str(token)
                else:
                    # If not streaming, send the complete response
                    yield str(response.output)

                # Process events
                if event_handler:
                    while not event_handler._aqueue.empty():
                        event = await event_handler._aqueue.get()
                        event_response = event.get('data')
                        if event_response is not None:
                            yield event_response

                if response.is_last:
                    # Signal the end of the stream
                    yield "END_OF_STREAM"
                    break

            except Exception as e:
                yield f"error during step execution: {str(e)}"
                break
