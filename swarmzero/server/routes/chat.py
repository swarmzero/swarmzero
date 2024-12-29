import json
import logging
from datetime import datetime, timezone
from typing import List

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse
from langtrace_python_sdk import inject_additional_attributes  # type: ignore   # noqa
from llama_index.core.llms import ChatMessage, MessageRole
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from swarmzero.chat import ChatManager
from swarmzero.chat.schemas import ChatData, ChatHistorySchema
from swarmzero.database.database import DatabaseManager, get_db
from swarmzero.llms.openai import OpenAIMultiModalLLM
from swarmzero.sdk_context import SDKContext
from swarmzero.server.routes.files import insert_files_to_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_llm_instance(id, sdk_context: SDKContext):
    attributes = sdk_context.get_attributes(
        id, "llm", "agent_class", "tools", "instruction", "tool_retriever", "enable_multi_modal", "max_iterations"
    )

    if attributes['agent_class'] == OpenAIMultiModalLLM:
        llm_instance = attributes["agent_class"](
            attributes["llm"],
            attributes["tools"],
            attributes["instruction"],
            attributes["tool_retriever"],
            max_iterations=attributes["max_iterations"],
            sdk_context=sdk_context,
        ).agent
    else:
        llm_instance = attributes["agent_class"](
            attributes["llm"],
            attributes["tools"],
            attributes["instruction"],
            attributes["tool_retriever"],
            sdk_context=sdk_context,
        ).agent
    return llm_instance, attributes["enable_multi_modal"]


def setup_chat_routes(router: APIRouter, id, sdk_context: SDKContext):
    async def validate_chat_data(chat_data):
        if len(chat_data.messages) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No messages provided",
            )
        last_message = chat_data.messages.pop()
        if last_message.role != MessageRole.USER:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Last message must be from user",
            )
        return last_message, [ChatMessage(role=m.role, content=m.content) for m in chat_data.messages]

    @router.post("/chat")
    async def chat(
        request: Request,
        user_id: str = Form(...),
        session_id: str = Form(...),
        chat_data: str = Form(...),
        files: List[UploadFile] = File([]),
        verbose: bool = Form(False),
        db: AsyncSession = Depends(get_db),
    ):
        try:
            chat_data_parsed = ChatData.model_validate_json(chat_data)
        except ValidationError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Chat data is malformed: {e.json()}",
            )

        llm_instance, enable_multi_modal = get_llm_instance(id, sdk_context)
        callback_handler = sdk_context.get_utility("reasoning_callback") if verbose else None

        chat_manager = ChatManager(
            llm_instance, user_id=user_id, session_id=session_id, enable_multi_modal=enable_multi_modal
        )
        db_manager = DatabaseManager(db)

        last_message, _ = await validate_chat_data(chat_data_parsed)

        stored_files = []
        if files and len(files) > 0:
            stored_files = await insert_files_to_index(files, id, sdk_context, user_id, session_id)

        response = ""
        async for chunk in inject_additional_attributes(
            lambda: chat_manager.generate_response(
                db_manager, last_message, stored_files, callback_handler, stream_mode=False, verbose=verbose
            ),
            {"user_id": user_id},
        ):
            try:
                if verbose:
                    if isinstance(chunk, dict):
                        response += f"0:{json.dumps(chunk)}\n"
                    elif isinstance(chunk, list):
                        response += f"2:{json.dumps(chunk)}\n"
                    else:
                        response += f"1:{json.dumps(chunk)}\n"
                else:
                    if isinstance(chunk, str):
                        response += chunk
            except Exception as e:
                logger.error(f"Error processing chunk: {e}")

        return response

    @router.post("/chat_stream")
    async def chat_stream(
        request: Request,
        user_id: str = Form(...),
        session_id: str = Form(...),
        chat_data: str = Form(...),
        files: List[UploadFile] = File([]),
        verbose: bool = Form(False),
        db: AsyncSession = Depends(get_db),
    ):
        try:
            chat_data_parsed = ChatData.model_validate_json(chat_data)
        except ValidationError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Chat data is malformed: {e.json()}",
            )

        llm_instance, enable_multi_modal = get_llm_instance(id, sdk_context)
        callback_handler = sdk_context.get_utility("reasoning_callback") if verbose else None

        chat_manager = ChatManager(
            llm_instance, user_id=user_id, session_id=session_id, enable_multi_modal=enable_multi_modal
        )
        db_manager = DatabaseManager(db)

        last_message, _ = await validate_chat_data(chat_data_parsed)

        stored_files = []
        if files and len(files) > 0:
            stored_files = await insert_files_to_index(files, id, sdk_context, user_id, session_id)

        async def stream_response():
            async for chunk in chat_manager.generate_response(
                db_manager, last_message, stored_files, callback_handler, stream_mode=True, verbose=verbose
            ):
                try:
                    if verbose:
                        if isinstance(chunk, dict):
                            yield f"0:{json.dumps(chunk)}\n"
                        elif isinstance(chunk, list):
                            yield f"2:{json.dumps(chunk)}\n"
                        else:
                            yield f"1:{json.dumps(chunk)}\n"
                    else:
                        if isinstance(chunk, str):
                            yield f"data: {chunk}\n\n"
                except Exception as e:
                    logger.error(f"Error processing chunk: {e}")
                    yield f"data: Error: {str(e)}\n\n"

        return StreamingResponse(
            inject_additional_attributes(stream_response, {"user_id": user_id}), media_type="text/event-stream"
        )

    @router.get("/chat_history", response_model=List[ChatHistorySchema])
    async def get_chat_history(
        user_id: str = Query(...),
        session_id: str = Query(...),
        db: AsyncSession = Depends(get_db),
    ):

        llm_instance, enable_multi_modal = get_llm_instance(id, sdk_context)

        chat_manager = ChatManager(llm_instance, user_id=user_id, session_id=session_id)
        db_manager = DatabaseManager(db)
        chat_history = await chat_manager.get_messages(db_manager)
        if not chat_history:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No chat history found for this user",
            )

        return [
            ChatHistorySchema(
                user_id=user_id,
                session_id=session_id,
                message=msg.content,
                role=msg.role,
                timestamp=str(datetime.now(timezone.utc)),
            )
            for msg in chat_history
        ]

    @router.get("/all_chats")
    async def get_all_chats(user_id: str = Query(...), db: AsyncSession = Depends(get_db)):

        llm_instance, enable_multi_modal = get_llm_instance(id, sdk_context)

        chat_manager = ChatManager(llm_instance, user_id=user_id, session_id="")
        db_manager = DatabaseManager(db)
        all_chats = await chat_manager.get_all_chats_for_user(db_manager)

        if not all_chats:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No chats found for this user",
            )

        return all_chats
