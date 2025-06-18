import json
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
from langtrace_python_sdk import inject_additional_attributes
from llama_index.core.llms import ChatMessage, MessageRole
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from swarmzero.chat import ChatManager
from swarmzero.chat.schemas import ChatData, ChatHistorySchema
from swarmzero.database.database import DatabaseManager, get_db
from swarmzero.server.routes.files import insert_files_to_index


def setup_swarm_chat_routes(router: APIRouter, swarm, sdk_context):
    async def validate_chat_data(chat_data_parsed: ChatData):
        if not chat_data_parsed.messages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No messages provided",
            )
        messages_copy = list(chat_data_parsed.messages)
        last_message_data = messages_copy.pop()
        if last_message_data.role != MessageRole.USER:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Last message must be from user",
            )
        last_message = ChatMessage(role=last_message_data.role, content=last_message_data.content)
        chat_history = [ChatMessage(role=m.role, content=m.content) for m in messages_copy]
        return last_message, chat_history

    @router.post("/swarm/chat")
    async def chat(
        request: Request,
        user_id: str = Form(...),
        session_id: str = Form(...),
        chat_data: str = Form(...),
        files: List[UploadFile] = File([]),
        verbose: bool = Form(False),
        db: AsyncSession = Depends(get_db),
    ):
        await swarm._ensure_utilities_loaded()
        internal_swarm_agent = getattr(swarm, "_Swarm__swarm", None)
        if internal_swarm_agent is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Swarm is not initialized. Ensure that agents are properly configured.",
            )
        try:
            chat_data_parsed = ChatData.model_validate_json(chat_data)
        except ValidationError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Chat data is malformed: {e.json()}",
            )
        last_message, _ = await validate_chat_data(chat_data_parsed)
        callback_handler = sdk_context.get_utility("reasoning_callback") if verbose else None
        chat_manager = ChatManager(internal_swarm_agent, user_id=user_id, session_id=session_id, swarm_id=swarm.id)
        db_manager = DatabaseManager(db)
        stored_files = []
        if files and len(files) > 0:
            stored_files = await insert_files_to_index(files, swarm.id, sdk_context, user_id, session_id)
        response_content = ""
        async for chunk in inject_additional_attributes(
            lambda: chat_manager.generate_response(
                db_manager, last_message, stored_files, callback_handler, stream_mode=False, verbose=verbose
            ),
            {"user_id": user_id},
        ):
            try:
                if verbose:
                    if isinstance(chunk, dict):
                        response_content += f"0:{json.dumps(chunk)}\n"
                    elif isinstance(chunk, list):
                        response_content += f"2:{json.dumps(chunk)}\n"
                    else:
                        response_content += f"1:{json.dumps(chunk)}\n"
                else:
                    if isinstance(chunk, str):
                        response_content += chunk
            except Exception as e:
                pass
        return response_content

    @router.post("/swarm/chat_stream")
    async def chat_stream(
        request: Request,
        user_id: str = Form(...),
        session_id: str = Form(...),
        chat_data: str = Form(...),
        files: List[UploadFile] = File([]),
        verbose: bool = Form(False),
        db: AsyncSession = Depends(get_db),
    ):
        await swarm._ensure_utilities_loaded()
        internal_swarm_agent = getattr(swarm, "_Swarm__swarm", None)
        if internal_swarm_agent is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Swarm is not initialized. Ensure that agents are properly configured.",
            )
        try:
            chat_data_parsed = ChatData.model_validate_json(chat_data)
        except ValidationError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Chat data is malformed: {e.json()}",
            )
        last_message, _ = await validate_chat_data(chat_data_parsed)
        callback_handler = sdk_context.get_utility("reasoning_callback") if verbose else None
        chat_manager = ChatManager(internal_swarm_agent, user_id=user_id, session_id=session_id, swarm_id=swarm.id)
        db_manager = DatabaseManager(db)
        stored_files = []
        if files and len(files) > 0:
            stored_files = await insert_files_to_index(files, swarm.id, sdk_context, user_id, session_id)

        async def stream_response_generator():
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
                    yield f"data: Error: {str(e)}\n\n"

        return StreamingResponse(
            inject_additional_attributes(stream_response_generator, {"user_id": user_id}),
            media_type="text/event-stream",
        )
