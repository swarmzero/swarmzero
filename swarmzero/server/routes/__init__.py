from typing import Any

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI

from swarmzero.database.database import get_db, initialize_db, setup_chats_table
from swarmzero.sdk_context import SDKContext

from .chat import setup_chat_routes
from .database import setup_database_routes
from .files import setup_files_routes
from .vectorindex import setup_vectorindex_routes

load_dotenv()


def setup_routes(app: FastAPI, id: str, sdk_context: SDKContext):

    @app.on_event("startup")
    async def startup_event():

        await initialize_db()

        async for db in get_db():
            await setup_chats_table(db)

    @app.get("/")
    def read_root():
        return {"message": "Agent is running"}

    v1 = APIRouter()

    setup_database_routes(v1)
    setup_chat_routes(v1, id, sdk_context)
    setup_files_routes(v1, id, sdk_context)
    setup_vectorindex_routes(v1)

    app.include_router(v1, prefix="/api/v1")
