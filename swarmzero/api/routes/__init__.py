from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI

from swarmzero.core.sdk_context import SDKContext
from swarmzero.core.services.database import get_db, initialize_db, setup_chats_table

from .chat import setup_chat_routes
from .database import setup_database_routes
from .files import setup_files_routes
from .vectorindex import setup_vectorindex_routes

load_dotenv()


async def setup_routes(app: FastAPI, id: str, sdk_context: SDKContext):

    @app.on_event("startup")
    async def startup_event():

        await initialize_db()

        async with get_db() as db:
            await setup_chats_table(db)

    @app.on_event("shutdown")
    async def shutdown_event():
        db_manager = sdk_context.get_utility("db_manager")
        if db_manager:
            await db_manager.db.close()

    @app.get("/")
    def read_root():
        return {"message": "Agent is running"}

    v1 = APIRouter()

    setup_database_routes(v1)
    setup_chat_routes(v1, id, sdk_context)
    setup_files_routes(v1, id, sdk_context)
    setup_vectorindex_routes(v1)

    app.include_router(v1, prefix="/api/v1")
