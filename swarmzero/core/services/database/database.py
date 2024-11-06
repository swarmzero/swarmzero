import json
import logging
import os
import traceback
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Type

from dotenv import load_dotenv
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    select,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeMeta, declarative_base, sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy.types import Boolean, Float

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs("swarmzero-data/db", exist_ok=True)
db_url = os.getenv("SWARMZERO_DATABASE_URL") or "sqlite+aiosqlite:///swarmzero-data/db/swarmzero.db"

poolclass = None
connect_args = {}

if db_url.startswith("postgresql+asyncpg://"):
    connect_args = {"statement_cache_size": 0}
    poolclass = NullPool

engine = create_async_engine(db_url, echo=False, connect_args=connect_args, poolclass=poolclass)
SessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)  # type: ignore
Base = declarative_base()

_model_registry: Dict[str, Tuple[Type[DeclarativeMeta], MetaData]] = {}


class TableDefinition(Base):  # type: ignore
    __tablename__ = "table_definitions"
    id = Column(Integer, primary_key=True, index=True)
    table_name = Column(String, unique=True, index=True)
    columns = Column(JSON)


async def initialize_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@asynccontextmanager
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session with proper resource management"""
    session = SessionLocal()
    try:
        yield session
    finally:
        await session.close()


async def setup_chats_table(db: AsyncSession):
    db_manager = DatabaseManager(db)
    table_exists = await db_manager.get_table_definition("chats")

    if table_exists:
        logger.info("Table 'chats' already exists. Skipping creation.")
        return

    columns = {
        "user_id": "String",
        "session_id": "String",
        "message": "String",
        "role": "String",
        "timestamp": "String",
        "agent_id": "String",
        "swarm_id": "String",
        "event": "JSON",
    }

    columns_json = json.dumps(columns)
    await db_manager.create_table("chats", columns)
    logger.info("Table 'chats' created successfully.")

    return db_manager


class DatabaseManager:
    sqlalchemy_types = {
        "String": String,
        "Integer": Integer,
        "JSON": JSON,
        "Float": Float,
        "DateTime": DateTime,
        "Boolean": Boolean,
        "Text": Text,
    }

    def __init__(self, db: AsyncSession):
        self.db = db
        self._engine = None
        self._session = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()

    async def close(self):
        """Explicitly close the database connection"""
        if self.db:
            await self.db.close()

    async def cleanup(self):
        """Cleanup database resources"""
        if hasattr(self, 'db') and self.db is not None:
            await self.db.close()
            self.db = None

    @staticmethod
    @lru_cache
    def _generate_model_class(table_name: str, columns_str: str):
        """Cache the model creation to prevent duplicate class definitions"""
        if table_name in _model_registry:
            return _model_registry[table_name]

        # Handle both JSON string and dict inputs
        if isinstance(columns_str, str):
            try:
                columns = json.loads(columns_str)
            except json.JSONDecodeError:
                # If it fails to parse as JSON, try evaluating as a string representation of a dict
                columns = eval(columns_str)  # Note: Only use eval for internal trusted data
        else:
            columns = columns_str

        metadata = MetaData()
        columns_list: List[Column] = []
        for name, column_type in columns.items():
            column_type_class = DatabaseManager.sqlalchemy_types.get(column_type)
            if not column_type_class:
                raise ValueError(f"Unsupported column type: {column_type}")
            columns_list.append(Column(name, column_type_class))
        columns_list.insert(0, Column("id", Integer, primary_key=True))
        table = Table(table_name, metadata, *columns_list)

        model = type(
            f"{table_name.capitalize()}Model",  # Make model name unique
            (Base,),
            {
                "__tablename__": table_name,
                "__table__": table,
                "__mapper_args__": {"eager_defaults": True},
            },
        )

        _model_registry[table_name] = (model, metadata)
        return model, metadata

    async def create_table(self, table_name: str, columns: Dict[str, str]):
        logger.info(f"Creating table '{table_name}' with columns: {columns}")
        try:
            if not isinstance(columns, dict):
                raise ValueError("Columns must be a dictionary")

            columns_json = json.dumps(columns)
            table_definition = TableDefinition(table_name=table_name, columns=columns)
            self.db.add(table_definition)
            await self.db.commit()
            logger.info(f"Table definition for '{table_name}' saved.")

            _, metadata = self._generate_model_class(table_name, columns_json)
            async with engine.begin() as conn:
                await conn.run_sync(metadata.create_all)
            logger.info(f"Table '{table_name}' created successfully.")
        except SQLAlchemyError as e:
            await self.db.rollback()
            logger.error(f"Error creating table '{table_name}': {str(e)}")
            raise ValueError(f"Error creating table: {str(e)}")

    async def get_table_definition(self, table_name: str):
        logger.info(f"Retrieving table definition for '{table_name}'")
        try:
            result = await self.db.execute(select(TableDefinition).filter_by(table_name=table_name))
            table_definition = result.scalars().first()
            if table_definition:
                logger.info(f"Table definition for '{table_name}' retrieved successfully.")
                return table_definition.columns
            logger.warning(f"Table definition for '{table_name}' not found.")
            return None
        except SQLAlchemyError as e:
            logger.error(f"Error retrieving table definition for '{table_name}': {str(e)}")
            raise ValueError(f"Error retrieving table definition: {str(e)}")

    async def insert_data(self, table_name: str, data: Dict[str, Any]):
        logger.info(f"Inserting data into '{table_name}': {data}")
        try:
            columns = await self.get_table_definition(table_name)
            if not columns:
                raise ValueError(f"Table '{table_name}' does not exist.")

            model, metadata = self._generate_model_class(table_name, str(columns))
            async with engine.begin() as conn:
                await conn.run_sync(metadata.create_all)

            instance = model(**data)
            self.db.add(instance)
            await self.db.commit()
            await self.db.refresh(instance)
            logger.info(f"Data inserted into '{table_name}' successfully, id: {instance.id}")
            return instance
        except SQLAlchemyError as e:
            await self.db.rollback()
            logger.error(f"Error inserting data into '{table_name}': {str(e)}")
            raise ValueError(f"Error inserting data: {str(e)}")

    async def read_data(self, table_name: str, filters: Optional[Dict[str, List[Any]]] = None):
        try:
            columns = await self.get_table_definition(table_name)
            if not columns:
                raise ValueError(f"Table '{table_name}' does not exist.")

            columns_str = json.dumps(columns, sort_keys=True)
            model, metadata = self._generate_model_class(table_name, columns_str)

            query = select(model)
            if filters:
                for key, values in filters.items():
                    if key == "details":
                        for value in values:
                            for sub_key, sub_value in value.items():
                                query = query.where(getattr(model, key)[sub_key] == sub_value)
                    else:
                        query = query.filter(getattr(model, key).in_(values))

            result = await self.db.execute(query)
            instances = result.scalars().all()
            data = [{column: getattr(instance, column) for column in columns.keys()} for instance in instances]
            logger.info(f"Data read from '{table_name}' successfully.")
            return data
        except Exception as e:
            logger.error(f"Error reading data from '{table_name}': {str(e)}")
            raise

    async def update_data(self, table_name: str, row_id: int, new_data: Dict[str, Any]):
        logger.info(f"Updating data in '{table_name}' for id {row_id} with new data: {new_data}")
        try:
            columns = await self.get_table_definition(table_name)
            if not columns:
                raise ValueError(f"Table '{table_name}' does not exist.")

            model, metadata = self._generate_model_class(table_name, str(columns))
            async with engine.begin() as conn:
                await conn.run_sync(metadata.create_all)

            instance = await self.db.get(model, row_id)
            if instance:
                for key, value in new_data.items():
                    setattr(instance, key, value)
                await self.db.commit()
                await self.db.refresh(instance)
                logger.info(f"Data in '{table_name}' for id {row_id} updated successfully.")
            else:
                logger.warning(f"No data found with id {row_id} in '{table_name}'.")
                raise ValueError(f"No data found with id {row_id} in '{table_name}'.")
        except SQLAlchemyError as e:
            await self.db.rollback()
            logger.error(f"Error updating data in '{table_name}' for id {row_id}: {str(e)}")
            raise ValueError(f"Error updating data: {str(e)}")

    async def delete_data(self, table_name: str, row_id: int):
        logger.info(f"Deleting data from '{table_name}' for id {row_id}")
        try:
            columns = await self.get_table_definition(table_name)
            if not columns:
                raise ValueError(f"Table '{table_name}' does not exist.")

            model, metadata = self._generate_model_class(table_name, str(columns))
            async with engine.begin() as conn:
                await conn.run_sync(metadata.create_all)

            instance = await self.db.get(model, row_id)
            if instance:
                await self.db.delete(instance)
                await self.db.commit()
                logger.info(f"Data deleted from '{table_name}' for id {row_id} successfully.")
            else:
                logger.warning(f"No data found with id {row_id} in '{table_name}'.")
                raise ValueError(f"No data found with id {row_id} in '{table_name}'.")
        except SQLAlchemyError as e:
            await self.db.rollback()
            logger.error(f"Error deleting data from '{table_name}' for id {row_id}: {str(e)}")
            raise ValueError(f"Error deleting data: {str(e)}")


def register_cleanup():
    """Register database cleanup handler for graceful shutdown"""
    import atexit

    def cleanup_handler():
        try:
            # Synchronous cleanup only
            _model_registry.clear()
            engine.sync_engine.dispose()
            logger.info("Database cleanup completed successfully")
        except Exception as e:
            logger.warning(f"Cleanup handler encountered an error: {e}")

    atexit.register(cleanup_handler)


# Call this immediately
register_cleanup()


@asynccontextmanager
async def transaction():
    """Context manager for database transactions"""
    async with SessionLocal() as session:
        async with session.begin():
            try:
                yield session
            except Exception:
                await session.rollback()
                raise


def print_call_stack(message: str = ""):
    print(f"\n{message} Call Stack:")
    for line in traceback.format_stack():
        print(line.strip())
