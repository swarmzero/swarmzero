import hashlib
import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

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
    asc,
    desc,
    select,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy.types import Boolean, Float

# Import UUID type based on database backend
try:
    from sqlalchemy.dialects.postgresql import UUID as PostgresUUID

    UUID_TYPE = PostgresUUID(as_uuid=True)
except ImportError:
    UUID_TYPE = String  # Fallback for SQLite

load_dotenv()
logger = logging.getLogger(__name__)

os.makedirs("swarmzero-data/db", exist_ok=True)
db_url = os.getenv("SWARMZERO_DATABASE_URL") or "sqlite+aiosqlite:///swarmzero-data/db/swarmzero.db"

poolclass = None
connect_args = {}

if db_url.startswith("postgresql+asyncpg://"):
    connect_args = {"statement_cache_size": 0}
    poolclass = NullPool

    # Register UUID types for asyncpg
    try:
        import asyncpg

        # Register UUID type with asyncpg
        asyncpg.pgproto.pgproto.UUID = uuid.UUID
        logger.info("UUID types registered for asyncpg")
    except ImportError as e:
        logger.warning(f"Could not register UUID types for asyncpg: {e}")
    except Exception as e:
        logger.warning(f"Error registering UUID types for asyncpg: {e}")

engine = create_async_engine(db_url, echo=False, connect_args=connect_args, poolclass=poolclass)
SessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)  # type: ignore
Base = declarative_base()


class TableDefinition(Base):  # type: ignore
    __tablename__ = "table_definitions"
    id = Column(UUID_TYPE, primary_key=True, index=True, default=uuid.uuid4)  # UUID type for PostgreSQL compatibility
    table_name = Column(String, unique=True, index=True)
    columns = Column(JSON)


async def initialize_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    async with SessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def setup_chats_table(db: AsyncSession):
    db_manager = DatabaseManager(db)
    table_exists = await db_manager.get_table_definition("chats")

    # Correct schema definition matching PostgreSQL init.sql
    correct_columns = {
        "user_id": "String",  # User ID remains TEXT as per schema
        "session_id": "String",
        "message": "String",
        "role": "String",
        "timestamp": "TIMESTAMP WITH TIME ZONE",  # TIMESTAMP WITH TIME ZONE field as per PostgreSQL schema
        "agent_id": "UUID",  # UUID field as per PostgreSQL schema
        "swarm_id": "UUID",  # UUID field as per PostgreSQL schema
        "event": "JSON",
    }

    if table_exists:
        # Check if the existing schema is correct
        needs_update = False
        for key, expected_type in correct_columns.items():
            if key in table_exists:
                existing_type = table_exists[key]
                if isinstance(existing_type, dict):
                    existing_type_str = existing_type.get("type", "")
                else:
                    existing_type_str = str(existing_type)

                if existing_type_str != expected_type:
                    needs_update = True
                    break

        if needs_update:
            logger.info("Updating 'chats' table schema definition to correct types.")
            # Update the stored schema definition
            session = await db_manager.get_session()
            async with session as session:
                from sqlalchemy import update

                stmt = (
                    update(TableDefinition).where(TableDefinition.table_name == "chats").values(columns=correct_columns)
                )
                await session.execute(stmt)
                await session.commit()
            logger.info("'chats' table schema definition updated successfully.")
        else:
            logger.info("Table 'chats' already exists with correct schema. Skipping.")

        # Force update the schema to ensure it's correct and clear cache
        await db_manager.force_update_chats_schema()
        return

    await db_manager.create_table("chats", correct_columns)
    logger.info("Table 'chats' created successfully.")


class DatabaseManager:
    sqlalchemy_types = {
        "String": String,
        "Integer": Integer,
        "JSON": JSON,
        "Float": Float,
        "DateTime": DateTime,
        "Boolean": Boolean,
        "Text": Text,
        "UUID": UUID_TYPE,  # Use the imported UUID type
    }

    # Add a cache for generated model classes
    _model_cache = {}

    def __init__(self, db: AsyncSession):
        self.db = db
        self.logger = logging.getLogger(__name__)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """Close the database session."""
        if self.db:
            await self.db.close()

    async def get_session(self) -> AsyncSession:
        """Get a new database session."""
        return SessionLocal()

    def clear_model_cache(self):
        """Clear the model cache to force regeneration of models."""
        self._model_cache.clear()
        self.logger.info("Model cache cleared")

    async def force_update_chats_schema(self):
        """Force update the chats table schema to ensure correct types."""
        correct_columns = {
            "user_id": "String",
            "session_id": "String",
            "message": "String",
            "role": "String",
            "timestamp": "TIMESTAMP WITH TIME ZONE",
            "agent_id": "UUID",
            "swarm_id": "UUID",
            "event": "JSON",
        }

        session = await self.get_session()
        async with session as session:
            from sqlalchemy import update

            stmt = update(TableDefinition).where(TableDefinition.table_name == "chats").values(columns=correct_columns)
            await session.execute(stmt)
            await session.commit()
        self.logger.info("Forced update of chats table schema")
        self.clear_model_cache()

        # Verify the update worked
        updated_schema = await self.get_table_definition("chats")
        if updated_schema:
            self.logger.info("Updated schema verification completed")

    def _get_column_sqlalchemy_type(self, type_info: any, backend: str, column_name: str = None):
        """
        Convert database type information to appropriate SQLAlchemy type.
        Let the database schema define the types - don't guess!
        """
        # Extract type string from dict or use as is
        if isinstance(type_info, dict):
            type_str = type_info.get("type", "").upper()
        else:
            type_str = str(type_info).upper()

        # Handle UUID type explicitly declared in schema or when column name is "id" and type contains "uuid"
        if type_str in ("UUID", "UNIQUEIDENTIFIER") or (column_name == "id" and "uuid" in type_str.lower()):
            try:
                if backend == "postgresql":
                    from sqlalchemy.dialects.postgresql import UUID

                    # Create UUID type with proper configuration for asyncpg
                    uuid_type = UUID(as_uuid=True)

                    # Update the sqlalchemy_types dictionary for future use
                    self.sqlalchemy_types["UUID"] = uuid_type
                    return uuid_type
                else:
                    # SQLite: store UUIDs as strings with validation
                    self.sqlalchemy_types["UUID"] = String
                    return String
            except ImportError as e:
                self.logger.error(f"Failed to import UUID type for {backend}: {e}")
                # Fallback to String for UUID fields if import fails
                self.sqlalchemy_types["UUID"] = String
                return String
            except Exception as e:
                self.logger.error(f"Error creating UUID type for {backend}: {e}")
                # Fallback to String for UUID fields if creation fails
                self.sqlalchemy_types["UUID"] = String
                return String

        # Handle timestamp types - both TIMESTAMP and TIMESTAMP WITH TIME ZONE
        if type_str in ("TIMESTAMP", "TIMESTAMP WITH TIME ZONE") or (
            column_name == "timestamp" and type_str in ("TIMESTAMP", "String", "VARCHAR")
        ):
            from sqlalchemy import TIMESTAMP

            return TIMESTAMP(timezone=True)
        # Special case: if column name is "timestamp", always use TIMESTAMP regardless of stored type
        elif column_name == "timestamp":
            from sqlalchemy import TIMESTAMP

            return TIMESTAMP(timezone=True)
        elif type_str == "DATETIME":
            if backend == "postgresql":
                from sqlalchemy import TIMESTAMP

                return TIMESTAMP(timezone=True)
            else:
                return DateTime
        elif type_str in ("JSON", "JSONB"):
            if backend == "postgresql":
                from sqlalchemy.dialects.postgresql import JSONB

                return JSONB
            else:
                return Text
        elif type_str in ("TEXT", "STRING", "VARCHAR"):
            return String
        elif type_str in ("INTEGER", "INT", "BIGINT"):
            return Integer
        elif type_str in ("BOOLEAN", "BOOL"):
            return Boolean
        else:
            self.logger.error(
                f"Unsupported or unknown column type: {type_str} "
                f"(backend: {backend}, info: {type_info}) - falling back to String"
            )
            return String

    def _generate_model_class(self, table_name: str, columns: Dict[str, str]):
        # Create a unique class name based on table name and schema hash
        # This prevents SQLAlchemy registry conflicts when schema changes
        schema_str = json.dumps(columns, sort_keys=True)
        schema_hash = hashlib.md5(schema_str.encode()).hexdigest()[:8]
        unique_class_name = f"{table_name.capitalize()}_{schema_hash}"

        # Use the cache if available with the unique key
        cache_key = f"{table_name}_{schema_hash}"

        # FORCE CLEAR CACHE FOR TIMESTAMP ISSUE - TEMPORARY FIX
        if "timestamp" in columns:
            # Clear all cached models for this table to ensure fresh generation
            keys_to_remove = [key for key in self._model_cache.keys() if key.startswith(f"{table_name}_")]
            for key in keys_to_remove:
                del self._model_cache[key]

        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        else:
            self.logger.info(f"Generating new model for {table_name}")

        metadata = MetaData()
        columns_list: List[Column] = []
        backend = engine.url.get_backend_name()

        for name, column_type in columns.items():
            # Extract primary key info
            if isinstance(column_type, dict):
                is_pk = column_type.get("primary_key", False)
            else:
                is_pk = False

            # Use the centralized type mapping method
            sqlalchemy_type = self._get_column_sqlalchemy_type(column_type, backend, name)

            # Determine if column should be nullable (all columns except primary keys)
            is_nullable = not is_pk

            # Set default value for primary key columns
            default_value = None
            if is_pk:
                if backend == "postgresql" and str(sqlalchemy_type).find("UUID") != -1:
                    default_value = uuid.uuid4
                elif backend == "sqlite" and str(sqlalchemy_type).find("String") != -1:

                    def generate_uuid_string():
                        return str(uuid.uuid4())

                    default_value = generate_uuid_string

            columns_list.append(
                Column(name, sqlalchemy_type, primary_key=is_pk, nullable=is_nullable, default=default_value)
            )

        if "id" not in columns:
            pk_col = None
            for col_name, col_def in columns.items():
                if isinstance(col_def, dict) and col_def.get("primary_key"):
                    pk_col = col_name
                    break
            if pk_col:
                new_columns_list = []
                for col in columns_list:
                    if col.name == pk_col:
                        new_columns_list.append(Column(col.name, col.type, primary_key=True))
                    else:
                        new_columns_list.append(col)
                columns_list = new_columns_list
            else:
                if backend == "postgresql":
                    from sqlalchemy.dialects.postgresql import UUID

                    columns_list.insert(
                        0, Column("id", UUID(as_uuid=True), primary_key=True, nullable=False, default=uuid.uuid4)
                    )
                else:
                    columns_list.insert(
                        0, Column("id", String, primary_key=True, nullable=False, default=lambda: str(uuid.uuid4()))
                    )

        table = Table(table_name, metadata, *columns_list)

        # Create the model with unique class name
        model = type(
            unique_class_name,  # Use unique class name instead of table_name.capitalize()
            (Base,),
            {
                "__tablename__": table_name,
                "__table__": table,
                "__mapper_args__": {"eager_defaults": True},
            },
        )

        # Cache the model and metadata with unique key
        self._model_cache[cache_key] = (model, metadata)
        return model, metadata

    async def create_table(self, table_name: str, columns: Dict[str, str]):
        self.logger.info(f"Creating table '{table_name}' with columns: {columns}")
        try:
            if not isinstance(columns, dict):
                raise ValueError("Columns must be a dictionary")

                # Ensure we have a proper primary key definition
            columns_with_pk = columns.copy()

            # If no 'id' column exists and no other primary key is defined, add UUID primary key
            if "id" not in columns_with_pk:
                has_pk = any(
                    isinstance(col_def, dict) and col_def.get("primary_key", False)
                    for col_def in columns_with_pk.values()
                )
                if not has_pk:
                    # Add UUID primary key to the stored definition
                    columns_with_pk["id"] = {"type": "UUID", "primary_key": True}

            session = await self.get_session()
            async with session as session:
                table_definition = TableDefinition(table_name=table_name, columns=columns_with_pk)
                session.add(table_definition)
                await session.commit()
                self.logger.info(f"Table definition for '{table_name}' saved.")

                _, metadata = self._generate_model_class(table_name, columns_with_pk)
                async with engine.begin() as conn:
                    await conn.run_sync(metadata.create_all)
                self.logger.info(f"Table '{table_name}' created successfully.")
        except SQLAlchemyError as e:
            self.logger.error(f"Error creating table '{table_name}': {str(e)}")
            raise ValueError(f"Error creating table: {str(e)}")

    async def get_table_definition(self, table_name: str):
        self.logger.info(f"Retrieving table definition for '{table_name}'")
        try:
            session = await self.get_session()
            async with session as session:
                result = await session.execute(select(TableDefinition).filter_by(table_name=table_name))
                table_definition = result.scalars().first()
                if table_definition:
                    self.logger.info(f"Table definition for '{table_name}' retrieved successfully.")
                    return table_definition.columns
                self.logger.warning(f"Table definition for '{table_name}' not found.")
                return None
        except SQLAlchemyError as e:
            self.logger.error(f"Error retrieving table definition for '{table_name}': {str(e)}")
            raise ValueError(f"Error retrieving table definition: {str(e)}")

    async def insert_data(self, table_name: str, data: Dict[str, Any]):
        self.logger.info(f"Inserting data into '{table_name}'")
        try:
            columns = await self.get_table_definition(table_name)
            if not columns:
                raise ValueError(f"Table '{table_name}' does not exist.")

            processed_data = data.copy()
            backend = engine.url.get_backend_name()
            for key, value in data.items():
                col_type_info = columns.get(key, {})
                if isinstance(col_type_info, dict):
                    type_str = col_type_info.get("type", "").upper()
                else:
                    type_str = str(col_type_info).upper()

                # Handle explicit UUID types
                if type_str == "UUID":
                    if value is None:
                        # Keep None for UUID fields that allow NULL
                        processed_data[key] = None
                    elif isinstance(value, str):
                        try:
                            if backend == "postgresql":
                                processed_data[key] = uuid.UUID(value)
                            else:
                                uuid.UUID(value)  # Validate format
                                processed_data[key] = value
                        except ValueError:
                            self.logger.warning(f"Invalid UUID value '{value}' for field '{key}', keeping as string")
                            processed_data[key] = value
                    else:
                        # Keep non-string values as-is (e.g., already UUID objects)
                        processed_data[key] = value
                elif type_str in ("TIMESTAMP", "TIMESTAMP WITH TIME ZONE") and value is not None:
                    from datetime import datetime

                    if isinstance(value, str):
                        try:
                            # Support both Z and +00:00 timezone formats
                            converted = datetime.fromisoformat(value.replace('Z', '+00:00'))
                            processed_data[key] = converted
                        except ValueError:
                            self.logger.error(f"Failed to convert timestamp string '{value}'")
                            raise ValueError(f"Invalid timestamp format for {key}: {value}")
                    elif isinstance(value, datetime):
                        processed_data[key] = value
                    else:
                        self.logger.error(f"Invalid timestamp type: {type(value)} for value: {value}")
                        raise ValueError(
                            f"Timestamp value for {key} must be a datetime object or ISO8601 string, got {type(value)}"
                        )

            import json

            for col, col_type in columns.items():
                if isinstance(col_type, dict):
                    type_str = col_type.get("type", "")
                else:
                    type_str = col_type
                if type_str in ("JSON", "JSONB") and isinstance(processed_data.get(col), (dict, list)):
                    if backend == "sqlite":
                        processed_data[col] = json.dumps(processed_data[col])

            # FORCE direct SQL for chats table
            if table_name == "chats":
                return await self._insert_chats_direct_sql(processed_data)

            # Use cached model for other tables
            model, metadata = self._generate_model_class(table_name, columns)
            async with engine.begin() as conn:
                await conn.run_sync(metadata.create_all)
            session = await self.get_session()
            async with session as session:
                instance = model(**processed_data)
                session.add(instance)
                await session.commit()
                await session.refresh(instance)
                self.logger.info(f"Data inserted into '{table_name}' successfully, id: {instance.id}")
                return instance
        except SQLAlchemyError as e:
            self.logger.error(f"Error inserting data into '{table_name}': {str(e)}")
            # Clear the model cache to force regeneration on next attempt
            self.clear_model_cache()

            # If it's a timestamp error, force update the schema and clear cache again
            if "timestamp" in str(e) and "character varying" in str(e):
                self.logger.info("[DEBUG] Timestamp type mismatch detected, forcing schema update")
                try:
                    await self.force_update_chats_schema()
                except Exception as schema_error:
                    self.logger.error(f"Error updating schema: {schema_error}")

            raise ValueError(f"Error inserting data: {str(e)}")

    async def _insert_chats_direct_sql(self, data: Dict[str, Any]):
        """Insert data into chats table using direct SQL to avoid timestamp type issues."""
        import json

        from sqlalchemy import text

        # Generate UUID for id if not provided
        if 'id' not in data:
            data['id'] = str(uuid.uuid4())

        # Process data for direct SQL - handle JSON fields
        processed_data = {}
        for key, value in data.items():
            if key == 'event' and isinstance(value, (list, dict)):
                # Convert list/dict to JSON string for PostgreSQL
                processed_data[key] = json.dumps(value)
            else:
                processed_data[key] = value

        # Build the SQL query
        columns = list(processed_data.keys())
        placeholders = [f":{col}" for col in columns]

        sql = f"""
        INSERT INTO chats ({', '.join(columns)})
        VALUES ({', '.join(placeholders)})
        RETURNING id
        """

        session = await self.get_session()
        async with session as session:
            result = await session.execute(text(sql), processed_data)
            row = result.fetchone()
            await session.commit()

            return {"id": row[0]}

    async def read_data(
        self,
        table_name: str,
        filters: Optional[Dict[str, List[Any]]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ):
        self.logger.info(
            f"Reading data from '{table_name}' with filters: {filters}, "
            f"order_by: {order_by}, limit: {limit}, offset: {offset}"
        )
        try:
            columns = await self.get_table_definition(table_name)
            if not columns:
                raise ValueError(f"Table '{table_name}' does not exist.")

            # Special handling for chats table to avoid timestamp issues
            backend = engine.url.get_backend_name()
            if table_name == "chats" and backend == "postgresql":
                return await self._read_chats_direct_sql(filters, order_by, limit, offset)

            # Use cached model for other tables
            model, metadata = self._generate_model_class(table_name, columns)
            async with engine.begin() as conn:
                await conn.run_sync(metadata.create_all)
            query = select(model)

            if filters:
                for key, values in filters.items():
                    if key == "details":
                        for value in values:
                            for sub_key, sub_value in value.items():
                                query = query.where(getattr(model, key)[sub_key] == sub_value)
                    else:
                        column = getattr(model, key)
                        backend = engine.url.get_backend_name()

                        # Get column type from schema
                        col_type_info = columns.get(key, {})
                        if isinstance(col_type_info, dict):
                            type_str = col_type_info.get("type", "").upper()
                        else:
                            type_str = str(col_type_info).upper()

                        # Handle explicit UUID types
                        if type_str == "UUID":
                            if backend == "postgresql":
                                from sqlalchemy import cast
                                from sqlalchemy.dialects.postgresql import UUID

                                uuid_values = []
                                for value in values:
                                    try:
                                        if isinstance(value, str):
                                            uuid.UUID(value)
                                        uuid_values.append(cast(value, UUID))
                                    except ValueError:
                                        self.logger.warning(f"Invalid UUID value '{value}' for field '{key}', skipping")
                                        continue
                                if uuid_values:
                                    query = query.filter(column.in_(uuid_values))
                            else:
                                # SQLite: UUIDs as strings
                                valid_values = []
                                for value in values:
                                    if isinstance(value, str):
                                        try:
                                            uuid.UUID(value)  # Validate format
                                            valid_values.append(value)
                                        except ValueError:
                                            self.logger.warning(
                                                f"Invalid UUID value '{value}' for field '{key}', skipping"
                                            )
                                            continue
                                    else:
                                        valid_values.append(value)
                                if valid_values:
                                    query = query.filter(column.in_(valid_values))
                        elif isinstance(column.type, String) and len(values) == 1:
                            # Only use ILIKE for actual string fields, not UUID fields
                            query = query.where(column.ilike(f"%{values[0]}%"))
                        else:
                            query = query.filter(column.in_(values))

            if order_by:
                if order_by.startswith("-"):
                    query = query.order_by(desc(getattr(model, order_by[1:])))
                else:
                    query = query.order_by(asc(getattr(model, order_by)))

            if offset is not None:
                query = query.offset(offset)
            if limit is not None:
                query = query.limit(limit)

            session = await self.get_session()
            async with session as session:
                result = await session.execute(query)
                instances = result.scalars().all()
                data = [{column: getattr(instance, column) for column in columns.keys()} for instance in instances]

            self.logger.info(f"Data read from '{table_name}' successfully.")
            return data

        except SQLAlchemyError as e:
            self.logger.error(f"Error reading data from '{table_name}': {str(e)}")
            raise ValueError(f"Error reading data: {str(e)}")

    async def _read_chats_direct_sql(
        self,
        filters: Optional[Dict[str, List[Any]]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ):
        """Read data from chats table using direct SQL to avoid timestamp type issues."""
        from sqlalchemy import text

        # Build the SQL query
        sql = "SELECT * FROM chats"
        params = {}

        # Add WHERE clause if filters provided
        if filters:
            conditions = []
            for key, values in filters.items():
                if len(values) == 1:
                    conditions.append(f"{key} = :{key}")
                    params[key] = values[0]
                else:
                    placeholders = [f":{key}_{i}" for i in range(len(values))]
                    conditions.append(f"{key} IN ({', '.join(placeholders)})")
                    for i, value in enumerate(values):
                        params[f"{key}_{i}"] = value

            if conditions:
                sql += " WHERE " + " AND ".join(conditions)

        # Add ORDER BY
        if order_by:
            if order_by.startswith("-"):
                sql += f" ORDER BY {order_by[1:]} DESC"
            else:
                sql += f" ORDER BY {order_by}"

        # Add LIMIT and OFFSET
        if limit is not None:
            sql += f" LIMIT {limit}"
        if offset is not None:
            sql += f" OFFSET {offset}"

        session = await self.get_session()
        async with session as session:
            result = await session.execute(text(sql), params)
            rows = result.fetchall()

            # Convert to list of dictionaries
            columns = result.keys()
            data = [dict(zip(columns, row)) for row in rows]

            return data

    async def update_data(self, table_name: str, row_id: Any, new_data: Dict[str, Any]):
        self.logger.info(f"Updating data in '{table_name}' for id {row_id} with new data: {new_data}")
        try:
            columns = await self.get_table_definition(table_name)
            if not columns:
                raise ValueError(f"Table '{table_name}' does not exist.")
            # Use cached model
            model, metadata = self._generate_model_class(table_name, columns)
            async with engine.begin() as conn:
                await conn.run_sync(metadata.create_all)

            # Convert row_id based on backend for UUID types only
            backend = engine.url.get_backend_name()
            if isinstance(row_id, str):
                try:
                    if backend == "postgresql":
                        row_id = uuid.UUID(row_id)
                    else:
                        uuid.UUID(row_id)  # Validate format for SQLite
                except ValueError:
                    pass  # Keep as string if not valid UUID

            # Process data based on schema definitions only
            processed_data = new_data.copy()
            for key, value in new_data.items():
                col_type_info = columns.get(key, {})
                if isinstance(col_type_info, dict):
                    type_str = col_type_info.get("type", "").upper()
                else:
                    type_str = str(col_type_info).upper()

                # Only handle explicit UUID types
                if type_str == "UUID" and value is not None and isinstance(value, str):
                    try:
                        if backend == "postgresql":
                            processed_data[key] = uuid.UUID(value)
                        else:
                            uuid.UUID(value)  # Validate format
                            processed_data[key] = value
                    except ValueError:
                        pass  # Keep as string if not valid UUID
                elif type_str == "TIMESTAMP" and value is not None:
                    from datetime import datetime

                    if isinstance(value, str):
                        try:
                            # Support both Z and +00:00 timezone formats
                            converted = datetime.fromisoformat(value.replace('Z', '+00:00'))
                            processed_data[key] = converted
                        except ValueError:
                            self.logger.error(f"Failed to convert timestamp string '{value}'")
                            raise ValueError(f"Invalid timestamp format for {key}: {value}")
                    elif isinstance(value, datetime):
                        processed_data[key] = value
                    else:
                        self.logger.error(f"Invalid timestamp type: {type(value)} for value: {value}")
                        raise ValueError(
                            f"Timestamp value for {key} must be a datetime object or ISO8601 string, got {type(value)}"
                        )

            instance = await self.db.get(model, row_id)
            if instance:
                for key, value in processed_data.items():
                    setattr(instance, key, value)
                await self.db.commit()
                await self.db.refresh(instance)
                self.logger.info(f"Data in '{table_name}' for id {row_id} updated successfully.")
            else:
                self.logger.warning(f"No data found with id {row_id} in '{table_name}'.")
                raise ValueError(f"No data found with id {row_id} in '{table_name}'.")
        except SQLAlchemyError as e:
            self.logger.error(f"Error updating data in '{table_name}' for id {row_id}: {str(e)}")
            raise ValueError(f"Error updating data: {str(e)}")

    async def delete_data(self, table_name: str, row_id: Any):
        self.logger.info(f"Deleting data from '{table_name}' for id {row_id}")
        try:
            columns = await self.get_table_definition(table_name)
            if not columns:
                raise ValueError(f"Table '{table_name}' does not exist.")
            # Use cached model
            model, metadata = self._generate_model_class(table_name, columns)
            async with engine.begin() as conn:
                await conn.run_sync(metadata.create_all)

            # Convert row_id based on backend
            backend = engine.url.get_backend_name()
            if isinstance(row_id, str):
                try:
                    if backend == "postgresql":
                        # PostgreSQL: convert to UUID object
                        row_id = uuid.UUID(row_id)
                    else:
                        # SQLite: validate UUID format but keep as string
                        uuid.UUID(row_id)  # Validate format
                        # Keep as string for SQLite
                except ValueError:
                    pass  # Keep as string if not valid UUID

            instance = await self.db.get(model, row_id)
            if instance:
                await self.db.delete(instance)
                await self.db.commit()
                self.logger.info(f"Data deleted from '{table_name}' for id {row_id} successfully.")
            else:
                self.logger.warning(f"No data found with id {row_id} in '{table_name}'.")
                raise ValueError(f"No data found with id {row_id} in '{table_name}'.")
        except SQLAlchemyError as e:
            self.logger.error(f"Error deleting data from '{table_name}' for id {row_id}: {str(e)}")
            raise ValueError(f"Error deleting data: {str(e)}")
