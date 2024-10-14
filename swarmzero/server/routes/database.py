import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from swarmzero.database.database import DatabaseManager, get_db
from swarmzero.database.schemas import (
    DataDelete,
    DataInsert,
    DataRead,
    DataUpdate,
    TableCreate,
)

# TODO: get log level from config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_database_routes(router: APIRouter):
    @router.post("/database/create-table", response_model=Dict[str, str])
    async def create_table_handler(table: TableCreate, db: AsyncSession = Depends(get_db)):
        logger.info(f"Received request to create table: {table.table_name}")
        db_manager = DatabaseManager(db)
        try:
            await db_manager.create_table(table.table_name, table.columns)
            logger.info(f"Table {table.table_name} created successfully.")
            return {"message": f"Table {table.table_name} created successfully."}
        except ValueError as e:
            logger.error(f"ValueError: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Exception: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")

    @router.post("/database/insert-data", response_model=Dict[str, Any])
    async def insert_data_handler(data: DataInsert, db: AsyncSession = Depends(get_db)):
        logger.info(f"Received request to insert data into table: {data.table_name}")
        db_manager = DatabaseManager(db)
        try:
            instance = await db_manager.insert_data(data.table_name, data.data)
            logger.info(f"Data inserted successfully into table {data.table_name}, id: {instance.id}")
            return {"message": "Data inserted successfully.", "id": instance.id}
        except ValueError as e:
            logger.error(f"ValueError: {str(e)}")
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Exception: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")

    @router.post("/database/read-data", response_model=List[Dict[str, Any]])
    async def read_data_handler(data: DataRead, db: AsyncSession = Depends(get_db)):
        logger.info(f"Received request to read data from table: {data.table_name}")
        db_manager = DatabaseManager(db)
        try:
            result = await db_manager.read_data(data.table_name, data.filters)
            logger.info(f"Data read successfully from table {data.table_name}")
            return result
        except ValueError as e:
            logger.error(f"ValueError: {str(e)}")
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Exception: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")

    @router.put("/database/update-data", response_model=Dict[str, str])
    async def update_data_handler(data: DataUpdate, db: AsyncSession = Depends(get_db)):
        logger.info(f"Received request to update data in table: {data.table_name}, id: {data.id}")
        db_manager = DatabaseManager(db)
        try:
            await db_manager.update_data(data.table_name, data.id, data.data)
            logger.info(f"Data updated successfully in table {data.table_name}, id: {data.id}")
            return {"message": "Data updated successfully."}
        except ValueError as e:
            logger.error(f"ValueError: {str(e)}")
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Exception: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")

    @router.delete("/database/delete-data", response_model=Dict[str, str])
    async def delete_data_handler(data: DataDelete, db: AsyncSession = Depends(get_db)):
        logger.info(f"Received request to delete data from table: {data.table_name}, id: {data.id}")
        db_manager = DatabaseManager(db)
        try:
            await db_manager.delete_data(data.table_name, data.id)
            logger.info(f"Data deleted successfully from table {data.table_name}, id: {data.id}")
            return {"message": "Data deleted successfully."}
        except ValueError as e:
            logger.error(f"ValueError: {str(e)}")
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Exception: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")
