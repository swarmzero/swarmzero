import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from fastapi.routing import APIRouter
from sqlalchemy.ext.asyncio import AsyncSession

from swarmzero.api.routes.database import setup_database_routes
from swarmzero.core.services.database import (
    DataDelete,
    DataInsert,
    DataRead,
    DataUpdate,
    TableCreate,
)


class TestDatabaseRoutes(unittest.IsolatedAsyncioTestCase):

    @patch("swarmzero.server.routes.database.DatabaseManager")
    async def test_create_table_handler(self, mock_manager):
        mock_db = AsyncMock(spec=AsyncSession)
        mock_manager_instance = mock_manager.return_value
        mock_manager_instance.create_table = AsyncMock()

        router = APIRouter()
        setup_database_routes(router)

        table_create = TableCreate(table_name="test_table", columns={"col1": "String", "col2": "Integer"})
        result = await router.routes[0].endpoint(table_create, mock_db)
        self.assertEqual(result, {"message": "Table test_table created successfully."})

        mock_manager_instance.create_table.side_effect = ValueError("Test error")
        with self.assertRaises(HTTPException) as cm:
            await router.routes[0].endpoint(table_create, mock_db)
        self.assertEqual(cm.exception.status_code, 400)
        self.assertEqual(cm.exception.detail, "Test error")

    @patch("swarmzero.server.routes.database.DatabaseManager")
    async def test_insert_data_handler(self, mock_manager):
        mock_db = AsyncMock(spec=AsyncSession)
        mock_instance = AsyncMock(id=1)
        mock_manager_instance = mock_manager.return_value
        mock_manager_instance.insert_data = AsyncMock(return_value=mock_instance)

        router = APIRouter()
        setup_database_routes(router)

        data_insert = DataInsert(table_name="test_table", data={"col1": "value1", "col2": 2})
        result = await router.routes[1].endpoint(data_insert, mock_db)
        self.assertEqual(result, {"message": "Data inserted successfully.", "id": 1})

        mock_manager_instance.insert_data.side_effect = ValueError("Test error")
        with self.assertRaises(HTTPException) as cm:
            await router.routes[1].endpoint(data_insert, mock_db)
        self.assertEqual(cm.exception.status_code, 404)
        self.assertEqual(cm.exception.detail, "Test error")

    @patch("swarmzero.server.routes.database.DatabaseManager")
    async def test_read_data_handler(self, mock_manager):
        mock_db = AsyncMock(spec=AsyncSession)
        mock_data = [{"col1": "value1", "col2": 2}]
        mock_manager_instance = mock_manager.return_value
        mock_manager_instance.read_data = AsyncMock(return_value=mock_data)

        router = APIRouter()
        setup_database_routes(router)

        data_read = DataRead(table_name="test_table", filters=None)
        result = await router.routes[2].endpoint(data_read, mock_db)
        self.assertEqual(result, mock_data)

        mock_manager_instance.read_data.side_effect = ValueError("Test error")
        with self.assertRaises(HTTPException) as cm:
            await router.routes[2].endpoint(data_read, mock_db)
        self.assertEqual(cm.exception.status_code, 404)
        self.assertEqual(cm.exception.detail, "Test error")

    @patch("swarmzero.server.routes.database.DatabaseManager")
    async def test_update_data_handler(self, mock_manager):
        mock_db = AsyncMock(spec=AsyncSession)
        mock_manager_instance = mock_manager.return_value
        mock_manager_instance.update_data = AsyncMock()

        router = APIRouter()
        setup_database_routes(router)

        data_update = DataUpdate(table_name="test_table", id=1, data={"col1": "new_value"})
        result = await router.routes[3].endpoint(data_update, mock_db)
        self.assertEqual(result, {"message": "Data updated successfully."})

        mock_manager_instance.update_data.side_effect = ValueError("Test error")
        with self.assertRaises(HTTPException) as cm:
            await router.routes[3].endpoint(data_update, mock_db)
        self.assertEqual(cm.exception.status_code, 404)
        self.assertEqual(cm.exception.detail, "Test error")

    @patch("swarmzero.server.routes.database.DatabaseManager")
    async def test_delete_data_handler(self, mock_manager):
        mock_db = AsyncMock(spec=AsyncSession)
        mock_manager_instance = mock_manager.return_value
        mock_manager_instance.delete_data = AsyncMock()

        router = APIRouter()
        setup_database_routes(router)

        data_delete = DataDelete(table_name="test_table", id=1)
        result = await router.routes[4].endpoint(data_delete, mock_db)
        self.assertEqual(result, {"message": "Data deleted successfully."})

        mock_manager_instance.delete_data.side_effect = ValueError("Test error")
        with self.assertRaises(HTTPException) as cm:
            await router.routes[4].endpoint(data_delete, mock_db)
        self.assertEqual(cm.exception.status_code, 404)
        self.assertEqual(cm.exception.detail, "Test error")
