import unittest
from unittest.mock import AsyncMock, patch

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.declarative import declarative_base

from swarmzero.database.database import DatabaseManager

Base = declarative_base()


class TestDatabaseManager(unittest.IsolatedAsyncioTestCase):
    @patch("swarmzero.database.database.DatabaseManager.__init__")
    async def test_create_table(self, mock_init):
        mock_session = AsyncMock(spec=AsyncSession)
        mock_init.return_value = None
        db_manager = DatabaseManager(mock_session)
        db_manager.db = mock_session

        # Test valid case
        columns = {"name": "String", "age": "Integer"}
        await db_manager.create_table("users", columns)
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

        # Test invalid columns
        with self.assertRaises(ValueError):
            await db_manager.create_table("users", "invalid_columns")

    @patch("swarmzero.database.database.DatabaseManager.__init__")
    async def test_get_table_definition(self, mock_init):
        # mock_session = AsyncMock(spec=AsyncSession)
        # mock_init.return_value = None
        # db_manager = DatabaseManager(mock_session)
        # db_manager.db = mock_session
        #
        # # Test existing table
        # table_definition = json.dumps({"name": "String", "age": "Integer"})
        # mock_scalar = AsyncMock()
        # mock_scalar.first = AsyncMock(return_value=AsyncMock(columns=table_definition))
        # mock_result = AsyncMock()
        # mock_result.scalars = AsyncMock(return_value=mock_scalar)
        # db_manager.db.execute = AsyncMock(return_value=mock_result)
        # result = await db_manager.get_table_definition("users")
        # self.assertEqual(result, table_definition)
        #
        # # Test non-existing table
        # mock_scalar.first.return_value = None
        # result = await db_manager.get_table_definition("non_existing")
        # self.assertIsNone(result)

        pass

    @patch("swarmzero.database.database.DatabaseManager.__init__")
    async def test_insert_data(self, mock_init):
        mock_session = AsyncMock(spec=AsyncSession)
        mock_init.return_value = None
        db_manager = DatabaseManager(mock_session)
        db_manager.db = mock_session

        # Test valid case
        table_definition = {"name": "String", "age": "Integer"}
        db_manager.get_table_definition = AsyncMock(return_value=table_definition)
        data = {"name": "John", "age": 30}
        await db_manager.insert_data("users", data)
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

        # Test non-existing table
        db_manager.get_table_definition.return_value = None
        with self.assertRaises(ValueError):
            await db_manager.insert_data("non_existing", data)

    @patch("swarmzero.database.database.DatabaseManager.__init__")
    async def test_read_data(self, mock_init):
        # mock_session = AsyncMock(spec=AsyncSession)
        # mock_init.return_value = None
        # db_manager = DatabaseManager(mock_session)
        # db_manager.db = mock_session
        #
        # # Mock the result of the query
        # mock_instance = AsyncMock(name="John", age=30)
        # mock_scalar = AsyncMock()
        # mock_scalar.all.return_value = [mock_instance]
        # mock_result = AsyncMock()
        # mock_result.scalars.return_value = mock_scalar
        #
        # # Set the return value of the execute method to be the mock_result
        # db_manager.db.execute.return_value = mock_result
        #
        # # Test valid case
        # table_definition = json.dumps({"name": "String", "age": "Integer"})
        # db_manager.get_table_definition = AsyncMock(return_value=table_definition)
        # data = await db_manager.read_data("users")
        # self.assertEqual(data, [{"name": "John", "age": 30}])
        #
        # # Test non-existing table
        # db_manager.get_table_definition.return_value = None
        # with self.assertRaises(ValueError):
        #     await db_manager.read_data("non_existing")

        # # Test filters
        # filters = {"name": ["John"], "details": [{"age": 30}]}
        # await db_manager.read_data("users", filters)
        # mock_session.execute.assert_called()

        pass

    @patch("swarmzero.database.database.DatabaseManager.__init__")
    async def test_update_data(self, mock_init):
        mock_session = AsyncMock(spec=AsyncSession)
        mock_init.return_value = None
        db_manager = DatabaseManager(mock_session)
        db_manager.db = mock_session

        # Test valid case
        table_definition = {"name": "String", "age": "Integer"}
        db_manager.get_table_definition = AsyncMock(return_value=table_definition)
        mock_instance = AsyncMock(name="John", age=30)
        db_manager.db.get.return_value = mock_instance
        new_data = {"name": "Jane", "age": 35}
        await db_manager.update_data("users", 1, new_data)
        mock_instance.name = "Jane"
        mock_instance.age = 35
        mock_session.commit.assert_called_once()

        # Test non-existing table
        db_manager.get_table_definition.return_value = None
        with self.assertRaises(ValueError):
            await db_manager.update_data("non_existing", 1, new_data)

        # Test non-existing row
        db_manager.db.get.return_value = None
        with self.assertRaises(ValueError):
            await db_manager.update_data("users", 2, new_data)

    @patch("swarmzero.database.database.DatabaseManager.__init__")
    async def test_delete_data(self, mock_init):
        mock_session = AsyncMock(spec=AsyncSession)
        mock_init.return_value = None
        db_manager = DatabaseManager(mock_session)
        db_manager.db = mock_session

        # Test valid case
        table_definition = {"name": "String", "age": "Integer"}
        db_manager.get_table_definition = AsyncMock(return_value=table_definition)
        mock_instance = AsyncMock()
        db_manager.db.get.return_value = mock_instance
        await db_manager.delete_data("users", 1)
        mock_session.delete.assert_called_once_with(mock_instance)
        mock_session.commit.assert_called_once()

        # Test non-existing table
        db_manager.get_table_definition.return_value = None
        with self.assertRaises(ValueError):
            await db_manager.delete_data("non_existing", 1)

        # Test non-existing row
        db_manager.db.get.return_value = None
        with self.assertRaises(ValueError):
            await db_manager.delete_data("users", 2)
