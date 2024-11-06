import unittest
from unittest.mock import MagicMock

from sqlalchemy import Boolean, Column, Integer, MetaData, String, Table, create_engine

from swarmzero.core.sdk_context import SDKContext
from swarmzero.tools.agent_db.schema import get_db_schemas


class TestGetDBSchemas(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        self.metadata = MetaData()

        self.users_table = Table(
            "users",
            self.metadata,
            Column("id", Integer, primary_key=True),
            Column("name", String(50), nullable=False),
            Column("email", String(100), nullable=True),
            Column("is_active", Boolean, nullable=False, default=True),
        )
        self.metadata.create_all(self.engine)

        self.sdk_context = MagicMock(spec=SDKContext)
        self.sdk_context.get_utility.side_effect = lambda x: self.engine if x == 'text2sql_engine' else self.metadata

    def tearDown(self):
        self.engine.dispose()

    def test_get_db_schemas(self):
        schemas = get_db_schemas(self.sdk_context)

        expected_schema = [
            {"name": "id", "type": "INTEGER", "primary_key": True, "nullable": False},
            {
                "name": "name",
                "type": "VARCHAR(50)",
                "primary_key": False,
                "nullable": False,
            },
            {
                "name": "email",
                "type": "VARCHAR(100)",
                "primary_key": False,
                "nullable": True,
            },
            {
                "name": "is_active",
                "type": "BOOLEAN",
                "primary_key": False,
                "nullable": False,
            },
        ]

        self.assertIn("users", schemas)
        self.assertListEqual(schemas["users"], expected_schema)

    def test_empty_database(self):
        empty_metadata = MetaData()
        self.sdk_context.get_utility.side_effect = lambda x: (
            create_engine("sqlite:///:memory:") if x == 'text2sql_engine' else empty_metadata
        )
        schemas = get_db_schemas(self.sdk_context)
        self.assertEqual(schemas, {})

    def test_multiple_tables(self):
        products_table = Table(
            "products",
            self.metadata,
            Column("id", Integer, primary_key=True),
            Column("name", String(100), nullable=False),
            Column("price", Integer, nullable=False),
        )
        self.metadata.create_all(self.engine)

        schemas = get_db_schemas(self.sdk_context)

        self.assertIn("users", schemas)
        self.assertIn("products", schemas)
        self.assertEqual(len(schemas["products"]), 3)

    def test_invalid_engine(self):
        self.sdk_context.get_utility.side_effect = Exception("Invalid engine")

        with self.assertRaises(Exception) as context:
            get_db_schemas(self.sdk_context)

        self.assertTrue("Invalid engine" in str(context.exception))
