import os
import unittest
from unittest.mock import MagicMock, patch

from dotenv import load_dotenv
from llama_index.core.query_engine import NLSQLTableQueryEngine
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine
from sqlalchemy.exc import ArgumentError

from swarmzero.sdk_context import SDKContext
from swarmzero.tools.agent_db.nl_query import text_2_sql

os.environ["OPENAI_API_KEY"] = "test_key"
os.environ["MODEL"] = "gpt-3.5-turbo"

load_dotenv()

OpenAI = MagicMock()


@patch.dict(os.environ, {"OPENAI_API_KEY": "test_key", "MODEL": "gpt-3.5-turbo"})
class TestText2SQL(unittest.TestCase):
    def setUp(self):
        self.db_url = "sqlite:///:memory:"
        self.engine = create_engine(self.db_url)
        self.connection = self.engine.connect()
        self.metadata = MetaData()

        self.entries = Table(
            "entries",
            self.metadata,
            Column("id", Integer, primary_key=True),
            Column("name", String),
            Column("age", Integer),
        )
        self.chats = Table(
            "chats",
            self.metadata,
            Column("id", Integer, primary_key=True),
            Column("user_id", String),
            Column("message", String),
        )
        self.metadata.create_all(self.engine)

        self.connection.execute(
            self.entries.insert(),
            [
                {"name": "Alice", "age": 25},
                {"name": "Bob", "age": 32},
                {"name": "Charlie", "age": 19},
            ],
        )

        self.sdk_context = MagicMock(spec=SDKContext)
        self.sdk_context.get_utility.return_value = self.engine

    def tearDown(self):
        self.connection.close()
        self.engine.dispose()

    @patch.object(NLSQLTableQueryEngine, "query")
    def test_text_2_sql_success(self, mock_query):
        mock_query.return_value.response = "Mock LLM response"

        prompt = "What are the names of people in the entries table?"
        response = text_2_sql(self.sdk_context, prompt, tables=["entries"])
        self.assertEqual(response, "Mock LLM response")

        prompt = "What are the names of people older than 20 in the entries table?"
        response = text_2_sql(self.sdk_context, prompt, tables=["entries"])
        self.assertEqual(response, "Mock LLM response")

        prompt = "What are the ages of people in the entries table?"
        response = text_2_sql(self.sdk_context, prompt, tables=["entries"])
        self.assertEqual(response, "Mock LLM response")

    def test_invalid_url(self):
        self.sdk_context.get_utility.side_effect = ArgumentError("Invalid URL")

        with self.assertRaises(ArgumentError) as context:
            text_2_sql(self.sdk_context, "What are the names of people in the entries table?")

        self.assertTrue("Invalid URL" in str(context.exception))

    def test_nonexistent_table(self):
        with self.assertRaises(ValueError) as context:
            text_2_sql(
                self.sdk_context,
                "What are the names of people in the non_existent_table?",
                tables=["non_existent_table"],
            )

        self.assertTrue("include_tables {'non_existent_table'} not found in database" in str(context.exception))

    @patch.object(NLSQLTableQueryEngine, "query")
    def test_invalid_prompt(self, mock_query):
        mock_query.side_effect = ValueError("Could not generate SQL query from prompt")

        with self.assertRaises(ValueError) as context:
            text_2_sql(self.sdk_context, "This is not a valid prompt", tables=["entries"])

        self.assertTrue("Could not generate SQL query from prompt" in str(context.exception))

    @patch.object(NLSQLTableQueryEngine, "query")
    def test_malformed_parameters(self, mock_query):
        mock_query.side_effect = TypeError("Bad input")

        with self.assertRaises(TypeError) as context:
            text_2_sql(self.sdk_context, 12345, tables=["entries"])

        self.assertTrue("Bad input" in str(context.exception))
