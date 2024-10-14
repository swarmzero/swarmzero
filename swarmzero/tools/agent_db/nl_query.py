import os

from dotenv import load_dotenv
from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.settings import Settings

from swarmzero.sdk_context import SDKContext

load_dotenv()


def text_2_sql(sdkcontext: SDKContext, prompt: str, tables=None):
    """
    Convert a natural language prompt into an SQL query and execute it against the specified database.

    :param prompt: The natural language query to convert into SQL.
    :param tables: Optional list of tables to include in the query. Defaults to ["entries"] if not specified.
    :return: The response from the query execution.
    """

    if tables is None:
        tables = ["chats"]

    engine = sdkcontext.get_utility('text2sql_engine')
    sql_database = SQLDatabase(engine, include_tables=tables)

    query_engine = NLSQLTableQueryEngine(sql_database=sql_database, tables=tables, llm=Settings.llm)

    answer = query_engine.query(prompt)

    return answer.response
