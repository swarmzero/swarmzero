from sqlalchemy import MetaData, create_engine

from swarmzero.sdk_context import SDKContext


def get_db_schemas(sdkcontext: SDKContext):
    """
    Connect to the database specified by db_url, reflect its schema,
    and return a dictionary mapping table names to their schema definitions.

    :param db_url: The database URL connection string.
    :return: A dictionary where each key is a table name and the value is a list of dictionaries
             containing column details such as name, type, primary key status, and nullable status.
    """

    engine = sdkcontext.get_utility('text2sql_engine')
    metadata = sdkcontext.get_utility('text2sql_metadata')

    schemas = {}
    for table_name, table in metadata.tables.items():
        columns = []
        for column in table.columns:
            columns.append(
                {
                    "name": column.name,
                    "type": str(column.type),
                    "primary_key": column.primary_key,
                    "nullable": column.nullable,
                }
            )
        schemas[table_name] = columns
    return schemas
