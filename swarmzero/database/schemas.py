from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class TableCreate(BaseModel):
    table_name: str
    columns: Dict[str, str]


class DataInsert(BaseModel):
    table_name: str
    data: Dict[str, Any]


class DataRead(BaseModel):
    table_name: str
    filters: Optional[Dict[str, List[Any]]] = None


class DataUpdate(BaseModel):
    table_name: str
    id: Union[int, str]  # Support both integer and UUID/string IDs
    data: Dict[str, Any]


class DataDelete(BaseModel):
    table_name: str
    id: Union[int, str]  # Support both integer and UUID/string IDs
