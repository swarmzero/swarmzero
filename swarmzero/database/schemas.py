from typing import Any, Dict, List, Optional

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
    id: int
    data: Dict[str, Any]


class DataDelete(BaseModel):
    table_name: str
    id: int
