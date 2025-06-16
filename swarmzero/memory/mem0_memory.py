import os
from typing import Any, Dict, Optional

from llama_index.memory.mem0 import Mem0Memory


def create_mem0_memory(
    api_key: Optional[str] = None,
    host: Optional[str] = None,
    org_id: Optional[str] = None,
    project_id: Optional[str] = None,
    search_msg_limit: int = 5,
    **kwargs: Any,
) -> Mem0Memory:
    """Create a ``Mem0Memory`` instance from environment variables or arguments."""
    api_key = api_key or os.getenv("MEM0_API_KEY")
    host = host or os.getenv("MEM0_HOST")
    org_id = org_id or os.getenv("MEM0_ORG_ID")
    project_id = project_id or os.getenv("MEM0_PROJECT_ID")

    context: Dict[str, Any] = {
        "org_id": org_id or "",
        "project_id": project_id or "",
    }

    memory = Mem0Memory.from_client(
        context=context,
        api_key=api_key,
        host=host,
        org_id=org_id,
        project_id=project_id,
        search_msg_limit=search_msg_limit,
        **kwargs,
    )
    return memory
