from typing import Dict, List, Optional

from pydantic import BaseModel


class ToolInstallRequest(BaseModel):
    github_url: str
    functions: List[str]
    install_path: Optional[str]
    github_token: Optional[str] = None
    env_vars: Optional[Dict[str, str]] = None
