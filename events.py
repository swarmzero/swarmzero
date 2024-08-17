from typing import Callable, List, Optional

from llama_index.core.workflow import Context, Event, StartEvent, StopEvent, Workflow, step


class InitializeEvent(Event):
    pass


class OrchestratorEvent(Event):
    request: Optional[str] = None
    just_completed: Optional[str] = None
    need_help: Optional[bool] = None


class ChooseAgentEvent(Event):
    request: str
    agent_name: str
