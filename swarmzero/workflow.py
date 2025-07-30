from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Iterable, List, Optional

import inspect
from swarmzero.llms.llm import LLM
from swarmzero.sdk_context import SDKContext
from swarmzero.agent import Agent
from swarmzero.swarm import Swarm


class StepMode(Enum):
    SEQUENTIAL = auto()
    PARALLEL = auto()
    CONDITIONAL = auto()
    LOOP = auto()


@dataclass
class WorkflowStep:
    runner: Any
    mode: StepMode = StepMode.SEQUENTIAL
    condition: Optional[Callable[[Any], bool]] = None
    max_iterations: Optional[int] = 1
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    llm: Optional[LLM] = None
    sdk_context: Optional[SDKContext] = None
    name: str = ""


class Workflow:
    def __init__(
        self,
        name: str,
        steps: List[WorkflowStep],
        instruction: str = "",
        description: str = "",
        default_llm: Optional[LLM] = None,
        sdk_context: Optional[SDKContext] = None,
        default_user_id: str = "default_user",
        default_session_id: str = "default_chat",
        workflow_id: Optional[str] = None,
    ) -> None:
        self.id = workflow_id or str(uuid.uuid4())
        self.name = name
        self.instruction = instruction
        self.description = description
        self.default_llm = default_llm
        self.sdk_context = sdk_context or SDKContext.get_instance()
        self.default_user_id = default_user_id
        self.default_session_id = default_session_id
        self.steps = steps
        self.sdk_context.add_resource(self, resource_type="workflow")

    async def _execute_runner(
        self,
        runner: Any,
        prompt: Any,
        user_id: str,
        session_id: str,
        llm: Optional[LLM],
        sdk_context: SDKContext,
    ) -> Any:
        # Agent
        if isinstance(runner, Agent):
            result = runner.chat(
                prompt,
                user_id=user_id,
                session_id=session_id,
            )
        # Swarm
        elif isinstance(runner, Swarm):
            result = runner.chat(
                prompt,
                user_id=user_id,
                session_id=session_id,
            )
        # Workflow
        elif isinstance(runner, Workflow):
            result = runner.run(prompt)
        # Callable function
        elif callable(runner):
            sig = inspect.signature(runner)
            kwargs = {}
            for name, param in sig.parameters.items():
                if name == 'prompt':
                    kwargs['prompt'] = prompt
                elif name == 'user_id':
                    kwargs['user_id'] = user_id
                elif name == 'session_id':
                    kwargs['session_id'] = session_id
                elif name == 'llm':
                    kwargs['llm'] = llm
                elif name == 'sdk_context':
                    kwargs['sdk_context'] = sdk_context
            result = runner(**kwargs)
        else:
            raise ValueError(f"Unsupported runner type: {type(runner)}")
        if asyncio.iscoroutine(result):
            return await result
        return result

    async def run(self, initial_prompt: Any) -> Any:
        result = initial_prompt
        for step in self.steps:
            user_id = step.user_id or self.default_user_id
            session_id = step.session_id or self.default_session_id
            llm = step.llm or self.default_llm
            sdk_context = step.sdk_context or self.sdk_context

            if step.mode == StepMode.SEQUENTIAL:
                result = await self._execute_runner(step.runner, result, user_id, session_id, llm, sdk_context)

            elif step.mode == StepMode.PARALLEL:
                # Handle both single runner and list of runners for parallel execution
                if isinstance(step.runner, list):
                    runners = step.runner
                elif hasattr(step.runner, '__iter__') and not isinstance(step.runner, (str, bytes)):
                    runners = list(step.runner)
                else:
                    runners = [step.runner]
                
                results = await asyncio.gather(
                    *[self._execute_runner(r, result, user_id, session_id, llm, sdk_context) for r in runners]
                )
                result = results

            elif step.mode == StepMode.CONDITIONAL:
                if step.condition is None or step.condition(result):
                    result = await self._execute_runner(step.runner, result, user_id, session_id, llm, sdk_context)

            elif step.mode == StepMode.LOOP:
                iterations = 0
                max_iter = step.max_iterations or 1
                while True:
                    result = await self._execute_runner(step.runner, result, user_id, session_id, llm, sdk_context)
                    iterations += 1
                    if step.condition and step.condition(result):
                        break
                    if iterations >= max_iter:
                        break
            else:
                raise ValueError(f"Unsupported step mode {step.mode}")
        return result
