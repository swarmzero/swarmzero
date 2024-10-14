import importlib
import json
from pathlib import Path
from typing import Any, List, Optional

import dotenv
from colorama import Fore, Style  # type: ignore
from fastapi import FastAPI, HTTPException
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent.runner.base import AgentRunner
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.llms.openai import OpenAI  # type: ignore
from pydantic import BaseModel
from redis import Redis  # type: ignore # noqa

from tasks import Task
from runners import (
    BranchRunner,
    DelayedRunner,
    LoopRunner,
    ParallelRunner,
    PriorityRunner,
    SequentialRunner,
    Runner,
    Runners,
)

dotenv.load_dotenv()

app = FastAPI()
redis_client = Redis(host="localhost", port=6379, db=0)

TASK_QUEUE = "task_queue"
TASK_STATUS = "task_status"


def load_config(config_file: str, agent_name: str) -> dict:
    """
    Load agent configuration from a local file.
    """
    with Path(__file__).parent.with_name(config_file).open("r", encoding="utf-8") as file:
        config: dict[str, dict[str, str]] = json.load(file)
        if agent_name not in config:
            raise KeyError(f"Agent '{agent_name}' not found.")
        return config[agent_name]


def load_tools(tools: List[str]) -> List[BaseTool]:
    """
    Loads a list of tools from a module.

    Args:
        tools (List[str]): A list of tool names to load.

    Returns:
        List[BaseTool]: A list of loaded tools.

    Raises:
        AttributeError: If a tool is not found in the module.
        TypeError: If a tool is not callable.
        ImportError: If the module cannot be imported.
        Exception: If an unexpected error occurs.
    """
    module_name = "tools"
    llm_tools: List[BaseTool] = []

    for method_name in tools:
        try:
            module = importlib.import_module(module_name)

            if not hasattr(module, method_name):
                raise AttributeError(f"Module '{module_name}' has no function '{method_name}'")

            method = getattr(module, method_name)

            if not callable(method):
                raise TypeError(f"'{method_name}' in module '{module_name}' is not callable")

            llm_tools.append(FunctionTool.from_defaults(fn=method))

        except ImportError as e:
            print(f"Error importing module '{module_name}': {e}")
        except AttributeError as e:
            print(e)
        except TypeError as e:
            print(e)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    return llm_tools


class Agent:
    """
    An Agent that can execute tasks and respond to user requests.

    Attributes:
        name (str): The name of the Agent.
        system_prompt (str): The system prompt for the Agent.
        llm (FunctionCallingLLM): The language model used by the Agent.
        tools (List[BaseTool]): The tools available to the Agent.
        agent_runner (AgentRunner): The runner that executes the Agent's tasks.

    Methods:
        __init__(name, system_prompt, llm, tools): Initializes the Agent.
        execute(request, chat_history=None): Executes a user request and returns a response.

    Notes:
        The Agent uses a language model to generate responses to user requests.
        It also has a set of tools that can be used to perform tasks or request help.
    """

    name: str
    system_prompt: str
    llm: FunctionCallingLLM
    tools: Optional[List[BaseTool]]
    agent_runner: AgentRunner

    def __init__(
        self,
        name: str,
        system_prompt: str,
        llm: FunctionCallingLLM,
        tools: Optional[List[BaseTool]],
    ):
        """
        Initializes an Agent instance.

        Args:
            name (str): The name of the Agent.
            system_prompt (str): The system prompt for the Agent.
            llm (FunctionCallingLLM): The language model used by the Agent.
            tools (Optional[List[BaseTool]]): The tools available to the Agent.

        Returns:
            None
        """
        self.name = name
        self.system_prompt = system_prompt
        self.llm = llm

        def done() -> None:
            """When you complete your task, call this tool."""
            print(f"{self.name} is complete")
            # TODO redis

        def need_help(reason: str) -> None:
            """If the user asks to do something you don't know how to do, call this."""
            print(f"{self.name} needs help {reason}")
            # TODO redis

        self.tools = [FunctionTool.from_defaults(fn=done), FunctionTool.from_defaults(fn=need_help)]
        if tools:
            for t in tools:
                self.tools.append(t)

        self.agent_runner = FunctionCallingAgentWorker.from_tools(
            self.tools,
            llm=self.llm,
            allow_parallel_tool_calls=False,
            system_prompt=self.system_prompt,
        ).as_agent()

    def execute(self, request: str, chat_history: Optional[List[Any]] = None) -> str:
        """
        Executes a chat request and returns the response.

        Args:
            request (str): The user's chat request.
            chat_history (Optional[List[ChatMessage]], optional): The history of chat messages.

        Returns:
            str: The response to the user's chat request.
        """
        # self.context.data["memory"].put(ChatMessage(role=MessageRole.USER, content=ev.request))
        # chat_history = self.context.data["memory"].get()

        response = str(self.agent_runner.chat(message=request, chat_history=chat_history))

        # self.context.data["memory"].put(ChatMessage(role=MessageRole.ASSISTANT, content=response))

        print(Fore.MAGENTA + str(response) + Style.RESET_ALL)

        return response


@app.post("/execute_task")
async def execute_task(task: Task):
    """
    Endpoint to receive a task for the agent to execute.
    """

    print(f"Processing subtask: {task.query}")

    config = load_config("agents.json", task.agent_name)
    agent = Agent(
        name=task.agent_name,
        system_prompt=config["system_prompt"],
        llm=OpenAI(model=task.llm, temperature=0.4),
        tools=load_tools(config["tools"]),
    )

    try:
        result = agent.execute(task.query, task.chat_history)

        task_info = json.loads(redis_client.hget(TASK_STATUS, task.task_id))  # type: ignore
        task_info["completed_subtasks"] += 1
        task_info["results"][task.query] = result
        if task_info["completed_subtasks"] == task_info["total_subtasks"]:
            task_info["status"] = "completed"
        else:
            task_info["status"] = "in-progress"

        redis_client.hset(TASK_STATUS, task.task_id, json.dumps(task_info))

        return {"task_id": task.task_id, "status": task_info["status"]}
    except Exception as e:  # pylint: disable=broad-except
        return {"status": "error", "message": str(e)}


class RunnerRequest(BaseModel):
    runner_type: str
    tasks: List[Task]
    iterations: Optional[int] = None
    delay: Optional[float] = None
    branch_conditions: Optional[dict] = None


@app.post("/create_runner")
async def create_runner(request: RunnerRequest):
    runner_class = get_runner_class(request.runner_type)

    if runner_class is None:
        raise HTTPException(status_code=400, detail=f"Invalid runner type: {request.runner_type}")

    runner = create_runner_instance(runner_class, request)

    for task in request.tasks:
        runner.add_task(task)

    result = runner.run()

    return {"status": "success", "result": result}


def get_runner_class(runner_type: str) -> Optional[type]:
    runner_classes = {
        "sequential": SequentialRunner,
        "parallel": ParallelRunner,
        "loop": LoopRunner,
        "branch": BranchRunner,
        "delayed": DelayedRunner,
        "priority": PriorityRunner,
    }
    return runner_classes.get(runner_type.lower())


def create_runner_instance(runner_class: type, request: RunnerRequest) -> Runner:
    if runner_class == LoopRunner:
        return runner_class(iterations=request.iterations or 1)
    elif runner_class == DelayedRunner:
        return runner_class(delay=request.delay or 1)
    elif runner_class == BranchRunner:
        return runner_class(branch_conditions=request.branch_conditions or {})
    else:
        return runner_class()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
