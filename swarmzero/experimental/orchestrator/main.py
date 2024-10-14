import asyncio
import json
import uuid
from pathlib import Path

import dotenv
import requests  # type: ignore
from fastapi import FastAPI, HTTPException
from llama_index.llms.openai import OpenAI  # type: ignore
from pydantic import BaseModel
from redis import Redis  # type: ignore # noqa

dotenv.load_dotenv()

app = FastAPI()
redis_client = Redis(host="localhost", port=6379, db=0)

TASK_QUEUE = "task_queue"
TASK_STATUS = "task_status"
AGENT_URL = "http://localhost:8000/execute_task"


def load_config(config_file: str) -> dict:
    """
    Load agent configuration from a local file.
    """
    with Path(__file__).parent.with_name(config_file).open("r", encoding="utf-8") as file:
        agents: dict[str, dict[str, str]] = json.load(file)
        return agents


class Body(BaseModel):
    query: str
    orchestrator_llm: str = "gpt-4o"  # todo dynamic


@app.post("/query")
async def process_query(body: Body):
    """
    API endpoint to process the user's query.
    """
    response = OpenAI(model=body.orchestrator_llm, temperature=0.4).complete(
        f"""
            Given a user question, and a list of agents, output a list of
            relevant sub-questions, such that the answers to all the
            sub-questions put together will answer the question. Respond
            in pure JSON without any markdown, like this:
            {{
                "subtasks": [
                    {{"task": "What is the population of San Francisco?", "agent": "web_agent"}},
                    {{"task": "what is 2+2", "agent": "math_agent"}}
                ]
            }}
            Here is the user question: {body.query}

            And here is the list of agents: {load_config("agents.json")}
            """
    )

    subtasks = json.loads(str(response))["subtasks"]
    print(f"subtasks are {subtasks}")

    task_id = str(uuid.uuid4())
    task_status = {
        "status": "pending",
        "completed_subtasks": 0,
        "total_subtasks": len(subtasks),
        "subtasks": subtasks,
        "results": {},
    }

    await redis_client.hset(TASK_STATUS, task_id, json.dumps(task_status))
    for subtask in subtasks:
        task_data = json.dumps({"task_id": task_id, "subtask": subtask["task"], "agent_name": subtask["agent"]})
        await redis_client.rpush(TASK_QUEUE, task_data)

    return {"task_id": task_id, "status": "Task created and subtasks queued."}


@app.get("/query_status")
async def query_status(task_id: str):
    """
    Endpoint to check the status of the agent.
    """
    task_data = redis_client.hget(TASK_STATUS, task_id)

    if not task_data:
        raise HTTPException(status_code=404, detail="Task not found")

    task_info = json.loads(task_data)  # type: ignore
    return task_info


@app.on_event("startup")
async def startup_event():
    for _ in range(1):  # Start 1 consumers;
        asyncio.create_task(consume_tasks())


async def consume_tasks():
    while True:
        task_data = redis_client.lpop(TASK_QUEUE)

        if task_data:
            task = json.loads(task_data)
            task_id = task["task_id"]
            query = task["subtask"]
            agent_name = task["agent_name"]

            response = requests.post(
                AGENT_URL, json={"task_id": task_id, "query": query, "agent_name": agent_name}, timeout=30000
            )
            if response.status_code == 200:
                print(f"Task {task_id} subtask dispatched to {agent_name}.")
            else:
                print(f"Failed to dispatch task {task_id} subtask to {agent_name}.")

        else:
            await asyncio.sleep(1)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=3000)


app = FastAPI()
