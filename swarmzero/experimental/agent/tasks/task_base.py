from typing import List, Dict, Any
from pydantic import BaseModel
from llama_index.llms.openai import OpenAI


class Task(BaseModel):

    task_id: str
    query: str
    agent_name: str
    chat_history: List[Any] | None = None
    llm: str
    priority: int | None = None
    tools: Dict[str, List[str]] | None = None

    def run(self):
        from main import Agent

        llm = OpenAI(model=self.llm, temperature=0.1)
        agent = Agent(self.agent_name, "", llm, self.tools)
        response = agent.execute(self.query, self.chat_history)

        print(f"Task {self.task_id} response: {response}")
        return response
