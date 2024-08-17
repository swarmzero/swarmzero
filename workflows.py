from typing import List, Optional

import dotenv
from agents import agents_list, get_agent_class
from colorama import Fore, Style  # type: ignore
from events import ChooseAgentEvent, InitializeEvent, OrchestratorEvent
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step  # type: ignore
from llama_index.llms.openai import OpenAI  # type: ignore
from llama_index.utils.workflow import draw_all_possible_flows  # type: ignore

dotenv.load_dotenv()


class OrchestratorWorkflow(Workflow):

    @step(pass_context=True)
    async def initialize(self, ctx: Context, ev: InitializeEvent) -> OrchestratorEvent:
        ctx.data["user"] = {
            "username": None,
            "session_token": None,
            "account_id": None,
            "account_balance": None,
        }
        ctx.data["success"] = None
        ctx.data["redirecting"] = None
        ctx.data["overall_request"] = None
        ctx.data["llm"] = OpenAI(model="gpt-4o", temperature=0.4)
        ctx.data["memory"] = ChatMemoryBuffer.from_defaults(llm=ctx.data["llm"])

        return OrchestratorEvent()  # type: ignore

    @step(pass_context=True)
    async def orchestrator(
        self, ctx: Context, ev: OrchestratorEvent | StartEvent
    ) -> InitializeEvent | OrchestratorEvent | ChooseAgentEvent | StopEvent:
        # initialize user if not already done
        if "user" not in ctx.data:
            return InitializeEvent()

        request = str(ev.request)

        if ctx.data["overall_request"] is not None and not ev.need_help:
            print("There's an overall request in progress, it's ", ctx.data["overall_request"])
            last_request = ctx.data["overall_request"]
            ctx.data["overall_request"] = None
            request = last_request

        if isinstance(ev, OrchestratorEvent) and ev.need_help:
            print("The previous process needs help with ", request)

        print(f"Orchestrator received request: {request}")

        def choose_agent(sub_question: str, agent_name: str) -> bool:
            """Call this function with a sub_question and
            a valid agent_name if you found a relevant agent to work with.
            """
            print("__emitted: choose agent lookup", sub_question, agent_name)
            self.send_event(ChooseAgentEvent(request=sub_question, agent_name=agent_name))
            return True

        def emit_orchestrator() -> bool:
            """Call this if the user wants to do something else or
            you can't figure out what they want to do."""
            print("__emitted: OrchestratorEvent")
            self.send_event(OrchestratorEvent(request=ev.request))
            return True

        def emit_stop() -> bool:
            """Call this if the user wants to stop or exit the system."""
            print("__emitted: stop")
            self.send_event(StopEvent())
            return True

        tools: Optional[List[BaseTool]] = [
            FunctionTool.from_defaults(fn=choose_agent),
            FunctionTool.from_defaults(fn=emit_orchestrator),
            FunctionTool.from_defaults(fn=emit_stop),
        ]

        system_prompt = f"""
            You are on orchestration agent.
            Your job is to pick an agent to run based on the current state of the user and their query.
            You should breakdown the query into relevant sub-questions.
            Invoke the "choose_agent" tool with each sub-question and chosen agent.
            You do not need to call more than one tool.
            You do not need to figure out dependencies between agents; the agents will handle that themselves.
            If you did not call any tools, return the string "FAILED" without quotes and nothing else.
            Here is the list of agents to choose from: {agents_list}
        """

        agent_worker = FunctionCallingAgentWorker.from_tools(
            tools=tools, llm=ctx.data["llm"], allow_parallel_tool_calls=False, system_prompt=system_prompt
        )
        ctx.data["orchestrator"] = agent_worker.as_agent()

        orchestrator = ctx.data["orchestrator"]

        if request == "None" or not request or request == "":
            print(Fore.MAGENTA + str(str(orchestrator.chat(message="hello!"))) + Style.RESET_ALL)
            request = input("> ").strip()

        response = str(orchestrator.chat(message=request, chat_history=ctx.data["memory"].get()))

        if response == "FAILED":
            print("Orchestration agent failed; try again")
            return OrchestratorEvent(request=request)

        print(Fore.YELLOW + "Orchestrator: " + str(response) + Style.RESET_ALL)

        return None  # type: ignore

    @step(pass_context=True)
    async def choose_agent(self, ctx: Context, ev: ChooseAgentEvent) -> OrchestratorEvent:

        print(f"Choose Agent received request: {ev.request} {ev.agent_name}")
        orchestrator_agent = get_agent_class(ev.agent_name)

        return orchestrator_agent(ev.agent_name, self, ctx).handle_event(ev)


draw_all_possible_flows(OrchestratorWorkflow, filename="orchestrator_flows.html")


async def main():
    print(agents_list)
    c = OrchestratorWorkflow(timeout=1200, verbose=True)
    result = await c.run()
    print(result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
