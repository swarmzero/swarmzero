from typing import List, Optional

from colorama import Fore, Style  # type: ignore
from events import ChooseAgentEvent, OrchestratorEvent
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.workflow import Context, Event, Workflow
from llama_index.llms.openai import OpenAI  # type: ignore
from tools import add, lookup_stock_price, multiply, search_for_stock_symbol, social_media_post, web_search


class Agent:

    def __init__(self, name: str, workflow: Workflow, context: Context):
        self.name = name
        self.workflow = workflow
        self.context = context


class OrchestratorAgent(Agent):
    tools: Optional[List[BaseTool]]
    system_prompt: str
    llm: FunctionCallingLLM
    current_event: Event

    def __init__(
        self,
        name: str,
        workflow: Workflow,
        context: Context,
        tools: Optional[List[BaseTool]],
        system_prompt: str,
        llm: FunctionCallingLLM,
    ):
        super().__init__(name, workflow, context)
        self.workflow = workflow
        self.context = context
        self.llm = llm
        self.system_prompt = system_prompt
        self.context.data["redirecting"] = False

        # set up the tools including the ones everybody gets
        def done() -> None:
            """When you complete your task, call this tool."""
            print(f"{self.name} is complete")
            self.context.data["redirecting"] = True
            workflow.send_event(OrchestratorEvent(just_completed=self.name))

        def need_help(reason: str) -> None:
            """If the user asks to do something you don't know how to do, call this."""
            print(f"{self.name} needs help {reason}")
            self.context.data["redirecting"] = True
            workflow.send_event(OrchestratorEvent(request=self.current_event.request, need_help=True))

        self.tools = [FunctionTool.from_defaults(fn=done), FunctionTool.from_defaults(fn=need_help)]
        if tools:
            for t in tools:
                self.tools.append(t)

        agent_worker = FunctionCallingAgentWorker.from_tools(
            self.tools,
            llm=self.llm,
            allow_parallel_tool_calls=False,
            system_prompt=self.system_prompt,
        )
        self.agent = agent_worker.as_agent()

    def handle_event(self, ev: Event):
        self.current_event = ev

        self.context.data["memory"].put(ChatMessage(role=MessageRole.USER, content=ev.request))
        chat_history = self.context.data["memory"].get()

        response = str(self.agent.chat(message=ev.request, chat_history=chat_history))

        self.context.data["memory"].put(ChatMessage(role=MessageRole.ASSISTANT, content=response))

        print(Fore.MAGENTA + str(response) + Style.RESET_ALL)

        # if they're sending us elsewhere we're done here
        if self.context.data["redirecting"]:
            self.context.data["redirecting"] = False
            return None

        # otherwise, get some user input and then loop
        user_msg_str = input("> ").strip()
        return ChooseAgentEvent(request=user_msg_str, agent_name=self.name)


class StockLookupAgent(Agent):

    def __new__(cls, name: str, workflow: Workflow, context: Context):
        # Instead of initializing StockLookupAgent, return a OrchestratorAgent
        return OrchestratorAgent(
            name=name,
            workflow=workflow,
            tools=[
                FunctionTool.from_defaults(fn=lookup_stock_price),
                FunctionTool.from_defaults(fn=search_for_stock_symbol),
            ],
            context=context,
            llm=OpenAI(model="gpt-4o", temperature=0.4),
            system_prompt="""
                        You are a helpful assistant that is looking up stock prices.
                        The user may not know the stock symbol of the company they're interested in,
                        so you can help them look it up by the name of the company.
                        You can only look up stock symbols given to you by the search_for_stock_symbol tool, don't make them up. Trust the output of the search_for_stock_symbol tool even if it doesn't make sense to you.
                        Once you have retrieved a stock price, you *must* call the tool named "done" to signal that you are done. Do this before you respond.
                        If the user asks to do anything other than look up a stock symbol or price, call the tool "need_help" to signal some other agent should help.
                    """,
        )


class MathsAgent(Agent):

    def __new__(cls, name: str, workflow: Workflow, context: Context):

        return OrchestratorAgent(
            name=name,
            workflow=workflow,
            tools=[FunctionTool.from_defaults(add), FunctionTool.from_defaults(multiply)],
            context=context,
            llm=OpenAI(model="gpt-4", temperature=0.4),
            system_prompt="""
                    You are a helpful assistant that can solve simple math problems.
                    Once you solve the problem, you *must* call the tool named "done" to signal that you are done. Do this before you respond.
                    If the user asks to do anything other than solving math problems, call the tool "need_help" to signal some other agent should help.
                """,
        )


class WebAgent(Agent):

    def __new__(cls, name: str, workflow: Workflow, context: Context):

        return OrchestratorAgent(
            name=name,
            workflow=workflow,
            tools=[FunctionTool.from_defaults(web_search), FunctionTool.from_defaults(social_media_post)],
            context=context,
            llm=OpenAI(model="gpt-4o", temperature=0.4),
            system_prompt="""
                    You are a helpful assistant that can look up anything on web and post on social media. Trust the output of the tools even if it doesn't make sense to you.
                    Once you have retrieved an answer, you *must* call the tool named "done" to signal that you are done. Do this before you respond.
                    If the user asks to do anything other than search the internet or post on social media, call the tool "need_help" to signal some other agent should help.
                """,
        )


class AuthenticationAgent(Agent):
    def __new__(cls, name: str, workflow: Workflow, context: Context):

        def store_username(username: str) -> None:
            """Adds the username to the user state."""
            print("Recording username")
            context.data["user"]["username"] = username

        def login(password: str) -> None:
            """Given a password, logs in and stores a session token in the user state."""
            print(f"Logging in {context.data['user']['username']}")
            session_token = "output_of_login_function_goes_here"
            context.data["user"]["session_token"] = session_token

        def is_authenticated() -> bool:
            """Checks if the user has a session token."""
            print("Checking if authenticated")
            if context.data["user"]["session_token"] is not None:
                return True
            return False

        return OrchestratorAgent(
            name,
            workflow=workflow,
            tools=[
                FunctionTool.from_defaults(store_username),
                FunctionTool.from_defaults(login),
                FunctionTool.from_defaults(is_authenticated),
            ],
            context=context,
            llm=OpenAI(model="gpt-4o", temperature=0.4),
            system_prompt="""
                    You are a helpful assistant that is authenticating a user.
                    Your task is to get a valid session token stored in the user state.
                    To do this, the user must supply you with a username and a valid password. You can ask them to supply these.
                    If the user supplies a username and password, call the tool "login" to log them in.
                    Once you've called the login tool successfully, call the tool named "done" to signal that you are done. Do this before you respond.
                    If the user asks to do anything other than authenticate, call the tool "need_help" to signal some other agent should help.
                """,
        )


class AccountBalanceAgent(Agent):
    def __new__(cls, name: str, workflow: Workflow, context: Context):

        def get_account_id(account_name: str) -> str:
            """Useful for looking up an account ID."""
            print(f"Looking up account ID for {account_name}")
            account_id = "1234567890"
            context.data["user"]["account_id"] = account_id
            return f"Account id is {account_id}"

        def get_account_balance(account_id: str) -> str:
            """Useful for looking up an account balance."""
            print(f"Looking up account balance for {account_id}")
            context.data["user"]["account_balance"] = 1000
            return f"Account {account_id} has a balance of ${context.data['user']['account_balance']}"

        def is_authenticated() -> bool:
            """Checks if the user is authenticated."""
            print("Account balance agent is checking if authenticated")
            if context.data["user"]["session_token"] is not None:
                return True
            else:
                return False

        def authenticate():
            """Call this if the user needs to be authenticated."""
            print("Account balance agent is authenticating")
            context.data["redirecting"] = True
            context.data["overall_request"] = "Check account balance"

            workflow.send_event(OrchestratorEvent(request="Authenticate", need_help=True))

        return OrchestratorAgent(
            name,
            workflow=workflow,
            tools=[
                FunctionTool.from_defaults(get_account_id),
                FunctionTool.from_defaults(get_account_balance),
                FunctionTool.from_defaults(is_authenticated),
                FunctionTool.from_defaults(authenticate),
            ],
            context=context,
            llm=OpenAI(model="gpt-4", temperature=0.4),
            system_prompt="""
                You are a helpful assistant that is looking up account balances.
                The user may not know the account ID of the account they're interested in,
                so you can help them look it up by the name of the account.
                The user can only do this if they are authenticated, which you can check with the is_authenticated tool.
                If they aren't authenticated, call the "authenticate" tool to trigger the start of the authentication process; tell them you have done this.
                If they're trying to transfer money, they have to check their account balance first, which you can help with.
                Once you have supplied an account balance, you must call the tool named "done" to signal that you are done. Do this before you respond.
                If the user asks to do anything other than look up an account balance, call the tool "need_help" to signal some other agent should help.
            """,
        )


class TransferMoneyAgent(Agent):
    def __new__(cls, name: str, workflow: Workflow, context: Context):

        def transfer_money(from_account_id: str, to_account_id: str, amount: int) -> str:
            """Useful for transferring money between accounts."""
            print(f"Transferring {amount} from {from_account_id} account {to_account_id}")

            return f"Transferred {amount} to account {to_account_id}"

        def balance_sufficient(account_id: str, amount: int) -> bool:
            """Useful for checking if an account has enough money to transfer."""
            # todo: actually check they've selected the right account ID
            print("Checking if balance is sufficient")
            if context.data["user"]["account_balance"] >= amount:
                return True
            return False

        def has_balance() -> bool:
            """Useful for checking if an account has a balance."""
            print("Checking if account has a balance")
            if (
                context.data["user"]["account_balance"] is not None
                and context.data["user"]["account_balance"] > 0
            ):
                print("It does", context.data["user"]["account_balance"])
                return True
            else:
                return False

        def is_authenticated() -> bool:
            """Checks if the user has a session token."""
            print("Transfer money agent is checking if authenticated")
            if context.data["user"]["session_token"] is not None:
                return True
            else:
                return False

        def authenticate() -> None:
            """Call this if the user needs to authenticate."""
            print("Account balance agent is authenticating")
            context.data["redirecting"] = True
            context.data["overall_request"] = "Transfer money"
            workflow.send_event(OrchestratorEvent(request="Authenticate user", need_help=True))

        def check_balance() -> None:
            """Call this if the user needs to check their account balance."""
            print("Transfer money agent is checking balance")
            context.data["redirecting"] = True
            context.data["overall_request"] = "Transfer money"
            workflow.send_event(OrchestratorEvent(request="Check balance", need_help=True))

        return OrchestratorAgent(
            name,
            workflow=workflow,
            tools=[
                FunctionTool.from_defaults(transfer_money),
                FunctionTool.from_defaults(balance_sufficient),
                FunctionTool.from_defaults(has_balance),
                FunctionTool.from_defaults(is_authenticated),
                FunctionTool.from_defaults(authenticate),
                FunctionTool.from_defaults(check_balance),
            ],
            context=context,
            llm=OpenAI(model="gpt-4o", temperature=0.4),
            system_prompt="""
                You are a helpful assistant that transfers money between accounts.
                The user can only do this if they are authenticated, which you can check with the is_authenticated tool.
                If they aren't authenticated, tell them to authenticate first.
                The user must also have looked up their account balance already, which you can check with the has_balance tool.
                If they haven't already, tell them to look up their account balance first.
                Once you have transferred the money, you can call the tool named "done" to signal that you are done. Do this before you respond.
                If the user asks to do anything other than transfer money, call the tool "done" to signal some other agent should help.
            """,
        )


class OtherAgent(Agent):

    def __new__(cls, name: str, workflow: Workflow, context: Context):

        return OrchestratorAgent(
            name=name,
            workflow=workflow,
            tools=[],
            context=context,
            llm=OpenAI(model="gpt-4o", temperature=0.4),
            system_prompt="""
                    You are a helpful assistant that can help with anything.
                    Once you solve the problem, you *must* call the tool named "done" to signal that you are done. Do this before you respond.
                    If you can't help user asks, call the tool "need_help" to signal some other agent should help.
                """,
        )


agents = {
    "stock_lookup_agent": {
        "description": "Call this if the user wants to look up a stock price.",
        "class": StockLookupAgent,
    },
    "authenticate_agent": {
        "description": "Call this if the user needs to be authenticated.",
        "class": AuthenticationAgent,
    },
    "account_balance_agent": {
        "description": "Call this if the user wants to check an account balance.",
        "class": AccountBalanceAgent,
    },
    "transfer_money_agent": {
        "description": "Call this if the user wants to transfer money.",
        "class": TransferMoneyAgent,
    },
    "maths_agent": {
        "description": "Call this if the user wants help with solving math problems.",
        "class": MathsAgent,
    },
    "web_agent": {
        "description": "Call this if the user wants to search the internet or post on social media",
        "class": WebAgent,
    },
    "other_agent": {
        "description": "Call this if none of the other agents can help user.",
        "class": OtherAgent,
    },
}

agents_list = [{k: v["description"]} for k, v in agents.items()]


def get_agent_class(agent_name: str):
    """
    Dynamically retrieves the agent class based on the agent name.

    :param agent_name: The name of the agent to retrieve.
    :return: The agent class if found, otherwise None.
    """
    agent_info = agents.get(agent_name)
    if agent_info:
        return agent_info["class"]
    else:
        raise KeyError(f"Agent '{agent_name}' not found.")
