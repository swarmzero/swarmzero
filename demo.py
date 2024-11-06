import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from swarmzero import Agent, Swarm
from swarmzero.core.sdk_context import SDKContext

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / "tests" / ".env")

if not os.getenv('OPENAI_API_KEY'):
    raise ValueError("Failed to load OPENAI_API_KEY from .env file")


async def main():
    # context = SDKContext(config_path="./swarmzero_config_example.toml")
    context = await SDKContext.from_db()

    my_agent = Agent(
        name="my_agent",
        description="This is my agent",
        instruction="You are a helpful assistant",
        functions=[],
        sdk_context=context,
        swarm_mode=True,
    )

    your_agent = Agent(
        name="your_agent",
        description="This is your agent",
        instruction="You are a helpful assistant",
        functions=[],
        sdk_context=context,
        swarm_mode=True,
    )

    our_swarm = Swarm(
        name="our_swarm",
        description="This is our swarm",
        instruction="You are a helpful assistant",
        agents=[my_agent, your_agent],
        functions=[],
        sdk_context=context,
    )

    try:
        response = await our_swarm.chat("Hello, how are you?")
        print(response)
        # my_agent.run()
    finally:
        await our_swarm.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
