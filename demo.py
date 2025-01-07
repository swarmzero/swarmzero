import asyncio

from dotenv import load_dotenv

from swarmzero.agent import Agent
from swarmzero.sdk_context import SDKContext
from swarmzero.swarm import Swarm

load_dotenv()


# Example tool functions that our agents will use
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


async def demo_agent():
    print("\n=== Agent Demo ===")

    # Initialize SDK context
    print("\nInitializing SDK context...")
    sdk_context = SDKContext()
    await sdk_context.load_db_manager()

    # Create a new agent
    print("\nCreating new agent...")
    agent = Agent(
        name="math_agent",
        functions=[add_numbers, multiply_numbers],
        instruction="You are a helpful math assistant that can perform calculations.",
        description="A math-focused agent that can perform basic arithmetic.",
        role="calculator",
        sdk_context=sdk_context,
    )

    # Use the agent
    print("\nTesting agent...")
    response = await agent.chat("Can you help me add 5 and 3?")
    print(f"Agent response: {response}")

    # Save the agent to database
    print("\nSaving agent to database...")
    await agent.sdk_context.save_resource_to_db(agent.id)

    # Create a new SDK context and load the agent
    print("\nLoading agent from database...")
    new_context = SDKContext()
    await new_context.load_db_manager()
    loaded_agent = await new_context.load_resource_from_db(agent.id)

    # Test the loaded agent
    print("\nTesting loaded agent...")
    response = await loaded_agent.chat("What's 4 times 6?")
    print(f"Loaded agent response: {response}")

    # Find agents by type
    print("\nFinding all agents...")
    agents = await new_context.find_resources(resource_type="agent")
    print(f"Found {len(agents)} agents")


async def demo_swarm():
    print("\n=== Swarm Demo ===")

    # Initialize SDK context
    print("\nInitializing SDK context...")
    sdk_context = SDKContext()
    await sdk_context.load_db_manager()

    # Create multiple agents for the swarm
    print("\nCreating agents for swarm...")
    addition_agent = Agent(
        name="addition_specialist",
        functions=[add_numbers],
        instruction="You are an addition specialist.",
        description="Specializes in addition operations",
        role="adder",
        sdk_context=sdk_context,
    )

    multiplication_agent = Agent(
        name="multiplication_specialist",
        functions=[multiply_numbers],
        instruction="You are a multiplication specialist.",
        description="Specializes in multiplication operations",
        role="multiplier",
        sdk_context=sdk_context,
    )

    # Create a swarm
    print("\nCreating swarm...")
    swarm = Swarm(
        name="math_swarm",
        description="A swarm of specialized math agents",
        instruction="You are a math swarm that can delegate tasks to specialized agents.",
        functions=[add_numbers, multiply_numbers],
        agents=[addition_agent, multiplication_agent],
        sdk_context=sdk_context,
    )

    # Use the swarm
    print("\nTesting swarm...")
    response = await swarm.chat("I need to add 7 and 8, then multiply the result by 3.")
    print(f"Swarm response: {response}")

    # Save the swarm to database
    print("\nSaving swarm to database...")
    await swarm.sdk_context.save_resource_to_db(swarm.id)

    # Create a new SDK context and load the swarm
    print("\nLoading swarm from database...")
    new_context = SDKContext()
    await new_context.load_db_manager()
    loaded_swarm = await new_context.load_resource_from_db(swarm.id)

    # Test the loaded swarm
    print("\nTesting loaded swarm...")
    response = await loaded_swarm.chat("What's 5 plus 3 times 4?")
    print(f"Loaded swarm response: {response}")

    # Find swarms by type
    print("\nFinding all swarms...")
    swarms = await new_context.find_resources(resource_type="swarm")
    print(f"Found {len(swarms)} swarms")


async def main():
    # Run agent demo
    await demo_agent()

    # Run swarm demo
    await demo_swarm()


if __name__ == "__main__":
    asyncio.run(main())
