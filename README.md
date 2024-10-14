![](./assets/logo_dark.png)

# SwarmZero SDK

This library provides you with an easy way to create and run AI Agents and Swarms of Agents.

Join our Discord community for support and discussion.

[![](https://dcbadge.limes.pink/api/server/vnJvW4wZp9)](https://discord.gg/vnJvW4wZp9)

## Supported Large Language Models (LLMs)
- OpenAI
  - o1-preview
  - o1-mini 
  - gpt-4o
  - gpt-4o-mini
  - gpt-4-turbo
  - gpt-4
  - gpt-3.5-turbo
  - gpt-3.5-turbo-instruct
- Anthropic
  - claude-3-5-sonnet-20240620 
  - claude-3-opus-20240229
  - claude-3-sonnet-20240229
  - claude-3-haiku-20240307
- MistralAI
   - mistral-large-latest
   - mistral-small-latest
   - open-mixtral-8x22b
   - codestral-latest
- Gemini
   - gemini-1.5-flash
   - gemini-pro
   - gemini-1.5-pro
   - gemini-1.0-pro
- Popular open source LLMs
   - llama3.2
   - llama3.1
   - llava
   - gemma2:2b
   - gemma2
   - gemma:7b
   - gemma:2b
   - gemma2:27b
   - codegemma
   - mistral-nemo
   - deepseek-coder-v2
   - phi3:mini
   - phi3:medium
   - command-r
   - command-r-plus
   - and any supported by [Ollama](https://ollama.com/library)

## Project Requirements

- Python >= 3.11

## Installation

You can either directly install from pip:

```sh
pip install git+https://github.com/swarmzero/swarmzero.git@main
```

Or add it to your _requirements.txt_ file:

```sh
swarmzero @ git+https://github.com/swarmzero/swarmzero@main
```

### Optional Dependencies

To install with the optional web3 dependencies, you can specify them as follows:

```sh
pip install git+https://github.com/swarmzero/swarmzero.git@main#egg=swarmzero[web3]
```

Or add it to your _requirements.txt_ file:

```sh
swarmzero[web3] @ git+https://github.com/swarmzero/swarmzero@main
```

## Environment Setup

You need to specify an `OPENAI_API_KEY` in a _.env_ file in this directory.

Make a copy of the [.env.example](.env.example) file and rename it to _.env_.

## Configuration Setup

To use a configuration file with your `Agent`, follow these steps:

1. **Create a Configuration File**:

   - Create a TOML file (e.g., `swarmzero_config.toml`) in your project directory. (See [swarmzero_config_example.toml](./swarmzero_config_example.toml)).

2. **Create an SDK Context**:

   - Create an instance of SDKContext with the path to your configuration file.
   - The SDKContext allows you to manage configurations, resources, and utilities across your SwarmZero Agents more efficiently.

   ```python
   from swarmzero.sdk_context import SDKContext

   sdk_context = SDKContext(config_path="./swarmzero_config.toml")
   ```

2. **Specify the Configuration Path**:

   - When creating a `Agent` instance, provide the relative or absolute path to your configuration file.
   - Agent will use the configuration from the SDK Context. If you have one agent you can directly pass the config_path it will create the sdk_context for you.


   ```python
   from swarmzero import Agent

   simple_agent = Agent(
       name="Simple Agent",
       functions=[],
       instruction="your instructions for this agent's goal",
       # sdk_context=sdk_context
       config_path="./swarmzero_config.toml" 
   )
   ```

## Usage

First import the `Agent` class:

```python
from swarmzero import Agent
```

Load your environment variables:

```python
from dotenv import load_dotenv
load_dotenv()
```

Then create a Agent instance:

```python
my_agent = Agent(
    name="my_agent",
    functions=[],
    instruction="your instructions for this agent's goal",
)
```

Then, run your agent:

```python
my_agent.run()
```

Finally, call the API endpoint, `/api/v1/chat`, to see the result:

```sh
curl --request POST \
  --url http://localhost:8000/api/v1/chat \
  --header 'Content-Type: multipart/form-data' \
  --form 'user_id="test"' \
  --form 'session_id="test"' \
  --form 'chat_data={ "messages": [ { "role": "user", "content": "Who is Satoshi Nakamoto?" } ] }'
```

### Adding tools

You can create tools that help your agent handle more complex tasks. Here's an example:

````python
import os
from typing import Optional, Dict
from web3 import Web3
from swarmzero import Agent
from dotenv import load_dotenv

load_dotenv()

rpc_url = os.getenv("RPC_URL") # add an ETH Mainnet HTTP RPC URL to your `.env` file

def get_transaction_receipt(transaction_hash: str) -> Optional[Dict]:
    """
    Fetches the receipt of a specified transaction on the Ethereum blockchain and returns it as a dictionary.

    :param transaction_hash: The hash of the transaction to fetch the receipt for.
    :return: A dictionary containing the transaction receipt details, or None if the transaction cannot be found.
    """
    web3 = Web3(Web3.HTTPProvider(rpc_url))

    if not web3.is_connected():
        print("unable to connect to Ethereum")
        return None

    try:
        transaction_receipt = web3.eth.get_transaction_receipt(transaction_hash)
        return dict(transaction_receipt)
    except Exception as e:
        print(f"an error occurred: {e}")
        return None

if __name__ == "__main__":
    my_agent = Agent(
        name="my_agent",
        functions=[get_transaction_receipt]
    )

    my_agent.run()

    """
    [1] send a request:

    ```
    curl --request POST \
    --url http://localhost:8000/api/v1/chat \
    --header 'Content-Type: multipart/form-data' \
    --form 'user_id="test"' \
    --form 'session_id="test"' \
    --form 'chat_data={ "messages": [ { "role": "user", "content": "Who is the sender of this transaction - 0x5c504ed432cb51138bcf09aa5e8a410dd4a1e204ef84bfed1be16dfba1b22060" } ] }'
    ```

    [2] result:

    The address that initiated the transaction with hash 0x5c504ed432cb51138bcf09aa5e8a410dd4a1e204ef84bfed1be16dfba1b22060 is 0xA1E4380A3B1f749673E270229993eE55F35663b4.
    """
````

### Creating a Swarm

You can create a swarm of agents to collaborate on complex tasks. Here's an example of how to set up and use a swarm:

````python
from swarmzero.swarm import Swarm
from swarmzero.agent import Agent
from swarmzero.sdk_context import SDKContext

import asyncio

# Create SDK Context
sdk_context = SDKContext(config_path="./swarmzero_config_example.toml")


def save_report():
    return "save_item_to_csv"


def search_on_web():
    return "search_on_web"


# Create individual agents
agent1 = Agent(name="Research Agent", instruction="Conduct research on given topics", sdk_context=sdk_context,
               functions=[search_on_web])
agent2 = Agent(name="Analysis Agent", instruction="Analyze data and provide insights", sdk_context=sdk_context,
               functions=[save_report])
agent3 = Agent(name="Report Agent", instruction="Compile findings into a report", sdk_context=sdk_context, functions=[])

# Create swarm
swarm = Swarm(name="Research Team", description="A swarm of agents that collaborate on research tasks",
              instruction="Be helpful and collaborative", functions=[], agents=[agent1, agent2, agent3])


async def chat_with_swarm():
    return await swarm.chat("Can you analyze the following data: [1, 2, 3, 4, 5]")


if __name__ == "__main__":
    asyncio.run(chat_with_swarm())
````

### Adding Retriever

You can add retriever tools to create vector embeddings and retrieve semantic information. It will create vector index for every pdf documents under 'swarmzero-data/files/user' folder and can filter files with required_exts parameter.

- SwarmZero agent supports ".md", '.mdx' ,".txt", '.csv', '.docx', '.pdf' file types.
- SwarmZero agent supports 4 type of retriever (basic, chroma, pinecone-serverless, pinecone-pod) and controlled with retrieval_tool parameter.

````python
from swarmzero import Agent
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    my_agent = Agent(
        name="retrieve-test",
        functions=[],
        retrieve = True,
        required_exts = ['.md'],
        retrieval_tool='chroma'
    )

    my_agent.run()

    """
    [1] send a request:

    ```
    curl --request POST \
    --url http://localhost:8000/api/v1/chat \
    --header 'Content-Type: multipart/form-data' \
    --form 'user_id="test"' \
    --form 'session_id="test"' \
    --form 'chat_data={ "messages": [ { "role": "user", "content": "Can you summarise the documents?" } ] }'
    ```
    """
````


### Adding Sample Prompts

Users of your agent/swarm may not always be familiar with its abilities.
Providing sample prompts allows them to explore what you have built.
Here's how to add sample prompts which they can use before committing to use your agent/swarm.

#### Default
In your swarmzero_config.toml file, create a **top level** entry called `[sample_prompts]` and add a new array to the key `prompts` like this:
```toml
[sample_prompts]
prompts = [
  "What can you help me do?",
  "Which tools do you have access to?",
  "What are your capabilities?"
]
```

#### Specific agents in a swarm
```toml
[target_agent_id]
model = "gpt-3.5-turbo"
timeout = 15
environment = "dev"
enable_multi_modal = true
ollama_server_url = 'http://123.456.78.90:11434'
sample_prompts = [
    "What can you help me do?",
    "Which tools do you have access to?",
    "What are your capabilities?"
]
```

See [./swarmzero_config_example.toml](./swarmzero_config_example.toml) for an example configuration file.


## Tutorial

The complete tutorial can be found at [./tutorial.md](./tutorial.md).

## Contributing

### Setup

If you want to contribute to the codebase, you would need to set up your dev environment. Follow these steps:

- Create a new file called .env
- Copy the contents of [.env.example](.env.example) into your new .env file
- API keys for third-party tools are not provided.
  - `OPENAI_API_KEY` from OpenAI
- If you don't have Poetry installed, you can install it using the following commands:

```sh
curl -sSL https://install.python-poetry.org | python3 -

export PATH="$HOME/.local/bin:$PATH"
```

- Activate the Virtual Environment created by Poetry with the following command:

```sh
poetry shell
```

- Install dependencies.

```sh
poetry install --no-root
```

### Testing

- Make sure you're in the `tests/` directory:

```sh
cd tests/
```

- Run the test suite:

```sh
pytest
```

- Run tests for a specific module:

```sh
pytest tests/path/to/test_module.py
```

- Run with verbose output:

```sh
pytest -v
```

- Run with a detailed output of each test (including print statements):

```sh
pytest -s
```

- Run with coverage report:

```sh
pip install coverage pytest-cov
pytest  --cov --cov-report=html
```

Reports file `tests/htmlcov/index.html`

## API Doc

Open [http://localhost:8000/docs](http://localhost:8000/docs) with your browser to see the Swagger UI of the API.

## Learn More

<https://swarmzero.ai>
