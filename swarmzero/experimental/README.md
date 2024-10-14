# SwarmZero SDK

## Overview

The SDK is designed to automate the handling of complex user queries by breaking them down into manageable subtasks and invoking relevant AI agents. Each agent, running as a Docker container and exposed as a REST API service, is specialized in handling specific tasks. The orchestrator coordinates these tasks, ensures proper sequencing, manages user permissions, and handles communication between agents.

## Key Features

- **Orchestrator**: Decomposes user queries into subtasks and orchestrates the execution.
- **Agent**: Handles subtasks.
- **Task Queue with Redis**: Manages task distribution and processing.
- **User Permission Handling**: "Human-in-the-loop" approach for sensitive actions.

## Architecture

The system consists of the following components:

1. **Orchestrator (FastAPI, Port 3000)**

   - Endpoints:
     - `/query`: Accepts user queries, breaks them into subtasks, and queues them in Redis.
     - `/query_status`: Allows users to check the status of their query using the task ID.
     - `/approve_task`: Allows users to approve tasks that require explicit permission.

2. **Agent (FastAPI, Port 8000)**

   - Endpoints:
     - `/execute_task`: Processes individual subtask and it's status in Redis.

3. **Redis**: Used for queuing tasks (`task_queue`) and tracking task statuses (`task_status`).

4. **Queue Consumer**: Runs as a background task in the orchestrator to consume tasks from Redis and invoke the appropriate agent's REST API.

## Prerequisites

- Python 3.11+
- Docker
- Redis server

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/swarmzero/swarmzero.git
   cd swarmzero
   ```

2. **Install Python dependencies**:

   ```bash
   poetry install #or
   pip install pyproject.toml
   ```

3. **Start Redis Server**:

   You can run Redis locally or use Docker to start a Redis container:

   ```bash
   # using homebrew
   brew install redis
   redis-server #start redis
   brew services start redis # do this if the above command errors out
   brew services stop redis # stop service

   # GUI to explore redis
   brew install --cask another-redis-desktop-manager

   # using docker
   docker run -d -p 6379:6379 redis
   ```

## Running the Services

Run the orchestrator and agent FastAPI service in two separate terminals

```bash
python orchestrator/main.py
python agent/main.py
```

### Vscode Config

```json
  "flake8.args": ["--max-line-length=110", "--extend-ignore", "E203"],
  "black-formatter.args": ["--line-length", "110"],
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  }
```
