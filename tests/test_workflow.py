import pytest
from unittest.mock import patch

from swarmzero.sdk_context import SDKContext
from swarmzero.agent import Agent
from swarmzero.swarm import Swarm
from swarmzero.workflow import StepMode, Workflow, WorkflowStep


@pytest.mark.asyncio
async def test_sequential_defaults():
    called = {}

    async def runner(prompt, user_id=None, session_id=None, llm=None, sdk_context=None):
        called['params'] = (user_id, session_id, llm, sdk_context)
        return prompt + "-done"

    wf = Workflow(
        name="wf",
        instruction="instr",
        description="desc",
        default_llm="llm",
        sdk_context=SDKContext("./tests/swarmzero_config_test.toml"),
        default_user_id="u",
        default_session_id="s",
        steps=[WorkflowStep(runner=runner)],
    )

    result = await wf.run("hi")
    assert result == "hi-done"
    assert called["params"][:3] == ("u", "s", "llm")
    assert isinstance(called["params"][3], SDKContext)


@pytest.mark.asyncio
async def test_parallel():
    outputs = []

    async def r1(prompt, **kwargs):
        outputs.append("r1")
        return prompt + "-r1"

    async def r2(prompt, **kwargs):
        outputs.append("r2")
        return prompt + "-r2"

    wf = Workflow(
        name="wf",
        steps=[WorkflowStep(runner=[r1, r2], mode=StepMode.PARALLEL)],
    )

    result = await wf.run("x")
    assert sorted(outputs) == ["r1", "r2"]
    assert sorted(result) == ["x-r1", "x-r2"]


@pytest.mark.asyncio
async def test_conditional():
    async def first(prompt, **kwargs):
        return 5

    async def second(prompt, **kwargs):
        return "ran"

    wf = Workflow(
        name="wf",
        steps=[
            WorkflowStep(runner=first),
            WorkflowStep(runner=second, mode=StepMode.CONDITIONAL, condition=lambda r: r == 5),
        ],
    )

    result = await wf.run("start")
    assert result == "ran"


@pytest.mark.asyncio
async def test_loop():
    counter = {"i": 0}

    async def step(prompt, **kwargs):
        counter["i"] += 1
        return counter["i"]

    wf = Workflow(
        name="wf",
        steps=[WorkflowStep(runner=step, mode=StepMode.LOOP, max_iterations=3)],
    )

    result = await wf.run(0)
    assert result == 3
    assert counter["i"] == 3


@pytest.mark.asyncio
async def test_loop_until_condition():
    counter = {"i": 0}

    async def step(prompt, **kwargs):
        counter["i"] += 1
        return counter["i"]

    wf = Workflow(
        name="wf",
        steps=[
            WorkflowStep(
                runner=step,
                mode=StepMode.LOOP,
                condition=lambda r: r >= 2,
                max_iterations=5,
            )
        ],
    )

    result = await wf.run(0)
    assert result >= 2
    assert counter["i"] <= 2


@pytest.mark.asyncio
async def test_nested_workflow():
    async def inner(prompt, **kwargs):
        return prompt + 1

    inner_wf = Workflow(name="inner", steps=[WorkflowStep(runner=inner)])

    outer_wf = Workflow(name="outer", steps=[WorkflowStep(runner=inner_wf)])

    result = await outer_wf.run(0)
    assert result == 1

@pytest.mark.asyncio
@patch('swarmzero.sdk_context.SDKContext.save_sdk_context_to_db')
async def test_workflow_with_agent(mock_save_db):
    """Test workflow setup with Agent runner"""
    mock_save_db.return_value = None

    sdk_context = SDKContext("./tests/swarmzero_config_test.toml")

    agent = Agent(
        name="test_agent",
        functions=[],
        instruction="Test instruction",
        sdk_context=sdk_context,
        chat_only_mode=True,
    )

    wf = Workflow(
        name="agent_workflow",
        sdk_context=sdk_context,
        steps=[WorkflowStep(runner=agent)],
    )

    result = await wf.run("test agent input")
    assert isinstance(result, str)
    assert len(result) > 0

@pytest.mark.asyncio
@patch('swarmzero.sdk_context.SDKContext.save_sdk_context_to_db')
async def test_workflow_with_swarm(mock_save_db):
    """Test workflow setup with Swarm runner"""
    mock_save_db.return_value = None

    sdk_context = SDKContext("./tests/swarmzero_config_test.toml")

    agent1 = Agent(
        name="agent1",
        functions=[],
        instruction="First agent",
        sdk_context=sdk_context,
        swarm_mode=True,
    )

    agent2 = Agent(
        name="agent2",
        functions=[],
        instruction="Second agent",
        sdk_context=sdk_context,
        swarm_mode=True,
    )

    swarm = Swarm(
        name="test_swarm",
        agents=[agent1, agent2],
        functions=[],
        instruction="Test swarm instruction",
        description="A test swarm for workflow",
        sdk_context=sdk_context,
    )

    wf = Workflow(
        name="swarm_workflow",
        sdk_context=sdk_context,
        steps=[WorkflowStep(runner=swarm)],
    )

    result = await wf.run("test swarm input")
    assert isinstance(result, str)
    assert len(result) > 0

@pytest.mark.asyncio
async def test_workflow_step_parameters():
    """Test WorkflowStep with custom parameters"""
    called_params = {}
    
    async def test_runner(prompt, user_id=None, session_id=None, llm=None, sdk_context=None):
        called_params.update({
            'prompt': prompt,
            'user_id': user_id,
            'session_id': session_id,
            'llm': llm,
            'sdk_context': sdk_context
        })
        return "step_result"
    
    custom_sdk_context = SDKContext("./tests/swarmzero_config_test.toml")
    
    step = WorkflowStep(
        runner=test_runner,
        user_id="custom_user",
        session_id="custom_session",
        llm="custom_llm",
        sdk_context=custom_sdk_context,
        name="custom_step"
    )
    
    wf = Workflow(
        name="param_test",
        steps=[step],
        default_user_id="default_user",
        default_session_id="default_session",
    )
    
    result = await wf.run("test")
    
    assert result == "step_result"
    assert called_params['user_id'] == "custom_user"
    assert called_params['session_id'] == "custom_session" 
    assert called_params['llm'] == "custom_llm"
    assert called_params['sdk_context'] == custom_sdk_context


@pytest.mark.asyncio
async def test_workflow_with_mixed_step_modes():
    """Test workflow with multiple step modes"""
    results = []
    
    async def step1(prompt, **kwargs):
        results.append("step1")
        return 1
        
    async def step2a(prompt, **kwargs):
        results.append("step2a")
        return "2a"
        
    async def step2b(prompt, **kwargs):
        results.append("step2b") 
        return "2b"
        
    async def step3(prompt, **kwargs):
        results.append("step3")
        # prompt is the result from the parallel step (a list)
        return len(prompt) * 5  # Convert to a number we can test
    
    wf = Workflow(
        name="mixed_workflow",
        steps=[
            WorkflowStep(runner=step1, mode=StepMode.SEQUENTIAL),
            WorkflowStep(runner=[step2a, step2b], mode=StepMode.PARALLEL),
            WorkflowStep(runner=step3, mode=StepMode.CONDITIONAL, condition=lambda r: len(r) == 2),
        ]
    )
    
    result = await wf.run(5)
    
    assert "step1" in results
    assert "step2a" in results
    assert "step2b" in results
    assert "step3" in results
    assert result == 10  # len(['2a', '2b']) * 5 = 2 * 5 = 10


@pytest.mark.asyncio
@patch('swarmzero.sdk_context.SDKContext.save_sdk_context_to_db')
async def test_workflow_agent_and_swarm_combination(mock_save_db):
    """Test workflow with both Agent and Swarm steps"""
    mock_save_db.return_value = None
    
    sdk_context = SDKContext("./tests/swarmzero_config_test.toml")
    
    # Create individual agent
    individual_agent = Agent(
        name="individual_agent",
        functions=[],
        instruction="Process individual tasks",
        sdk_context=sdk_context,
        chat_only_mode=True
    )
    
    # Create agents for swarm
    swarm_agent1 = Agent(
        name="swarm_agent1",
        functions=[],
        instruction="Collaborate in swarm",
        sdk_context=sdk_context,
        swarm_mode=True
    )
    
    swarm_agent2 = Agent(
        name="swarm_agent2",
        functions=[],
        instruction="Collaborate in swarm", 
        sdk_context=sdk_context,
        swarm_mode=True
    )
    
    # Create swarm
    swarm = Swarm(
        name="collaborative_swarm",
        description="Swarm for collaboration",
        instruction="Work together on complex tasks",
        functions=[],
        agents=[swarm_agent1, swarm_agent2],
        sdk_context=sdk_context
    )
    
    # Create workflow with both
    wf = Workflow(
        name="agent_swarm_workflow",
        steps=[
            WorkflowStep(runner=individual_agent, name="individual_step"),
            WorkflowStep(runner=swarm, name="swarm_step"),
        ],
        sdk_context=sdk_context
    )
    
    result = await wf.run("Complex task requiring both individual and swarm processing")
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
@patch('swarmzero.sdk_context.SDKContext.save_sdk_context_to_db')
async def test_workflow_with_agent_parallel(mock_save_db):
    """Test parallel execution with multiple agents"""
    mock_save_db.return_value = None
    
    sdk_context = SDKContext("./tests/swarmzero_config_test.toml")
    
    agent1 = Agent(
        name="parallel_agent1",
        functions=[],
        instruction="First parallel agent",
        sdk_context=sdk_context,
        chat_only_mode=True
    )
    
    agent2 = Agent(
        name="parallel_agent2",
        functions=[],
        instruction="Second parallel agent",
        sdk_context=sdk_context,
        chat_only_mode=True
    )
    
    wf = Workflow(
        name="parallel_agents_workflow",
        steps=[
            WorkflowStep(
                runner=[agent1, agent2],
                mode=StepMode.PARALLEL,
                name="parallel_processing"
            )
        ],
        sdk_context=sdk_context
    )
    
    result = await wf.run("Process this in parallel")
    assert isinstance(result, list)
    assert len(result) == 2
    # Each result should be a string response from the agents
    assert all(isinstance(r, str) and len(r) > 0 for r in result)


@pytest.mark.asyncio
async def test_workflow_error_handling():
    """Test workflow with unsupported runner type"""
    
    wf = Workflow(
        name="error_test",
        steps=[WorkflowStep(runner="invalid_runner_type")]
    )
    
    with pytest.raises(ValueError, match="Unsupported runner type"):
        await wf.run("test")