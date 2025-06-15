import asyncio

import pytest

from swarmzero.sdk_context import SDKContext
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
        sdk_context=SDKContext("./swarmzero_config_test.toml"),
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
