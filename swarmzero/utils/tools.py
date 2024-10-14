import inspect
from typing import Callable, List

from llama_index.core.tools import FunctionTool


def is_async_function(func):
    return inspect.iscoroutinefunction(func)


# TODO: receive name and description individually
def tools_from_funcs(funcs: List[Callable], name="", description="") -> List[FunctionTool]:
    if len(funcs) == 0:
        return []

    function_tools = []
    for func in funcs:
        if is_async_function(func):
            fn = FunctionTool.from_defaults(async_fn=func, name=name, description=description)
        else:
            fn = FunctionTool.from_defaults(fn=func, name=name, description=description)

        function_tools.append(fn)

    return function_tools
