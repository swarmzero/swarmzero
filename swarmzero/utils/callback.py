from typing import Any, Dict, List, Optional

from llama_index.core.callbacks import (
    CBEventType,
    EventPayload,
)
from llama_index.core.callbacks.base_handler import BaseCallbackHandler

class ReasoningCaptureCallback(BaseCallbackHandler):
    def __init__(self, sdk_context, llm):
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self.current_reasoning_steps = []
        self.sdk_context = sdk_context
        self.__llm = llm
        self.__reasoning_prompt = """Your task is to convert the reasoning steps into a single, concise sentence that explains your current action to the user in first person, such as “Calling img2text tool to create image to text” or “Calling Script writer agent to write a script”. Do not ask any questions or seek additional input; simply describe your reasoning step in one clear sentence.
        Reasoning steps:
        {reasoning_steps}
        """

    def start_trace(self, trace_id: Optional[str] = None, **kwargs: Any) -> None:
        pass

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        pass

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[EventPayload, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> None:
        pass

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[EventPayload, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        if event_type == CBEventType.LLM and payload:
            llm_output = payload.get(EventPayload.RESPONSE)
            if llm_output and llm_output.message and llm_output.message.content:
                reasoning_steps = self.parse_reasoning_steps(llm_output.message.content)
                self.current_reasoning_steps.append(reasoning_steps)
                self.sdk_context.add_utility(reasoning_steps, "reasoning_steps", "steps")
                print(f"Captured reasoning steps: {reasoning_steps}")
                print(f"Human readable reasoning steps: {self.__llm.complete(self.__reasoning_prompt.format(reasoning_steps=reasoning_steps))}")

    def parse_reasoning_steps(self, llm_output: str):
        reasoning_steps = []
        lines = llm_output.strip().split('\n')
        current_step = {}
        for line in lines:
            if line.startswith("Thought:"):
                if current_step:
                    reasoning_steps.append(current_step)
                    current_step = {}
                current_step['Thought'] = line[len("Thought:"):].strip()
            elif line.startswith("Action:"):
                current_step['Action'] = line[len("Action:"):].strip()
            elif line.startswith("Action Input:"):
                current_step['Action Input'] = line[len("Action Input:"):].strip()
            elif line.startswith("Observation:"):
                current_step['Observation'] = line[len("Observation:"):].strip()
            elif line.startswith("Answer:"):
                current_step['Answer'] = line[len("Answer:"):].strip()
                reasoning_steps.append(current_step)
                current_step = {}
        if current_step:
            reasoning_steps.append(current_step)
        return reasoning_steps
