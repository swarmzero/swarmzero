import time
from .base_runner import Runner
import logging


class DelayedRunner(Runner):
    def __init__(self, delay):
        """
        Initialize with a delay duration.

        delay: Time in seconds to delay between running tasks.
        """
        super().__init__()
        self.delay = delay
        logging.basicConfig(level=logging.INFO)

    def run(self):
        """
        Run tasks with a delay between each task.

        tasks: List of tasks to be run.
        context: SDKContext object providing the context for tasks.
        """
        if not self.tasks:
            raise ValueError("No tasks provided to run.")

        task_responses = {}
        for task in self.tasks:
            result = task.run()
            task_responses[task.task_id] = result
            logging.info(f"Task {task.task_id} completed successfully and sleeps for {self.delay}.")
            time.sleep(self.delay)

        return task_responses
