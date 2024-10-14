from .base_runner import Runner
import logging


class LoopRunner(Runner):
    def __init__(self, iterations: int):
        """
        Initialize the LoopRunner with a specified number of iterations.

        :param iterations: Number of times to repeat the task execution.
        """
        super().__init__()
        self.iterations = iterations
        logging.basicConfig(level=logging.INFO)

    def run(self):
        """
        Execute the given tasks in a loop for the specified number of iterations.

        :param tasks: List of tasks to be executed.
        :param context: Context in which the tasks are run.
        """
        task_responses = {}
        for i in range(self.iterations):
            logging.info(f"Iteration {i + 1}/{self.iterations} started.")
            for task in self.tasks:
                try:
                    result = task.run()
                    task_responses[task.task_id] = result
                    logging.info(f"Task {task.task_id} completed successfully.")
                except Exception as e:
                    logging.error(f"Error executing task {task.task_id}: {e}")
            logging.info(f"Iteration {i + 1}/{self.iterations} completed.")
        return task_responses
