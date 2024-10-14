import concurrent.futures
from .base_runner import Runner
import logging


class ParallelRunner(Runner):
    def __init__(self):
        """
        Initialize the ParallelRunner.
        """
        super().__init__()
        logging.basicConfig(level=logging.INFO)

    def run(self):
        """
        Execute the given tasks in parallel.

        :param tasks: List of tasks to be executed.
        :param context: Context in which the tasks are run.
        """
        logging.info("Parallel task execution started.")
        task_responses = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(task.run): task for task in self.tasks}
            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    task_responses[task.task_id] = result
                    logging.info(f"Task {task.task_id} completed successfully.")
                except Exception as e:
                    logging.error(f"Error executing task {task.task_id}: {e}")
        logging.info("Parallel task execution completed.")
        return task_responses
