from .base_runner import Runner
import logging


class SequentialRunner(Runner):
    def __init__(self):
        """
        Initialize the SequentialRunner.
        """
        super().__init__()
        logging.basicConfig(level=logging.INFO)

    def run(self):
        """
        Execute the given tasks sequentially.

        :param tasks: List of tasks to be executed.
        :param context: Context in which the tasks are run.
        """
        logging.info("Sequential task execution started.")
        task_responses = {}
        for task in self.tasks:
            try:
                task_response = task.run()
                task_responses[task.task_id] = task_response
                logging.info(f"Task {task} completed successfully.")

            except Exception as e:
                logging.error(f"Error executing task {task}: {e}")

        logging.info("Sequential task execution completed.")
        return task_responses
