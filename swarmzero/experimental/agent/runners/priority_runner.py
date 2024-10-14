import logging
from .base_runner import Runner


class PriorityRunner(Runner):
    def __init__(self):
        """
        Initialize the PriorityRunner.
        """
        super().__init__()
        logging.basicConfig(level=logging.INFO)

    def run(self):
        """
        Execute the given tasks based on their priority.

        :param tasks_with_priority: List of TaskWithPriority objects to be executed.
        :param context: Context in which the tasks are run.
        """
        logging.info("Priority-based task execution started.")

        # Sort tasks by priority (higher priority tasks first)
        sorted_tasks = sorted(self.tasks, key=lambda x: x.priority or 0, reverse=True)

        task_responses = {}
        for task_with_priority in sorted_tasks:
            try:
                result = task_with_priority.run()
                task_responses[task_with_priority.task_id] = result
                logging.info(
                    f"Task {task_with_priority.task_id} with priority {task_with_priority.priority} completed successfully."
                )
            except Exception as e:
                logging.error(
                    f"Error executing task {task_with_priority.task_id} with priority {task_with_priority.priority}: {e}"
                )

        logging.info("Priority-based task execution completed.")
        return task_responses
