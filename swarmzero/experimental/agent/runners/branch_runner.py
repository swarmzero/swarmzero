from .base_runner import Runner
import logging


class BranchRunner(Runner):
    def __init__(self, branch_conditions):
        """
        Initialize with a dictionary mapping conditions to task lists.
        branch_conditions: dict where key is a condition function, and value is a list of tasks.
        """
        super().__init__()
        self.branch_conditions = branch_conditions
        logging.basicConfig(level=logging.INFO)

    def run(self):
        """
        Run the primary task and execute branch tasks based on conditions.

        tasks: list where the first element is the primary task and the rest are branch tasks.
        context: SDKContext object providing the context for tasks.
        """
        if not self.tasks:
            raise ValueError("No tasks provided to run.")

        task_responses = {}
        primary_task = self.tasks[0]
        primary_result = primary_task.run()
        task_responses[primary_task.task_id] = primary_result
        logging.info(f"Task {primary_task.task_id} completed successfully.")

        for condition, branch_tasks in self.branch_conditions.items():
            if condition(primary_result):
                for task in branch_tasks:
                    result = task.run()
                    task_responses[task.task_id] = result
                    logging.info(f"Task {task.task_id} completed successfully.")
                break  # Exit after the first matching branch

        return task_responses
