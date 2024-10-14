from typing import List, Any, Union, Iterable
from tasks.task_base import Task


class Runner:
    """
    Abstract base class for task runners.

    This class should be inherited by any runner that implements a specific
    task execution strategy.
    """

    def __init__(self):
        self.tasks: List[Task] = []

    def add_task(self, task_or_tasks: Union[Task, Iterable[Task]]):
        if isinstance(task_or_tasks, Task):
            self.tasks.append(task_or_tasks)
        elif isinstance(task_or_tasks, Iterable):
            for task in task_or_tasks:
                if not isinstance(task, Task):
                    raise TypeError("All elements must be instances of Task.")
            self.tasks.extend(task_or_tasks)
        else:
            raise TypeError("Argument must be an instance of Task or an iterable of Task instances.")


class Runners(list[Runner]):
    """
    Abstract base class for task runners.

    This class should be inherited by any runner that implements a specific
    task execution strategy.
    """

    def __init__(
        self,
    ):
        super().__init__()

    def add_runner(self, runner_cls: type):
        """
        Instantiates and adds a Runner subclass instance to the list of runners.

        :param runner_cls: The Runner subclass to instantiate and add.
        """
        if not issubclass(runner_cls, Runner):
            raise TypeError("Only subclasses of Runner can be added.")
        runner = runner_cls()
        self.append(runner)
