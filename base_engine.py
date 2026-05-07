from abc import ABC, abstractmethod

class BaseEngine(ABC):
    def __init__(self):
        """
        Initialize common variables. 
        Each engine will store its own 'pipe' here.
        """
        self.pipe = None

    @abstractmethod
    def load(self):
        """
        Force every child engine to implement a loading sequence.
        This is where resolve_snapshot_path and from_pretrained go.
        """
        pass

    @abstractmethod
    def execute(self, job_input):
        """
        Force every child engine to implement an execution sequence.
        This is where the actual inference (generation) happens.
        """
        pass