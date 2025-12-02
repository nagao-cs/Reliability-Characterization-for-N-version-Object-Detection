from abc import ABC, abstractmethod


class AbstractIntegrator(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def integrate(self, detections):
        pass
