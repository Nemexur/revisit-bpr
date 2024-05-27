from typing import Any
from abc import ABC, abstractmethod


class Experiment(ABC):
    @property
    @abstractmethod
    def metrics(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def run(self) -> Any:
        pass

    @abstractmethod
    def clean(self) -> None:
        pass

    def interrupt(self) -> None:
        return
