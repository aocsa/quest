from abc import ABC, abstractmethod
from typing import List

from record_batch import RecordBatch
from schema import Schema


class DataSource(ABC):
    @abstractmethod
    def schema(self) -> Schema:
        pass

    @abstractmethod
    def scan(self, projection: List[str]) -> List[RecordBatch]:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass
