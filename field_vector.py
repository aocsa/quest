from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import pyarrow as pa
import numpy as np
from dtypes import numpy_dtype_to_arrow_dtype
from tensor_df_utils import tensor_to_strings, strings_to_tensor

ArrowType = pa.DataType


@dataclass
class Field:
    name: str
    dtype: ArrowType

    def __str__(self) -> str:
        return f"{self.name} | {self.dtype}"


class ColumnVector(ABC):

    @abstractmethod
    def field(self) -> Field:
        pass

    @abstractmethod
    def get_value(self, i: int):
        pass

    @abstractmethod
    def data(self) -> torch.Tensor:
        pass

    @abstractmethod
    def size(self) -> int:
        pass


class FieldVector(ColumnVector):

    def __init__(self, array: torch.Tensor, field: Field):
        self.array_ = array
        self.field_ = field

    def name(self) -> int:
        return self.field_.name

    def dtype(self) -> int:
        return self.field_.dtype

    def field(self) -> Field:
        return self.field_

    def get_value(self, i: int):
        return self.array_[i]

    def size(self) -> int:
        return len(self.array_)

    def data(self) -> torch.Tensor:
        return self.array_

    def __str__(self) -> str:
        return f"{self.array_}"

    @staticmethod
    def np_to_tensor(array: np.ndarray) -> torch.Tensor:
        field_dtype = numpy_dtype_to_arrow_dtype(array.dtype)
        tensor = None
        if field_dtype == pa.string():
            tensor = strings_to_tensor(array)
        else:
            tensor = torch.from_numpy(array)
        return tensor
    
    @staticmethod
    def to_numpy(vector: 'FieldVector') -> np.ndarray:
        if vector.dtype() == pa.string():
            return np.array(tensor_to_strings(vector.data()))
        else:
            return vector.data().numpy()


@dataclass
class LiteralValueVector(ColumnVector):
    dtype: ArrowType
    value: Any
    num_rows: int

    def field(self) -> Field:
        return self.dtype

    def get_value(self, i: int) -> Any:
        return self.value

    def size(self) -> int:
        return self.num_rows

    def data(self) -> torch.Tensor:
        return FieldVector.np_to_tensor(np.repeat(self.value, self.size()))
