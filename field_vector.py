from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa

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
    def data(self) -> np.ndarray:
        pass

    @abstractmethod
    def size(self) -> int:
        pass


class ArrowFieldVector(ColumnVector):

    def __init__(self, array: np.ndarray, field: Field):
        self.array = array
        self.field_ = field

    def dtype(self):
        return self.field_.dtype

    def name(self):
        return self.field_.name

    def field(self) -> Field:
        return self.field_

    def get_value(self, i: int):
        return self.array[i]

    def size(self) -> int:
        return len(self.array)

    def data(self) -> np.ndarray:
        return self.array

    def __str__(self) -> str:
        return f"{self.array}"


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

    def data(self) -> np.ndarray:
        return np.repeat(self.value, self.size())


def numpy_dtype_to_arrow_dtype(numpy_dtype) -> pa.DataType:
    """
    Maps a NumPy dtype to an Apache Arrow data type.

    Parameters:
    - numpy_dtype: A NumPy data type (np.dtype).

    Returns:
    - An Apache Arrow data type (pa.DataType).
    """
    if numpy_dtype == np.dtype('int32'):
        return pa.int32()
    elif numpy_dtype == np.dtype('int64'):
        return pa.int64()
    elif numpy_dtype == np.dtype('float32'):
        return pa.float32()
    elif numpy_dtype == np.dtype('float64'):
        return pa.float64()
    elif numpy_dtype == np.dtype('bool'):
        return pa.bool_()
    elif numpy_dtype == np.dtype('object') or numpy_dtype.type == np.str_ or numpy_dtype.kind == 'U':
        return pa.string()
    elif numpy_dtype == np.dtype('datetime64[ms]'):
        return pa.timestamp('ms')
    else:
        raise ValueError(f"Unsupported NumPy dtype: {numpy_dtype}")
