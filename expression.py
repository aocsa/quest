from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa

from field_vector import Field, ColumnVector, LiteralValueVector, ArrowFieldVector, numpy_dtype_to_arrow_dtype
from record_batch import RecordBatch


class Expression(ABC):
    @abstractmethod
    def evaluate(self, input: RecordBatch) -> ColumnVector:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


@dataclass
class LiteralExpression(Expression):
    value: Any

    def evaluate(self, input: RecordBatch) -> ColumnVector:
        import numpy as np
        val = np.repeat(self.value, 1)

        dtype = numpy_dtype_to_arrow_dtype(val.dtype)
        return LiteralValueVector(dtype, self.value, input.row_count())  # Assuming a compatible constructor

    def __str__(self) -> str:
        return str(self.value)


@dataclass
class ColumnExpression(Expression):
    index: int

    def evaluate(self, input: RecordBatch) -> ColumnVector:
        return input.column(self.index)  # Assuming RecordBatch has a field() method

    def __str__(self) -> str:
        return f"#{self.index}"


class BooleanExpression(Expression):
    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def evaluate(self, input: RecordBatch) -> ColumnVector:
        ll = self.left.evaluate(input)
        rr = self.right.evaluate(input)
        np_bool_arr = self.evaluate_boolean_expression(ll.data(), rr.data())
        assert np_bool_arr.dtype == np.bool_
        assert len(np_bool_arr) == ll.size()
        assert len(np_bool_arr) == rr.size()
        return ArrowFieldVector(array=np_bool_arr, field=Field("result", pa.bool_))

    def evaluate_boolean_expression(self, l: np.ndarray, r: np.ndarray) -> np.ndarray:
        pass


class AndExpression(BooleanExpression):
    def evaluate_boolean_expression(self, l: np.ndarray, r: np.ndarray) -> np.ndarray:
        return l.__and__(r)

    def __str__(self) -> str:
        return f"{self.left} and {self.right}"


class OrExpression(BooleanExpression):
    def evaluate_boolean_expression(self, l: np.ndarray, r: np.ndarray) -> np.ndarray:
        return l.__or__(r)

    def __str__(self) -> str:
        return f"{self.left} or {self.right}"


class EqExpression(BooleanExpression):
    def evaluate_boolean_expression(self, l: np.ndarray, r: np.ndarray) -> np.ndarray:
        return l.__eq__(r)

    def __str__(self) -> str:
        return f"{self.left} == {self.right}"


class NeqExpression(BooleanExpression):
    def evaluate_boolean_expression(self, l: np.ndarray, r: np.ndarray) -> np.ndarray:
        return l.__ne__(r)

    def __str__(self) -> str:
        return f"{self.left} != {self.right}"


class LtExpression(BooleanExpression):
    def evaluate_boolean_expression(self, l: np.ndarray, r: np.ndarray) -> np.ndarray:
        return l.__lt__(r)

    def __str__(self) -> str:
        return f"{self.left} < {self.right}"


class LtEqExpression(BooleanExpression):
    def evaluate_boolean_expression(self, l: np.ndarray, r: np.ndarray) -> np.ndarray:
        return l.__le__(r)

    def __str__(self) -> str:
        return f"{self.left} <= {self.right}"


class GtExpression(BooleanExpression):
    def evaluate_boolean_expression(self, l: np.ndarray, r: np.ndarray) -> np.ndarray:
        return l.__gt__(r)

    def __str__(self) -> str:
        return f"{self.left} > {self.right}"


class GtEqExpression(BooleanExpression):
    def evaluate_boolean_expression(self, l: np.ndarray, r: np.ndarray) -> np.ndarray:
        return l.__ge__(r)

    def __str__(self) -> str:
        return f"{self.left} >= {self.right}"
