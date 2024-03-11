from abc import ABC, abstractmethod
from typing import Any

from field_vector import Field
from logical_plan import LogicalPlan


class LogicalExpr(ABC):
    @abstractmethod
    def to_field(self, input_plan: LogicalPlan) -> Field:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


class Column(LogicalExpr):
    def __init__(self, name: str):
        self.name = name

    def to_field(self, input_plan: LogicalPlan) -> Field:
        for field in input_plan.schema().fields:
            if field.name == self.name:
                return field
        raise ValueError(f"No column named '{self.name}' in schema")

    def __str__(self) -> str:
        return f"#{self.name}"


class ColumnIndex(LogicalExpr):
    def __init__(self, index: int):
        self.index = index

    def to_field(self, input_plan: LogicalPlan) -> Field:
        if self.index < 0 or self.index >= len(input_plan.schema().fields):
            raise IndexError("Column index out of range")
        return input_plan.schema().fields[self.index]

    def __str__(self) -> str:
        return f"#{self.index}"


def col(name: str) -> Column:
    return Column(name=name)


class Literal(LogicalExpr):
    def __init__(self, value: Any):
        self.value = value
        # Simplified: Directly using value's type as dtype for demonstration
        self.dtype = type(value)

    def to_field(self, input_plan: LogicalPlan) -> Field:
        # Assuming literals do not depend on the input schema for their field
        return Field(name="literal", dtype=self.dtype)

    def __str__(self) -> str:
        return str(self.value)


def lit(value: Any) -> Literal:
    return Literal(value=value)


# Binary expressions like And, Or, Eq are implemented similarly
class BinaryExpr(LogicalExpr):
    def __init__(self, left: LogicalExpr, right: LogicalExpr, op: str):
        self.left = left
        self.right = right
        self.op = op

    def to_field(self, input_plan: LogicalPlan) -> Field:
        # Simplification: Assuming binary expressions result in a boolean field
        return Field(name=f"{str(self.left)} {self.op} {str(self.right)}", dtype=bool)

    def __str__(self) -> str:
        return f"({str(self.left)} {self.op} {str(self.right)})"


class BooleanBinaryExpr(BinaryExpr):
    def __init__(self, left: LogicalExpr, right: LogicalExpr, op: str):
        self.left = left
        self.right = right
        self.op = op

    def to_field(self, input_plan: LogicalPlan) -> Field:
        # Simplification: Assuming binary expressions result in a boolean field
        return Field(name=f"{str(self.left)} {self.op} {str(self.right)}", dtype=bool)

    def __str__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


class And(BooleanBinaryExpr):
    def __init__(self, left: LogicalExpr, right: LogicalExpr):
        super().__init__(left, right, "AND")


class Or(BooleanBinaryExpr):
    def __init__(self, left: LogicalExpr, right: LogicalExpr):
        super().__init__(left, right, "OR")


class Eq(BooleanBinaryExpr):
    def __init__(self, left: LogicalExpr, right: LogicalExpr):
        super().__init__(left, right, "==")


class Neq(BooleanBinaryExpr):
    def __init__(self, left: LogicalExpr, right: LogicalExpr):
        super().__init__(left, right, "!=")


class Gt(BooleanBinaryExpr):
    def __init__(self, left: LogicalExpr, right: LogicalExpr):
        super().__init__(left, right, ">")


class GtEq(BooleanBinaryExpr):
    def __init__(self, left: LogicalExpr, right: LogicalExpr):
        super().__init__(left, right, ">=")


class Lt(BooleanBinaryExpr):
    def __init__(self, left: LogicalExpr, right: LogicalExpr):
        super().__init__(left, right, "<")


class LtEq(BooleanBinaryExpr):
    def __init__(self, left: LogicalExpr, right: LogicalExpr):
        super().__init__(left, right, "<=")


# Convenience functions for expressions
def eq(left: LogicalExpr, right: LogicalExpr) -> Eq:
    return Eq(left, right)


def neq(left: LogicalExpr, right: LogicalExpr) -> Neq:
    return Neq(left, right)


def gt(left: LogicalExpr, right: LogicalExpr) -> Gt:
    return Gt(left, right)


def gte(left: LogicalExpr, right: LogicalExpr) -> GtEq:
    return GtEq(left, right)


def lt(left: LogicalExpr, right: LogicalExpr) -> Lt:
    return Lt(left, right)


def lte(left: LogicalExpr, right: LogicalExpr) -> LtEq:
    return LtEq(left, right)
