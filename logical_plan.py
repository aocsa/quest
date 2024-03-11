from abc import ABC, abstractmethod
from typing import List

from csv_data_source import DataSource
from schema import Schema


class LogicalPlan(ABC):
    @abstractmethod
    def schema(self) -> Schema:
        pass

    @abstractmethod
    def children(self) -> List['LogicalPlan']:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


class Scan(LogicalPlan):
    def __init__(self, path: str, data_source: DataSource, projection: List[str] = None):
        self.path = path
        self.data_source = data_source
        self.projection = projection
        self._schema = self.derive_schema()

    def schema(self) -> Schema:
        return self._schema

    def children(self) -> List[LogicalPlan]:
        return []

    def __str__(self) -> str:
        if self.projection:
            proj_str = ", ".join(self.projection)
            return f"Scan: {self.path}; projection={proj_str}"
        return f"Scan: {self.path}; projection=None"

    def derive_schema(self) -> Schema:
        schema = self.data_source.schema()
        if self.projection:
            return schema.select(self.projection)
        return schema


class Selection(LogicalPlan):
    def __init__(self, input_plan: LogicalPlan, condition: 'LogicalExpr'):
        self.input_plan = input_plan
        self.condition = condition

    def schema(self) -> Schema:
        # Selection does not alter the schema
        return self.input_plan.schema()

    def children(self) -> List[LogicalPlan]:
        return [self.input_plan]

    def __str__(self) -> str:
        return f"Selection: {str(self.condition)}"


class Projection(LogicalPlan):
    def __init__(self, input_plan: LogicalPlan, expressions: List['LogicalExpr']):
        self.input_plan = input_plan
        self.expressions = expressions

    def schema(self) -> Schema:
        # Projection alters the schema based on the expressions
        fields = [expr.to_field(self.input_plan) for expr in self.expressions]
        return Schema(fields=fields)

    def children(self) -> List[LogicalPlan]:
        return [self.input_plan]

    def __str__(self) -> str:
        exprs_str = ", ".join(str(expr) for expr in self.expressions)
        return f"Projection: {exprs_str}"
