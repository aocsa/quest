from typing import List

from data_source import DataSource
from expression import *
from logical_expr import *
from logical_plan import LogicalPlan, Scan, Selection, Projection
from schema import Schema


class PhysicalPlan(ABC):
    @abstractmethod
    def schema(self) -> Schema:
        pass

    @abstractmethod
    def execute(self) -> List[RecordBatch]:
        # Replace Sequence with the appropriate return type
        pass

    @abstractmethod
    def children(self) -> List['PhysicalPlan']:
        pass

    @abstractmethod
    def __str__(self) -> str:
        return self.pretty()

    def pretty(self, indent: int = 0) -> str:
        indentation = '\t' * indent
        result = f"{indentation}{str(self)}\n"
        for child in self.children():
            result += child.__str__()
            child.pretty(indent + 1)
        return result


class ScanExec(PhysicalPlan):
    def __init__(self, ds: DataSource, projection: List[str]):
        self.ds = ds
        self.projection = projection

    def schema(self) -> Schema:
        return self.ds.schema().select(self.projection)

    def children(self) -> List[PhysicalPlan]:
        return []

    def execute(self):
        # Implementation for ds.scan() will vary depending on DataSource's definition
        return self.ds.scan(self.projection)

    def __str__(self) -> str:
        return f"ScanExec({self.ds}, {self.projection})"


class SelectionExec(PhysicalPlan):
    def __init__(self, input_plan: PhysicalPlan, expr: Expression):
        self.input = input_plan
        self.expr = expr

    def schema(self) -> Schema:
        return self.input.schema()

    def children(self) -> List[PhysicalPlan]:
        return [self.input]

    def execute(self) -> List[RecordBatch]:
        import numpy as np
        # The actual implementation will depend on how `Expression` evaluates batches
        input_result = self.input.execute()
        # Example filtering logic, assuming input_result is iterable
        for batch in input_result:
            bit_mask: ColumnVector = self.expr.evaluate(batch)
            indices = np.nonzero(bit_mask.data())
            new_columns: List[ColumnVector] = []
            for index, column in enumerate(batch.columns_):
                if len(indices[0]) > 0:
                    col = np.take(batch.column(index).data(), indices)
                    vector = ArrowFieldVector(array=col, field=column.field())
                    new_columns.append(vector)
            if len(new_columns) > 0 and new_columns[0].size() > 0:
                rb = RecordBatch(batch.schema(), new_columns)
                yield rb

    def __str__(self) -> str:
        return f"SelectionExec({self.input}, {self.expr})"


class ProjectionExec(PhysicalPlan):
    def __init__(self, input_plan: PhysicalPlan, schema: Schema, expressions: List[Expression]):
        self.input = input_plan
        self.schema = schema
        self.expressions = expressions

    def execute(self) -> List[RecordBatch]:
        batches = self.input.execute()
        result = []

        for batch in batches:
            columns = [expression.evaluate(batch) for expression in self.expressions]
            result.append(RecordBatch(self.schema, columns))

        return result

    def schema(self) -> Schema:
        return self.input.schema()

    def children(self) -> List[PhysicalPlan]:
        return [self.input]

    def __str__(self) -> str:
        return f"ProjectionExec({self.input})"


def create_physical_plan(plan: LogicalPlan) -> PhysicalPlan:
    if isinstance(plan, Scan):
        return ScanExec(plan.data_source, plan.projection)
    elif isinstance(plan, Selection):
        input_plan = create_physical_plan(plan.children()[0])
        filter_expr = create_physical_expr(plan.condition, plan.children()[0])
        return SelectionExec(input_plan, filter_expr)
    elif isinstance(plan, Projection):
        input_plan = create_physical_plan(plan.children()[0])
        projection_expr = [create_physical_expr(expr, plan.children()[0]) for expr in plan.expressions]
        projection_schema = Schema([expr.to_field(plan.children()[0]) for expr in plan.expressions])
        return ProjectionExec(input_plan, projection_schema, projection_expr)
    # Implement other conditions as needed
    else:
        raise ValueError("Unsupported logical plan")


def create_physical_expr(expr: LogicalExpr, input_plan: LogicalPlan) -> Expression:
    if isinstance(expr, Literal):
        return LiteralExpression(expr.value)
    elif isinstance(expr, ColumnIndex):
        return ColumnExpression(expr.index)
    elif isinstance(expr, Column):
        index = input_plan.schema().index_of_first(expr.name)
        if index == -1:
            raise ValueError(f"No column named: {expr.name}")
        return ColumnExpression(index)
    elif isinstance(expr, BinaryExpr):
        l = create_physical_expr(expr.left, input_plan)
        r = create_physical_expr(expr.right, input_plan)
        if isinstance(expr, Eq):
            return EqExpression(l, r)
        elif isinstance(expr, Neq):
            return NeqExpression(l, r)
        elif isinstance(expr, Lt):
            return LtExpression(l, r)
        elif isinstance(expr, LtEq):
            return LtEqExpression(l, r)
        elif isinstance(expr, Gt):
            return GtExpression(l, r)
        elif isinstance(expr, GtEq):
            return GtEqExpression(l, r)
        elif isinstance(expr, And):
            return AndExpression(l, r)
        elif isinstance(expr, Or):
            return OrExpression(l, r)
    else:
        raise ValueError(f"Can't create Physical Expr: {expr}")
