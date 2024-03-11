from typing import List

from field_vector import ColumnVector
from schema import Schema


class RecordBatch:
    def __init__(self, schema: Schema, columns: List[ColumnVector]) -> None:
        self.schema_ = schema
        self.columns_ = columns

    def row_count(self) -> int:
        return 0 if not self.columns_ else self.columns_[0].size()

    def column_count(self) -> int:
        return len(self.columns_)

    def column(self, i: int) -> ColumnVector:
        return self.columns_[i]

    def __str__(self) -> str:
        # return f"RecordBatch: {self.schema_.column_names()} | {self.columns_.data()}"
        seq = [col.data() for col in self.columns_]
        return f"{seq}"

    def schema(self) -> Schema:
        return self.schema_
