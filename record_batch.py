from typing import List

from field_vector import ColumnVector, FieldVector
from schema import Schema

class RecordBatch:
    def __init__(self, schema: Schema, columns: List[ColumnVector]) -> None:
        self.schema_ = schema
        self.columns_ = columns

    def num_rows(self) -> int:
        return 0 if not self.columns_ else self.columns_[0].size()

    def num_columns(self) -> int:
        return len(self.columns_)

    def column_count(self) -> int:
        return len(self.columns_)

    def column(self, i: int) -> ColumnVector:
        return self.columns_[i]

    def __str__(self) -> str:
        # return f"RecordBatch: {self.schema_.column_names()} | {self.columns_.data()}"
        columns = [FieldVector(self.columns_[index].data(), self.schema_.field(index)) for index in range(self.num_columns())]
        
        table = {column.name() : FieldVector.to_numpy(column)   for column in columns}
        import pandas as pd
        df = pd.DataFrame(table)
        return f"{df}"

    def schema(self) -> Schema:
        return self.schema_
