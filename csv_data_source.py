import os
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pyarrow as pa

from data_source import DataSource
from field_vector import ArrowFieldVector, numpy_dtype_to_arrow_dtype
from field_vector import Field
from record_batch import RecordBatch
from schema import Schema


def read_csv_schema(filename: str, delimiter: str) -> Schema:
    """
    Reads the first 'num_rows' of a CSV file using PyArrow.

    Parameters:
    - filename: Path to the CSV file.
    - num_rows: Number of rows to read from the start of the file. Defaults to 10.

    Returns:
    - A PyArrow Table containing the first 'num_rows' rows of the file.
    """
    num_rows = 10
    # todo use pyarrow
    # table = pv.read_csv(filename, read_options=pv.ReadOptions(autogenerate_column_names=True, skip_rows=0, block_size=1024), parse_options=pv.ParseOptions(delimiter=','))

    import pandas as pd
    df = pd.read_csv(filename, delimiter=delimiter)
    columns = df.columns
    fields = [[] for _ in columns]
    # dtype = [('id', 'i4'), ('name', 'U10'), ('dept', 'i4'), ('salary', 'f4'), ('tax', 'i4')]
    for index, col_name in enumerate(columns):
        fields[index] = Field(col_name, numpy_dtype_to_arrow_dtype(df[col_name].dtype))
    return Schema(fields)


@dataclass
class CsvParserSettings:
    delimiter: str = ','
    has_headers: bool = True
    batch_size: int = 10


class CsvDataSource(DataSource):

    def __init__(self, filename: str, settings: CsvParserSettings = None, schema: Optional[Schema] = None):
        self.filename = filename
        self.settings = settings
        self.schema_ = schema
        self.final_schema = None

        if not self.settings:
            self.settings = self.default_settings()

        if self.schema_ is None:
            self.final_schema = self.infer_schema()
        else:
            self.final_schema = self.schema_

    def default_settings(self) -> CsvParserSettings:
        return CsvParserSettings(delimiter=',')

    def infer_schema(self) -> Schema:
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"File not found: {self.filename}")
        return read_csv_schema(self.filename, self.settings.delimiter)

    def schema(self) -> Schema:
        return self.final_schema

    def scan(self, projection: List[str]) -> RecordBatch:
        batches = read_csv_in_batches(self.filename, self.settings.delimiter, self.settings.batch_size)
        if projection is not None and len(projection) > 0:
            self.final_schema = self.final_schema.select(projection)

        for batch in batches:
            if projection is not None and len(projection) > 0:
                batch = [column for column in batch if column.name() in projection]
            yield RecordBatch(self.final_schema, batch)

    def __str__(self) -> str:
        return f'CsvDataSource("{self.filename}")';


def read_csv(filename: str, delimiter: str = ',') -> [ArrowFieldVector]:
    import pandas as pd
    df = pd.read_csv(filename, delimiter=delimiter)
    columns = df.columns
    data = [[] for _ in columns]
    fields = [[] for _ in columns]
    # dtype = [('id', 'i4'), ('name', 'U10'), ('dept', 'i4'), ('salary', 'f4'), ('tax', 'i4')]
    for index, col_name in enumerate(columns):
        data[index] = df[col_name].to_numpy()
        fields[index] = Field(col_name, numpy_dtype_to_arrow_dtype(df[col_name].dtype))
    return [ArrowFieldVector(array=column, field=field) for field, column in zip(fields, data)]


def read_csv_in_batches(filename: str, delimiter: str, batch_size: int) -> [ArrowFieldVector]:
    import pandas as pd
    # Use chunksize to read the file in chunks.
    # Each chunk will be a DataFrame with 'batch_size' rows.
    for chunk in pd.read_csv(filename, delimiter=delimiter, chunksize=batch_size):
        # For each chunk, convert the columns to numpy arrays and yield batches
        columns = chunk.columns
        for start_row in range(0, chunk.shape[0], batch_size):
            batch_data = chunk.iloc[start_row:start_row + batch_size]
            # Splitting the chunk into batches if chunk size > batch size, 
            # although here each "chunk" is essentially a batch due to how we read the file.
            fields = [Field(col_name, numpy_dtype_to_arrow_dtype(batch_data[col_name].dtype))
                      for col_name in columns]
            data = [batch_data[col_name].to_numpy() for col_name in columns]
            yield [ArrowFieldVector(column, field) for column, field in zip(data, fields)]


def test_read_csv():
    filename = "test_data.csv"

    # write data 
    file_content = """
    column1,column2
    1,a
    2,b
    3,c
    """
    # Writing the content to the file
    with open(filename, "w") as file:
        file.write(file_content.strip())

    result = read_csv(filename)

    # Expected results based on the contents of test_data.csv
    # Adjust these based on the actual contents of your test CSV file
    expected_fields = [
        Field("column1", pa.int64()),
        Field("column2", pa.string()),
        # Add more fields based on your CSV structure
    ]
    expected_data = [
        np.array([1, 2, 3]),  # Example data for column1
        np.array(["a", "b", "c"]),  # Example data for column2
        # Add more arrays based on your CSV structure
    ]

    assert len(result) == len(expected_fields), "Number of fields mismatch"

    for arrow_array, expected_field, expected_column_data in zip(result, expected_fields, expected_data):
        assert arrow_array.name() == expected_field.name, "Field name mismatch"
        assert arrow_array.dtype() == expected_field.dtype, "Data type mismatch"
        np.testing.assert_array_equal(arrow_array.array, expected_column_data, "Column data mismatch")

    csv_source = CsvDataSource(filename=filename, settings=CsvParserSettings(batch_size=1))
    for batch in csv_source.scan(['column2']):
        record_batch: RecordBatch = batch
        assert record_batch.column(0).data() in ["a", "b", "c"]
