import os
from dataclasses import dataclass
from typing import Optional, List

import torch
import numpy as np
import pyarrow as pa
import pandas as pd
from data_source import DataSource
from field_vector import FieldVector
from field_vector import Field
from record_batch import RecordBatch
from schema import Schema
from dtypes import numpy_dtype_to_arrow_dtype
from tensor_df_utils import tensor_to_strings


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

def convert_df_to_afv(df:pd.DataFrame) -> List[FieldVector]:
    columns = df.columns
    field_vectors = []
    for col_name in columns:
        array = df[col_name].to_numpy()
        field = Field(col_name, numpy_dtype_to_arrow_dtype(array.dtype))
        field_vector = FieldVector(FieldVector.np_to_tensor(array), field)
        field_vectors.append(field_vector)
    return field_vectors


def read_csv(filename: str, delimiter: str = ',') -> List[FieldVector]:
    df = pd.read_csv(filename, delimiter=delimiter)
    return convert_df_to_afv(df)


def read_csv_in_batches(filename: str, delimiter: str, batch_size: int) -> List[FieldVector]:
    # Use chunksize to read the file in chunks.
    # Each chunk will be a DataFrame with 'batch_size' rows.
    for chunk in pd.read_csv(filename, delimiter=delimiter, chunksize=batch_size):
        # For each chunk, convert the columns to numpy arrays and yield batches
        columns = chunk.columns
        for start_row in range(0, chunk.shape[0], batch_size):
            batch_data = chunk.iloc[start_row:start_row + batch_size]
            yield convert_df_to_afv(batch_data)


def test_read_csv():
    filename = "test_data.csv"

    # write data 
    file_content = """
    column1,column2
    1,a
    2,bcd
    3,cdefg
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
        np.array(["a", "bcd", "cdefg"]),  # Example data for column2
    ]

    expected_data = [
        FieldVector.np_to_tensor(expected_data[0]),  
        FieldVector.np_to_tensor(expected_data[1]),   
    ]

    assert len(result) == len(expected_fields), "Number of fields mismatch"

    for column, expected_field, expected_column_data in zip(result, expected_fields, expected_data):        
        assert column.field().name == expected_field.name, "Field name mismatch"
        assert column.field().dtype == expected_field.dtype, "Data type mismatch"
        np.testing.assert_array_equal(column.data(), expected_column_data, "Column data mismatch")
    index = 0
    csv_source = CsvDataSource(filename=filename, settings=CsvParserSettings(batch_size=1))
    for batch in csv_source.scan(['column2']):
        record_batch: RecordBatch = batch
        print(record_batch)
        read_string_column = tensor_to_strings(record_batch.column(0).data())
        expected_string_column = tensor_to_strings(expected_data[1][index])
        assert read_string_column == expected_string_column, "Column data mismatch"
        index += 1