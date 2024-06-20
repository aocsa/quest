from typing import List

from field_vector import Field, ArrowType


class Schema:
    fields: List[Field]

    def __init__(self, fields: List[Field]) -> None:
        self.fields = fields

    def field(self, i: int) -> Field:
        return self.fields[i]

    def project(self, indices: List[int]) -> 'Schema':
        projected_fields: List[Field] = [self.fields[index] for index in indices]
        return Schema(projected_fields)

    def column_names(self) -> List[str]:
        return [field.name for field in self.fields]

    def column_types(self) -> List[ArrowType]:
        return [field.data_type for field in self.fields]

    def merge(self, other: 'Schema') -> 'Schema':
        merged_fields: List[Field] = self.fields + other.fields
        return Schema(merged_fields)

    def select(self, names: List[str]) -> 'Schema':
        selected_fields: List[Field] = []
        for name in names:
            matching_fields: List[Field] = [field for field in self.fields if field.name == name]
            if len(matching_fields) == 1:
                selected_fields.append(matching_fields[0])
            else:
                raise ValueError(f"Field {name} not found or ambiguous")
        return Schema(selected_fields)

    def index_of_first(self, col_name: str) -> int:
        for i, field in enumerate(self.fields):
            if field.name == col_name:
                return i
        return -1

    def __str__(self) -> str:
        return "\n".join([field.__str__() for field in self.fields])
