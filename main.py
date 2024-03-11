from csv_data_source import CsvDataSource, CsvParserSettings
from physical_plan import *

if __name__ == '__main__':
    # SELECT *
    # from employee WHERE
    # salary = 100
    # OR
    # tax = 40

    filename = 'employee.csv'

    csv_source = CsvDataSource(filename=filename, settings=CsvParserSettings(batch_size=1))
    scan = Scan("employee", csv_source, None)

    salary_filter_expression = eq(col("salary"), lit(200))
    tax_filter_expression = eq(col("tax"), lit(40))
    filter_expression = Or(salary_filter_expression, tax_filter_expression)

    selection = Selection(scan, filter_expression)

    projection_columns = [col("id"), col("salary"), col("tax"), col("name")]
    query_plan = Projection(selection, projection_columns);

    physical_plan = create_physical_plan(query_plan)
    print(physical_plan)
    query_result = physical_plan.execute()
    for (i, batch) in enumerate(query_result):
        print(f"batch {i}| data {batch}")
