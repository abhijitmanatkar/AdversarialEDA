from benchmark.atena.simulation.actions import (
    GroupAction,
    Column,
    AggregationFunction,
    BackAction,
    FilterAction,
    FilterOperator,
)

reference1 = [
    GroupAction(grouped_column=Column('airline'), aggregated_column=Column('flight_id'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('departure_delay'), aggregated_column=Column('flight_id'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    GroupAction(grouped_column=Column('origin_airport'), aggregated_column=Column('flight_id'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('departure_delay'), aggregated_column=Column('flight_id'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    BackAction(),
    BackAction(),
    GroupAction(grouped_column=Column('delay_reason'), aggregated_column=Column('flight_id'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('departure_delay'), aggregated_column=Column('flight_id'),
                aggregation_function=AggregationFunction.COUNT),
    FilterAction(filtered_column=Column('departure_delay'), filter_operator=FilterOperator.EQUAL,
                 filter_term='LARGE_DELAY'),
    GroupAction(grouped_column=Column('origin_airport'), aggregated_column=Column('flight_id'),
                aggregation_function=AggregationFunction.COUNT),
]

reference2 = [
    GroupAction(grouped_column=Column('airline'), aggregated_column=Column('flight_id'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('origin_airport'), aggregated_column=Column('flight_id'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('departure_delay'), aggregated_column=Column('flight_id'),
                aggregation_function=AggregationFunction.COUNT),
    FilterAction(filtered_column=Column('departure_delay'), filter_operator=FilterOperator.NOTEQUAL,
                 filter_term='ON_TIME'),
    BackAction(),
    BackAction(),
    BackAction(),
    BackAction(),
    GroupAction(grouped_column=Column('delay_reason'), aggregated_column=Column('flight_id'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('departure_delay'), aggregated_column=Column('flight_id'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('origin_airport'), aggregated_column=Column('flight_id'),
                aggregation_function=AggregationFunction.COUNT),
    FilterAction(filtered_column=Column('departure_delay'), filter_operator=FilterOperator.EQUAL,
                 filter_term='LARGE_DELAY'),
]

reference3 = [
    GroupAction(grouped_column=Column('delay_reason'), aggregated_column=Column('flight_id'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('departure_delay'), aggregated_column=Column('flight_id'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('airline'), aggregated_column=Column('flight_id'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    FilterAction(filtered_column=Column('delay_reason'), filter_operator=FilterOperator.EQUAL, filter_term='WEATHER'),
    GroupAction(grouped_column=Column('origin_airport'), aggregated_column=Column('flight_id'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    GroupAction(grouped_column=Column('day_of_year'), aggregated_column=Column('flight_id'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    BackAction(),
    BackAction(),
    GroupAction(grouped_column=Column('day_of_week'), aggregated_column=Column('flight_id'),
                aggregation_function=AggregationFunction.COUNT),
]

reference4 = [
    GroupAction(grouped_column=Column('departure_delay'), aggregated_column=Column('flight_id'),
                aggregation_function=AggregationFunction.COUNT),
    FilterAction(filtered_column=Column('departure_delay'), filter_operator=FilterOperator.EQUAL,
                 filter_term='ON_TIME'),
    GroupAction(grouped_column=Column('airline'), aggregated_column=Column('flight_id'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    GroupAction(grouped_column=Column('origin_airport'), aggregated_column=Column('flight_id'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    GroupAction(grouped_column=Column('day_of_week'), aggregated_column=Column('flight_id'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    BackAction(),
    BackAction(),
    GroupAction(grouped_column=Column('day_of_year'), aggregated_column=Column('flight_id'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('delay_reason'), aggregated_column=Column('flight_id'),
                aggregation_function=AggregationFunction.COUNT),
]


reference5 = [
    GroupAction(grouped_column=Column('delay_reason'), aggregated_column=Column('flight_id'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('departure_delay'), aggregated_column=Column('flight_id'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('airline'), aggregated_column=Column('flight_id'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    FilterAction(filtered_column=Column('delay_reason'), filter_operator=FilterOperator.EQUAL, filter_term='WEATHER'),
    GroupAction(grouped_column=Column('origin_airport'), aggregated_column=Column('flight_id'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    GroupAction(grouped_column=Column('day_of_year'), aggregated_column=Column('flight_id'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    BackAction(),
    BackAction(),
    GroupAction(grouped_column=Column('day_of_week'), aggregated_column=Column('flight_id'),
                aggregation_function=AggregationFunction.COUNT),
]


flights4_references = [reference1, reference2, reference3, reference4, reference5]

assert all([len(reference) == 12 for reference in flights4_references])

