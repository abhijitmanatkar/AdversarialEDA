from benchmark.atena.simulation.actions import (
    GroupAction,
    Column,
    AggregationFunction,
    BackAction,
    FilterAction,
    FilterOperator,
)

reference1 = [
    GroupAction(grouped_column=Column('eth_src'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    GroupAction(grouped_column=Column('highest_layer'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    FilterAction(filtered_column=Column('tcp_dstport'), filter_operator=FilterOperator.EQUAL, filter_term='80'),
    GroupAction(grouped_column=Column('tcp_stream'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    BackAction(),
    FilterAction(filtered_column=Column('info_line'), filter_operator=FilterOperator.CONTAINS,
                 filter_term='load.php?e=1'),
    BackAction(),
    FilterAction(filtered_column=Column('tcp_stream'), filter_operator=FilterOperator.EQUAL, filter_term='5'),
    BackAction(),
]

reference2 = [
    GroupAction(grouped_column=Column('eth_src'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('ip_src'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('highest_layer'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    BackAction(),
    BackAction(),
    FilterAction(filtered_column=Column('eth_src'), filter_operator=FilterOperator.EQUAL,
                 filter_term='52:54:00:12:35:00'),
    GroupAction(grouped_column=Column('highest_layer'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    BackAction(),
    FilterAction(filtered_column=Column('highest_layer'), filter_operator=FilterOperator.NOTEQUAL, filter_term='TCP'),
    FilterAction(filtered_column=Column('info_line'), filter_operator=FilterOperator.CONTAINS, filter_term='HTTP'),
]

reference3 = [
    GroupAction(grouped_column=Column('ip_src'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('ip_dst'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    GroupAction(grouped_column=Column('eth_src'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    BackAction(),
    GroupAction(grouped_column=Column('highest_layer'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    FilterAction(filtered_column=Column('highest_layer'), filter_operator=FilterOperator.EQUAL, filter_term='NBNS'),
    GroupAction(grouped_column=Column('ip_src'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('ip_dst'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('eth_dst'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
]

reference4 = [
    GroupAction(grouped_column=Column('eth_src'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('tcp_srcport'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('tcp_dstport'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    BackAction(),
    BackAction(),
    FilterAction(filtered_column=Column('tcp_dstport'), filter_operator=FilterOperator.EQUAL, filter_term='80'),
    FilterAction(filtered_column=Column('highest_layer'), filter_operator=FilterOperator.EQUAL, filter_term='HTTP'),
    GroupAction(grouped_column=Column('ip_dst'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('tcp_stream'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    BackAction(),
]

reference5 = [
    GroupAction(grouped_column=Column('ip_src'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('ip_dst'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    BackAction(),
    GroupAction(grouped_column=Column('highest_layer'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('length'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    BackAction(),
    FilterAction(filtered_column=Column('highest_layer'), filter_operator=FilterOperator.EQUAL, filter_term='NBNS'),
    BackAction(),
    FilterAction(filtered_column=Column('ip_dst'), filter_operator=FilterOperator.EQUAL, filter_term='192.168.56.52'),
    FilterAction(filtered_column=Column('highest_layer'), filter_operator=FilterOperator.EQUAL, filter_term='HTTP'),
]

reference6 = [
    GroupAction(grouped_column=Column('eth_src'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('ip_src'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    BackAction(),
    FilterAction(filtered_column=Column('eth_src'), filter_operator=FilterOperator.EQUAL,
                 filter_term='52:54:00:12:35:00'),
    GroupAction(grouped_column=Column('ip_dst'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('highest_layer'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('ip_src'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    BackAction(),
    BackAction(),
    FilterAction(filtered_column=Column('highest_layer'), filter_operator=FilterOperator.NOTEQUAL, filter_term='TCP'),
]

reference7 = [
    GroupAction(grouped_column=Column('eth_src'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('ip_src'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    BackAction(),
    GroupAction(grouped_column=Column('highest_layer'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    BackAction(),
    FilterAction(filtered_column=Column('highest_layer'), filter_operator=FilterOperator.EQUAL, filter_term='DNS'),
    BackAction(),
    FilterAction(filtered_column=Column('info_line'), filter_operator=FilterOperator.CONTAINS, filter_term='GET'),
    GroupAction(grouped_column=Column('ip_src'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('ip_dst'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
    GroupAction(grouped_column=Column('tcp_stream'), aggregated_column=Column('packet_number'),
                aggregation_function=AggregationFunction.COUNT),
]


cyber3_references = [reference1, reference2, reference3, reference4, reference5, reference6, reference7]

assert all([len(reference) == 12 for reference in cyber3_references])
