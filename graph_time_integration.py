import networkx as nx
from flow_time_conversion import flow_to_travel_time
from traffic_flow_predictor import TrafficFlowPredictor

def enrich_graph_with_travel_times(
    G: nx.DiGraph,
    predictor: TrafficFlowPredictor,
    time_of_day: str
) -> nx.DiGraph:
    """
    Enrich each edge in G with a 'travel_time_s' attribute based on
    time-dependent traffic flow predictions.

    Args:
        G: Directed graph with 'scat_point' and 'distance_km' on edges.
        predictor: Initialized TrafficFlowPredictor.
        time_of_day: String 'HH:MM'.

    Returns:
        G with updated 'travel_time_s' on each edge.
    """
    flow_cache: dict[int, float] = {}

    for u, v, data in G.edges(data=True):
        sensor = data.get('scat_point')
        dist   = data.get('distance_km', 0.0)

        if sensor not in flow_cache:
            # One lightweight prediction per sensor
            raw_count = predictor.predict(sensor, time_of_day)
            # Convert 15-min count to vehicles/hour
            flow_cache[sensor] = raw_count * 4

        # Compute travel time using true flow
        data['travel_time_s'] = flow_to_travel_time(
            flow_cache[sensor],
            dist
        )

    return G
