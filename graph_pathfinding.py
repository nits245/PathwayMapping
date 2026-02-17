import argparse
import math
import heapq
import itertools
import networkx as nx
from typing import Tuple, List
from datetime import datetime
from data_processing import DataProcessing
from traffic_flow_predictor import TrafficFlowPredictor
from graph_time_integration import enrich_graph_with_travel_times
from networkx_search_algorithms import ASTAR, Dijkstra, DFS, BFS, GBFS, IDASTAR
from map_plotter import generate_map_from_graph, show_map_gui


def parse_args():
    parser = argparse.ArgumentParser(
        description="Time-dependent pathfinding using SCATS data with custom algorithms"
    )
    parser.add_argument('model', help='Path to Keras model file or LSTM/GRU/CONVOLUTIONAL')
    parser.add_argument('algorithm',
                        choices=['ASTAR', 'DIJKSTRA', 'IDASTAR', 'GBFS', 'BFS', 'DFS'],
                        help='Search algorithm to use for pathfinding')
    parser.add_argument('time', help='Time of day in HHMM format, e.g. 0900')
    parser.add_argument('origin', type=int, help='Origin SCATS Number')
    parser.add_argument('destination', type=int, help='Destination SCATS Number')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of fastest routes to return')
    return parser.parse_args()


def compute_travel_time(G: nx.DiGraph, path: list[int]) -> float:
    """Sum travel_time_s over edges in the path."""
    return sum(
        G.edges[u, v].get('travel_time_s', 0.0)
        for u, v in zip(path, path[1:])
    )


def yen_k_shortest_paths(
    G: nx.DiGraph,
    source: int,
    target: int,
    Alg,
    K: int
    ) -> Tuple[List[List[int]], List[float]]:
    """
    Return up to K loopless paths from source to target using Yen's algorithm
    + spur-removal by SCATS number groups.
    G.nodes[n]['scats_number'] must exist for every node.
    """
    # helper to sum travel_time_s on a path in the original graph
    def compute_travel_time(path: List[int]) -> float:
        return sum(
            G.edges[u, v].get('travel_time_s', 0.0)
            for u, v in zip(path, path[1:])
        )

    # 1) first shortest path
    base = Alg(G, source, [target])
    found, _, _ = base.search()
    if target not in found:
        return [], []
    cost0, path0 = found[target]
    A = [path0]
    A_costs = [cost0]

    # min-heap of (cost, path) candidates
    B: List[Tuple[float, List[int]]] = []

    # build mapping: scats_number → set(nodes)
    scats_to_nodes: dict[int, set[int]] = {}
    for n, data in G.nodes(data=True):
        sn = data.get('scats_number')
        if sn is not None:
            scats_to_nodes.setdefault(sn, set()).add(n)

    # 2) generate 2nd..Kth paths
    for k_i in range(1, K):
        prev = A[k_i-1]

        for i in range(len(prev) - 1):
            spur_node = prev[i]
            spur_scats = G.nodes[spur_node]['scats_number']
            root = prev[:i+1]

            # make a fresh copy for spur search
            G_spur = G.copy()

            # 2.1) remove *all* edges between the same SCATS groups
            for p in A:
                if len(p) > i and p[:i+1] == root:
                    u_old, v_old = p[i], p[i+1]
                    su = G.nodes[u_old]['scats_number']
                    sv = G.nodes[v_old]['scats_number']
                    for u in scats_to_nodes.get(su, []):
                        for v in scats_to_nodes.get(sv, []):
                            if G_spur.has_edge(u, v):
                                G_spur.remove_edge(u, v)

            # 2.2) remove the root-path nodes (except spur_node) to avoid loops
            for n in root[:-1]:
                if G_spur.has_node(n):
                    G_spur.remove_node(n)

            # 2.3) spur-search from spur_node → target
            spur_alg = Alg(G_spur, spur_node, [target])
            spur_found, _, _ = spur_alg.search()
            if target in spur_found:
                cost_spur, spur_path = spur_found[target]
                # stitch root + spur (dropping duplicate spur_node)
                total_path = root[:-1] + spur_path
                total_cost = compute_travel_time(total_path)
                candidate = (total_cost, total_path)
                # avoid exact duplicates
                if candidate not in B:
                    heapq.heappush(B, candidate)

        if not B:
            break
        c, pth = heapq.heappop(B)
        A.append(pth)
        A_costs.append(c)

    return A, A_costs


def main():
    args = parse_args()

    # Resolve model name or path
    model_name = args.model.upper()
    if model_name == 'LSTM':
        model_path = 'models/lstm_model.keras'
    elif model_name == 'GRU':
        model_path = 'models/gru_model.keras'
    elif model_name == 'CONVOLUTIONAL':
        model_path = 'models/conv1d.keras'
    else:
        model_path = args.model

    # Parse HHMM into 'HH:MM'
    time_of_day = datetime.strptime(args.time, '%H%M').strftime('%H:%M')

    # Build and preprocess SCATS graph
    scats_fp = 'datasets/Scats Data October 2006.xls'
    dp = DataProcessing(scats_fp)
    dp.process_scats_data()
    G = dp.create_DiGraph()

    '''
        Force an edge between node 27 and node 138 - Victoria Street changes names to Barker's Road after 
        crossing the Yarra River at the Victoria Bridge, and there is no SCATS Data at that point to register
        and connect the intersections on either side. Due to this, we're forcing an edge between the nodes that
        would represent this road.
    '''

    if 27 in G.nodes and 138 in G.nodes:
        lat27, lon27 = G.nodes[27]['x'], G.nodes[27]['y']
        lat138, lon138 = G.nodes[138]['x'], G.nodes[138]['y']
        dist = dp.haversine_distance(lat27, lon27, lat138, lon138)
        # add bidirectional edge
        G.add_edge(27, 138, distance_km=dist, scat_point= G.nodes[27]['scats_number'])
        G.add_edge(138, 27, distance_km=dist, scat_point= G.nodes[138]['scats_number'])
    '''
        Same problem between node 119 and 134
    '''
    if 119 in G.nodes and 134 in G.nodes:
        lat27, lon27 = G.nodes[119]['x'], G.nodes[134]['y']
        lat138, lon138 = G.nodes[134]['x'], G.nodes[119]['y']
        dist = dp.haversine_distance(lat27, lon27, lat138, lon138)
        # add bidirectional edge
        G.add_edge(119, 134, distance_km=dist, scat_point= G.nodes[119]['scats_number'])
        G.add_edge(134, 119, distance_km=dist, scat_point= G.nodes[134]['scats_number'])
    '''
        SCATS 3001 suffers a similar issue to Node #27 and #138 - a road changes name but now at the intersection, 
        and of the 4 sensors: 2 refer to High Street and 2 refer to Church Street, even though they are both technically
        the same road. To solve this we force am edge between all 4 nodes present at that intersection.
    '''
    nodes_3001 = [nid for nid, data in G.nodes(data=True)
                  if data.get('scats_number') == 3001]
    # Generate all unique pairs
    for u, v in itertools.combinations(nodes_3001, 2):
        lat_u, lon_u = G.nodes[u]['x'], G.nodes[u]['y']
        lat_v, lon_v = G.nodes[v]['x'], G.nodes[v]['y']
        dist = dp.haversine_distance(lat_u, lon_u, lat_v, lon_v)
        # add bidirectional edges
        G.add_edge(u, v, distance_km=dist, scat_point=3001)
        G.add_edge(v, u, distance_km=dist, scat_point=3001)
    print(f"Forced edges added among nodes: {nodes_3001}")

    # Map SCATS Numbers to node IDs
    scats_to_nodes: dict[int, set[int]] = {}
    for u, v, data in G.edges(data=True):
        sensor = data.get('scat_point')
        if sensor is not None:
            scats_to_nodes.setdefault(sensor, set()).add(u)

    def find_closest_scats_pair(scats_from: int, scats_to: int) -> tuple[int, int]:
        orig_nodes = scats_to_nodes.get(scats_from)
        dest_nodes = scats_to_nodes.get(scats_to)
        if not orig_nodes or not dest_nodes:
            missing = scats_from if not orig_nodes else scats_to
            raise ValueError(f"No nodes for SCATS {missing}")
        return min(
            ((u, v,
              math.hypot(G.nodes[u]['x'] - G.nodes[v]['x'], G.nodes[u]['y'] - G.nodes[v]['y']))
             for u in orig_nodes for v in dest_nodes),
            key=lambda x: x[2]
        )[:2]

    # Enrich graph with travel times
    predictor = TrafficFlowPredictor(scats_fp, model_path)
    G = enrich_graph_with_travel_times(G, predictor, time_of_day)

    # Determine origin and destination nodes
    u, v = find_closest_scats_pair(args.origin, args.destination)

    # Select algorithm class
    alg_map = {
        'ASTAR': ASTAR, 'DIJKSTRA': Dijkstra, 'IDASTAR': IDASTAR,
        'GBFS': GBFS, 'BFS': BFS, 'DFS': DFS
    }
    Alg = alg_map[args.algorithm.upper()]

    # Find up to k fastest paths using Yen's algorithm
    k_paths, k_times = yen_k_shortest_paths(G, u, v, Alg, args.k)
    if not k_paths:
        print(f"No paths found from {args.origin} to {args.destination} using {args.algorithm}.")
        return

    # Print results
    print(f"Top {len(k_paths)} fastest routes from {args.origin} to {args.destination} "
          f"using {args.algorithm}:")
    for idx, (path, tt) in enumerate(zip(k_paths, k_times), start=1):
        print(f"{idx}. Path: {path} | Time: {tt:.1f} seconds")

    # Display map for the fastest route
    # best_path = k_paths[0]
    html_path = generate_map_from_graph(G, paths=k_paths)
    show_map_gui(html_path)


if __name__ == '__main__':
    main()
