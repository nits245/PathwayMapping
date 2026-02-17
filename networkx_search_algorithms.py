import heapq
import math
import networkx as nx
from collections import deque

class SearchAlgorithm:
    def __init__(self, graph, origin, destinations):
        self.graph = graph
        self.origin = origin
        self.destinations = destinations

    def search(self):
        raise NotImplementedError("Search method must be implemented in subclass")

class ASTAR(SearchAlgorithm):
    """
    A* implementation over a NetworkX DiGraph using travel-time as edge weights
    and Euclidean distance as the heuristic.
    """
    def search(self):
        print(f"[A*] Searching for path from [{self.origin}] to {self.destinations}")
        # Priority queue of (f_score, g_score, node, path)
        open_list = []
        # Best-known cost from origin to node (in seconds)
        g_scores = {self.origin: 0.0}
        # Push starting node with heuristic only
        heapq.heappush(open_list, (self._heuristic(self.origin), 0.0, self.origin, [self.origin]))
        closed = set()
        found = {}
        nodes_created = 0

        while open_list:
            f_score, g_score, node, path = heapq.heappop(open_list)
            if node in closed:
                continue
            closed.add(node)
            # If goal reached, record and possibly finish
            if node in self.destinations:
                found[node] = (g_score, path)
                if len(found) == len(self.destinations):
                    return found, nodes_created, None
            # Expand neighbors
            for nbr, attrs in self.graph[node].items():
                if nbr in closed:
                    continue
                # Edge cost: actual travel time in seconds
                cost = attrs.get('travel_time_s', float('inf'))
                tentative_g = g_score + cost
                if tentative_g < g_scores.get(nbr, float('inf')):
                    g_scores[nbr] = tentative_g
                    # f = g + h (Euclidean distance heuristic)
                    f = tentative_g + self._heuristic(nbr)
                    heapq.heappush(open_list, (f, tentative_g, nbr, path + [nbr]))
                    nodes_created += 1
        return found, nodes_created, None

    def _heuristic(self, node):
        """
        Euclidean straight-line distance (in same units as x,y) to the closest destination,
        used as an optimistic estimate (must be admissible for travel-time in seconds,
        so x,y should be projected to distance units or converted appropriately).
        """
        x1, y1 = self.graph.nodes[node]['x'], self.graph.nodes[node]['y']
        # Compute distances to each goal
        dists = []
        for dest in self.destinations:
            x2, y2 = self.graph.nodes[dest]['x'], self.graph.nodes[dest]['y']
            dists.append(math.hypot(x2 - x1, y2 - y1))
        return min(dists) if dists else 0.0


class Dijkstra(SearchAlgorithm):
    """
    Dijkstra's algorithm on a NetworkX DiGraph using 'travel_time_s' as edge weights.
    """
    def search(self):
        print(f"[Dijkstra] Searching from [{self.origin}] to {self.destinations}")
        # Min-heap of (cost, node, path)
        heap = [(0.0, self.origin, [self.origin])]
        # Best known costs
        g_scores = {node: float('inf') for node in self.graph.nodes}
        g_scores[self.origin] = 0.0
        visited = set()
        found_goals = {}
        nodes_created = 0

        while heap:
            cost, node, path = heapq.heappop(heap)
            if node in visited:
                continue
            visited.add(node)
            nodes_created += 1

            if node in self.destinations:
                found_goals[node] = (cost, path)
                if len(found_goals) == len(self.destinations):
                    return found_goals, nodes_created, None

            for nbr, attrs in self.graph[node].items():
                if nbr in visited:
                    continue
                edge_cost = attrs.get('travel_time_s', float('inf'))
                new_cost = cost + edge_cost
                if new_cost < g_scores.get(nbr, float('inf')):
                    g_scores[nbr] = new_cost
                    heapq.heappush(heap, (new_cost, nbr, path + [nbr]))
        return found_goals, nodes_created, None

class GBFS(SearchAlgorithm):
    """
    Greedy Best-First Search on a NetworkX DiGraph using Euclidean heuristic.
    """
    def search(self):
        print(f"[GBFS] Searching from [{self.origin}] to {self.destinations}")
        heap = [(self._heuristic(self.origin), self.origin, [self.origin])]
        visited = set()
        found_goals = {}
        nodes_created = 1

        while heap:
            h, node, path = heapq.heappop(heap)
            if node in visited:
                continue
            visited.add(node)

            if node in self.destinations:
                found_goals[node] = (len(path)-1, path)
                if len(found_goals) == len(self.destinations):
                    return found_goals, nodes_created, None

            for nbr in self.graph[node]:
                if nbr not in visited:
                    heapq.heappush(heap, (self._heuristic(nbr), nbr, path + [nbr]))
                    nodes_created += 1
        return found_goals, nodes_created, None

    def _heuristic(self, node):
        x1, y1 = self.graph.nodes[node]['x'], self.graph.nodes[node]['y']
        return min(
            math.hypot(self.graph.nodes[d]['x'] - x1, self.graph.nodes[d]['y'] - y1)
            for d in self.destinations
        )

class IDASTAR(SearchAlgorithm):
    """
    Iterative-Deepening A* on a NetworkX DiGraph using 'travel_time_s' and Euclidean heuristic.
    """
    def search(self):
        print(f"[IDA*] Searching for path from [{self.origin}] to {self.destinations}")
        found_goals = {}
        nodes_created = 0
        for goal in self.destinations:
            threshold = self._heuristic(self.origin, goal)
            path = [self.origin]
            g = 0.0
            while True:
                temp, result_path, nodes_created = self._dfs(path, g, threshold, goal, nodes_created)
                if result_path is not None:
                    found_goals[goal] = (self._path_cost(result_path), result_path)
                    break
                if temp == float('inf'):
                    break
                threshold = temp
        return found_goals, nodes_created, None

    def _dfs(self, path, g, threshold, goal, nodes_created):
        node = path[-1]
        f = g + self._heuristic(node, goal)
        if f > threshold:
            return f, None, nodes_created
        if node == goal:
            return threshold, path, nodes_created
        min_thresh = float('inf')
        for nbr, attrs in self.graph[node].items():
            if nbr not in path:
                nodes_created += 1
                cost = attrs.get('travel_time_s', float('inf'))
                t, res_path, nodes_created = self._dfs(path + [nbr], g + cost, threshold, goal, nodes_created)
                if res_path is not None:
                    return t, res_path, nodes_created
                min_thresh = min(min_thresh, t)
        return min_thresh, None, nodes_created

    def _heuristic(self, node, goal):
        x1, y1 = self.graph.nodes[node]['x'], self.graph.nodes[node]['y']
        x2, y2 = self.graph.nodes[goal]['x'], self.graph.nodes[goal]['y']
        # straight-line distance (km) → time (s)
        dist = math.hypot(x2 - x1, y2 - y1)
        return (dist / 60.0) * 3600.0 + 30.0

    def _path_cost(self, path):
        total = 0.0
        for u, v in zip(path, path[1:]):
            total += self.graph[u][v].get('travel_time_s', 0.0)
        return total

class BFS(SearchAlgorithm):
    """
    Breadth-First Search on a NetworkX DiGraph (unweighted).
    """
    def search(self):
        print(f"[BFS] Searching for path from [{self.origin}] to {self.destinations}")
        queue = deque([[self.origin]])
        visited = set()
        found_goals = {}
        nodes_created = 0

        while queue:
            path = queue.popleft()
            node = path[-1]
            if node in visited:
                continue
            visited.add(node)
            nodes_created += 1

            if node in self.destinations:
                found_goals[node] = (len(path)-1, path)
                if len(found_goals) == len(self.destinations):
                    return found_goals, nodes_created, None

            for nbr in sorted(self.graph[node].keys()):
                if nbr not in visited:
                    queue.append(path + [nbr])
        return found_goals, nodes_created, None

class DFS(SearchAlgorithm):
    """
    Depth-First Search on a NetworkX DiGraph (unweighted).
    """
    def search(self):
        print(f"[DFS] Searching for path from [{self.origin}] to {self.destinations}")
        stack = [[self.origin]]
        visited = set()
        found_goals = {}
        nodes_created = 0

        while stack:
            path = stack.pop()
            node = path[-1]
            if node in visited:
                continue
            visited.add(node)
            nodes_created += 1

            if node in self.destinations:
                found_goals[node] = (len(path)-1, path)
                if len(found_goals) == len(self.destinations):
                    return found_goals, nodes_created, None

            for nbr in sorted(self.graph[node].keys(), reverse=True):
                if nbr not in visited:
                    stack.append(path + [nbr])
        return found_goals, nodes_created, None
