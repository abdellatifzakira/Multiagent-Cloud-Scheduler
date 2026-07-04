from collections import deque
import igraph as ig

def get_dag_levels(graph: ig.Graph):
    in_degree = [graph.degree(v, mode="in") for v in range(graph.vcount())]

    queue = deque([v for v in range(graph.vcount()) if in_degree[v] == 0])

    levels = []

    while queue:
        current_level = list(queue)
        levels.append(current_level)

        next_queue = deque()

        for node in current_level:
            for child in graph.neighbors(node, mode="out"):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    next_queue.append(child)

        queue = next_queue

    return levels
    
    

def get_dag_level(dag : ig.Graph, level):
    return get_dag_levels(dag)[level]
