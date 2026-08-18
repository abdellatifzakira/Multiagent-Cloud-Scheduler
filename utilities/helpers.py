import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import math
from sympy import symbols, lambdify
from sympy.parsing.sympy_parser import parse_expr



def plot_job_dags(jobs_list, figsize=(20, 5)):
    """Plot DAGs with hierarchical layout."""
    def closest_factors(n):
        if n == 0:
            return (0, 0)
        n_abs = abs(n)
        root = int(math.sqrt(n_abs))
        for x in range(root, 0, -1):
            if n_abs % x == 0:
                y = n_abs // x
                if n < 0:
                    return (-x, y)
                return (x, y)
        return (1, n)
    
    jobs = jobs_list.copy()
    num_jobs = len(jobs)
    if num_jobs > 6:
        jobs = jobs[:6]
        num_jobs = len(jobs)
    
    n, m = closest_factors(num_jobs)
    fig, axes = plt.subplots(n, m, figsize=figsize)
    
    # Handle single job case
    if num_jobs == 1:
        axes = [axes]
    else:
        axes = np.array(axes).reshape(-1)
    
    for idx, job in enumerate(jobs):
        ax = axes[idx]
        
        if job.data_transfer_weights is None:
            # All tasks as independent entries
            num_tasks = len(job.tasks)
            G = nx.DiGraph()
            G.add_nodes_from(range(num_tasks))
            
            pos = {}
            for node in range(num_tasks):
                x = (node % 3) - 1
                y = -(node // 3)
                pos[node] = (x, y)
            
            nx.draw_networkx_nodes(G, pos, node_color="#D4AF37", 
                                node_size=1500, ax=ax, 
                                edgecolors='black', linewidths=2)
            labels = {node: f"T{node}" for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
            
            # ← ADD THIS:
            max_row = (num_tasks - 1) // 3
            ax.set_xlim(-2, 2)
            ax.set_ylim(-max_row - 1, 1)
            
            ax.set_title(f"Job {job.id} (No DAG - Independent)", fontsize=11, fontweight='bold')
            ax.axis('off')
            continue
        
        # Convert igraph to networkx
        G = nx.DiGraph()
        
        for node_id in range(job.dag.vcount()):
            G.add_node(node_id)
        
        for edge in job.dag.es:
            source, target = edge.tuple
            weight = edge["weight"] if "weight" in edge.attributes() else 1
            G.add_edge(source, target, weight=weight)
        
        # Compute hierarchical levels
        levels = {}
        for node in G.nodes():
            if G.in_degree(node) == 0:
                levels[node] = 0
            else:
                preds = list(G.predecessors(node))
                levels[node] = 1 + max(levels.get(p, 0) for p in preds) if preds else 0
        
        # Hierarchical layout
        pos = {}
        max_level = max(levels.values()) if levels else 0
        
        for level in range(max_level + 1):
            nodes_at_level = [n for n, l in levels.items() if l == level]
            width = len(nodes_at_level)
            
            for i, node in enumerate(nodes_at_level):
                if width > 1:
                    x = -1 + 2 * (i / (width - 1))
                else:
                    x = 0
                y = -level
                pos[node] = (x, y)
        
        # Color nodes
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            in_degree = G.in_degree(node)
            out_degree = G.out_degree(node)
            
            if in_degree == 0 and out_degree == 0:
                node_colors.append('#FFD700')  # Gold: isolated
                node_sizes.append(1800)
            elif in_degree == 0:
                node_colors.append('#90EE90')  # Green: entry
                node_sizes.append(1500)
            elif out_degree == 0:
                node_colors.append('#FFB6C6')  # Red: exit
                node_sizes.append(1500)
            else:
                node_colors.append('#87CEEB')  # Blue: intermediate
                node_sizes.append(1200)
        
        # Draw
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=node_sizes, ax=ax, 
                              edgecolors='black', linewidths=1.5)
        
        nx.draw_networkx_edges(G, pos, ax=ax, arrowsize=15, 
                              arrowstyle='->', width=1.5, 
                              edge_color='#666666', connectionstyle='arc3,rad=0.1')
        
        # Labels
        labels = {node: f"T{node}" for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax, font_weight='bold')
        
        # Edge weights
        if G.edges():
            edge_labels = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6, ax=ax)
        
        ax.set_title(f"Job {job.id} ({G.number_of_nodes()} tasks)", fontsize=11, fontweight='bold')
        ax.axis('on')
    
    fig.suptitle('Job DAG Structures (Green=Entry, Blue=Intermediate, Red=Exit, Gold=Isolated)', 
                 fontsize=13, fontweight='bold', y=1.00)
    
    plt.tight_layout()
    plt.show()
    

def plot_server_network(farm):

    # ==============================
    # Convert igraph -> NetworkX
    # ==============================

    ig_G = farm.graph

    G = nx.Graph()

    # Copy nodes
    for node in ig_G.vs:
        G.add_node(
            node.index,
            **node.attributes()
        )

    # Copy edges
    for edge in ig_G.es:
        G.add_edge(
            edge.source,
            edge.target,
            **edge.attributes()
        )


    # ==============================
    # Layout
    # ==============================

    layout = nx.circular_layout(G)

    fig, ax = plt.subplots(figsize=(10, 8))


    # ==============================
    # Bandwidth classification
    # ==============================

    bandwidths = [
        G[u][v]["bandwidth"]
        for u, v in G.edges()
    ]

    min_bw = min(bandwidths)
    max_bw = max(bandwidths)

    step = (max_bw - min_bw) / 3

    low_threshold = min_bw + step
    high_threshold = min_bw + 2 * step


    edge_colors = []
    edge_widths = []


    for bw in bandwidths:

        if bw > high_threshold:
            # High bandwidth
            edge_colors.append("#FF0000")
            edge_widths.append(10)

        elif bw > low_threshold:
            # Medium bandwidth
            edge_colors.append("#FFA600")
            edge_widths.append(5)

        else:
            # Low bandwidth
            edge_colors.append("#000769")
            edge_widths.append(3)



    # ==============================
    # Node labels
    # ==============================

    node_labels = {}

    for node, data in G.nodes(data=True):

        cpu = data.get("cpu", "N/A")
        ram = data.get("ram", "N/A")

        node_labels[node] = (
            f"S{node}\n"
            f"CPU:{cpu}\n"
            f"RAM:{ram}"
        )


    # ==============================
    # Draw nodes
    # ==============================

    nx.draw_networkx_nodes(
        G,
        layout,
        node_color="lightblue",
        node_size=1200,
        ax=ax
    )


    # ==============================
    # Draw edges
    # ==============================

    nx.draw_networkx_edges(
        G,
        layout,
        edge_color=edge_colors,
        width=edge_widths,
        ax=ax
    )


    # ==============================
    # Draw node labels
    # ==============================

    nx.draw_networkx_labels(
        G,
        layout,
        labels=node_labels,
        font_color="blue",
        font_size=8,
        ax=ax
    )


    # ==============================
    # Edge bandwidth labels
    # ==============================

    edge_labels = {
        (u, v):
        f"{G[u][v]['bandwidth']}"
        for u, v in G.edges()
    }


    nx.draw_networkx_edge_labels(
        G,
        layout,
        edge_labels=edge_labels,
        font_size=8,
        label_pos=0.75,
        ax=ax
    )


    # ==============================
    # Final plot
    # ==============================

    plt.title("Server Farm Network Topology")

    plt.axis("off")
    plt.margins(0.25)

    plt.tight_layout()

    plt.show()
    

def parse_power_equation(expression_str):
    CPU, RAM, STORAGE = symbols('CPU RAM STORAGE')
    parsed_expr = parse_expr(expression_str)
    
    return lambdify([CPU, RAM, STORAGE], parsed_expr, modules="math")
