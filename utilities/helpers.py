import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import math

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