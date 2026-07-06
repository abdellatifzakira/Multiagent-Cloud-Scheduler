import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
def plot_job_dags(jobs, figsize=(20, 5)):
    """Plot DAGs horizontally side-by-side with better isolated node handling."""
    num_jobs = len(jobs)
    
    # Horizontal layout: 1 row, multiple columns
    fig, axes = plt.subplots(1, num_jobs, figsize=figsize)
    
    # Handle single job case
    if num_jobs == 1:
        axes = np.array([axes])
    else:
        axes = np.array(axes)
    
    for idx, job in enumerate(jobs):
        ax = axes[idx]
        
        if job.data_transfer_weights is None:
            # All tasks as independent entries
            num_tasks = len(job.tasks)
            G = nx.DiGraph()
            G.add_nodes_from(range(num_tasks))
            
            pos = {}
            for node in range(num_tasks):
                x = (node % 3) - 1  # 3 columns
                y = -(node // 3)    # multiple rows
                pos[node] = (x, y)
            
            nx.draw_networkx_nodes(G, pos, node_color="#D4AF37", 
                                  node_size=1500, ax=ax, 
                                  edgecolors='black', linewidths=2)
            labels = {node: f"T{node}" for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
            
            ax.set_title(f"Job {job.id} (No DAG - Independent)", fontsize=11, fontweight='bold')
            ax.axis('off')
            continue
        
        # Convert igraph to networkx - ADD ALL NODES
        G = nx.DiGraph()
        
        # Add ALL nodes first (including isolated ones)
        for node_id in range(job.dag.vcount()):
            G.add_node(node_id)
        
        # Then add edges
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
        
        # Count nodes per level
        level_counts = {}
        for node, level in levels.items():
            level_counts[level] = level_counts.get(level, 0) + 1
        
        # Assign positions
        pos = {}
        level_positions = {level: 0 for level in level_counts}
        max_level = max(levels.values()) if levels else 0
        
        for node, level in levels.items():
            x = level_positions[level] - level_counts[level] / 2
            y = max_level - level
            pos[node] = (x, y)
            level_positions[level] += 1
        
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
        
        # Edge weights (only if there are edges)
        if G.edges():
            edge_labels = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6, ax=ax)
        
        ax.set_title(f"Job {job.id} ({G.number_of_nodes()} tasks)", fontsize=11, fontweight='bold')
        ax.axis('off')
        ax.set_aspect('equal')

        xs = [pos[n][0] for n in G.nodes()]
        ys = [pos[n][1] for n in G.nodes()]

        x_center = (max(xs) + min(xs)) / 2
        y_center = (max(ys) + min(ys)) / 2

        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)

        pad = 0.5

        ax.set_xlim(x_center - x_range/2 - pad, x_center + x_range/2 + pad)
        ax.set_ylim(y_center - y_range/2 - pad, y_center + y_range/2 + pad)
    
    
    fig.suptitle('Job DAG Structures (Green=Entry, Blue=Intermediate, Red=Exit, Gold=Isolated)', 
                 fontsize=13, fontweight='bold', y=1.00)
    
    plt.tight_layout()
    plt.show()