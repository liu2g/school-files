#!venv/bin/python3

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from mplpatch import *
from helpers import REPO_ROOT

save_plot = True

mpl_usetex(True)
mpl_style("seaborn")

def main(topo):
    if topo == "fc":
        edges = [("A", "B"), ("B", "A"), ("A", "C"), ("C", "A"), ("B", "C"), ("C", "B")]
    elif topo == "cyclic":
        edges = [("A", "B"), ("B", "C"), ("C", "A")]
        title = "Cyclic Topology"
    elif topo == "cyclic_blink":
        edges = [("A", "B"), ("B", "C"), ("C", "A"), ("A", "C"), ("C", "B")]
        title = "Cyclic Topology with Back Link"
    elif topo == "star":
        edges = [("A", "B"), ("B", "A"), ("A", "C"), ("C", "A"), ("B", "C")]
        title = "Star Topology"
    else:
        raise ValueError("Invalid graph type")

    # Create a directed graph
    graph = nx.DiGraph()
    graph.add_edges_from(edges)

    # Initialize states for agents A, B, C
    agents = list(graph.nodes)
    num_steps = 15
    initial_states = {"A": 0.3, "B": 0.5, "C": 0.8}
    states = {node: [initial_states[node]] for node in agents}

    # Define the simulation logic
    for _ in range(num_steps):
        new_states = {}
        for agent in agents:
            neighbors = list(graph.predecessors(agent))  # Incoming edges
            new_states[agent] = np.mean([states[neighbor][-1] for neighbor in neighbors])
        for agent in agents:
            states[agent].append(new_states[agent])

    # Plot the state of each agent over time using the OOP API
    fig, ax = plt.subplots(figsize=(5, 3))
    for agent, state_history in states.items():
        ax.plot(state_history, label=f"{agent}", marker='o')

    ax.set_xlabel("Steps")
    ax.set_xticks(range(num_steps + 1))
    ax.set_ylabel("State")
    # ax.set_title(title)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True)
    fig.tight_layout()

    mpl_render(
        REPO_ROOT / f"figs/diagrams/graphsim-{topo}.pdf"
        if save_plot else None
    )

if __name__ == "__main__":
    for topo in ["cyclic", "cyclic_blink", "star"]:
        main(topo)
