import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mplpatch import *
from helpers import REPO_ROOT
import networkx as nx

save_plot = True

mpl_usetex(True)
mpl_style("seaborn")

def state_eq(t, xi, A, XI):
    N = A.shape[0]
    XI_matrix = xi.reshape((N, N))
    Wt = A * XI_matrix
    Dt = np.diag(Wt @ np.ones(N))
    Lt = Dt - Wt
    xidot = -np.kron(Lt, np.eye(N)) @ xi
    return xidot

initx = np.random.rand(9)

def main_TBCC2(topo):
    if topo == "cyclic":
        edges = [("A", "B"), ("B", "C"), ("C", "A")]
        title = "Cyclic Topology"
    elif topo == "cyclic_blink":
        edges = [("A", "B"), ("B", "C"), ("C", "A"), ("A", "C")]
        title = "Cyclic Topology with Back Link"
    elif topo == "star":
        edges = [("A", "B"), ("B", "A"), ("A", "C"), ("C", "A"), ("B", "C")]
        title = "Star Topology"
    else:
        raise ValueError("Invalid graph type")

    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    # Parameters
    N = 3  # number of agents

    # Construct adjacency matrix
    A = nx.adjacency_matrix(graph).toarray()

    XI = np.random.rand(N, N)
    np.fill_diagonal(XI, 1)

    W = A * XI

    # Simulation time
    ts = 10  # [sec]
    t_steps = 100
    tspan = np.linspace(0, ts, t_steps)

    # Solve the differential equation
    sol = solve_ivp(state_eq, [0, ts], initx, args=(A, XI), t_eval=tspan, rtol=1e-4, atol=1e-4*np.ones(N**2))

    fig, ax = plt.subplots(figsize=(5, 3))
    # Plotting
    for i in range(N):
        ax.plot(sol.t, sol.y[(i*N):(i+1)*N].T)

    # plt.gca().set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Trust')
    # plt.savefig(f'{savefilename}.pdf')
    mpl_render(f"truestnet_sim_{topo}.pdf" if save_plot else None)

if __name__ == "__main__":
    for topo in ["cyclic", "cyclic_blink", "star"]:
        main_TBCC2(topo)