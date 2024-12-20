#!venv/bin/python3

from matplotlib import pyplot as plt
import numpy as np
from helpers import REPO_ROOT
from mplpatch import *

save_plot = True
figsize = (6, 4)

mpl_usetex(True)
mpl_style("seaborn")
mpl_fontsize(15)

def plot(prefix: str):
    data_s = np.loadtxt(REPO_ROOT / "data" / f"{prefix}_romi_front.csv", delimiter=",")
    data_m = np.loadtxt(REPO_ROOT / "data" / f"{prefix}_romi_left.csv", delimiter=",")
    data_n = np.loadtxt(REPO_ROOT / "data" / f"{prefix}_romi_right.csv", delimiter=",")
    min_len = min(data_s.shape[0], data_m.shape[0], data_n.shape[0])
    data_s = data_s[:min_len, :]
    data_m = data_m[:min_len, :]
    data_n = data_n[:min_len, :]
    data_o = np.mean(np.array([data_m, data_n]), axis=0)

    # Plot trajectory
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(data_s[:, 0], data_s[:, 1], label="Robot $s$", linewidth=3, alpha=0.5)
    ax.plot(data_m[:, 0], data_m[:, 1], label="Robot $m$", linewidth=3)
    ax.plot(data_n[:, 0], data_n[:, 1], label="Robot $n$", linewidth=3)
    ax.plot(data_o[:, 0], data_o[:, 1], label="Payload", linestyle="dotted", linewidth=3)
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.legend(frameon=True, loc="upper right")
    fig.tight_layout()
    mpl_render(REPO_ROOT / "figs" / "diagrams" / f"romi_impl_{prefix}_traj.pdf" if save_plot else None)

    # Plot tracking error
    data_s = np.unique(data_s, axis=0)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    err_vec = []
    for x_o, y_o, _ in data_o:
        min_dist = float("inf")
        for x_s, y_s, _ in data_s:
            dist = np.linalg.norm(np.array([x_o, y_o]) - np.array([x_s, y_s]))
            min_dist = min(min_dist, dist)
        err_vec.append(min_dist)
    for i in range(0, len(err_vec)):
        if err_vec[i] < err_vec[i+1]:
            break
    time_vec = np.arange(i, len(err_vec)) * 0.1
    err_vec = err_vec[i:]
    ax.plot(time_vec, err_vec, label="Tracking error")
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Distance Error (cm)")
    ax.set_ylim(-1, 10)
    fig.tight_layout()
    mpl_render(REPO_ROOT / "figs" / "diagrams" / f"romi_impl_{prefix}_error.pdf" if save_plot else None)

if __name__ == "__main__":
    for prefix in ["sync", "async", "star_sync", "cyclic_sync"]:
        plot(prefix)
