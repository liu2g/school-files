#!venv/bin/python3

import numpy as np
from scipy.interpolate import splprep, splev
from romispecs import arm_length
from helpers import Pose
from typing import Tuple
from collections.abc import Generator

class Waypoints(Generator):
    nsamples = 20
    def __init__(self, pathxy: np.ndarray) -> None:
        super().__init__()
        pathxy_nodup = [pathxy[0]]
        for point in pathxy[1:]:
            if not np.array_equal(point, pathxy_nodup[-1]):
                pathxy_nodup.append(point)
        pathxy = np.array(pathxy_nodup)
        self.original_path = pathxy
        tck, u = splprep([pathxy[:, 0], pathxy[:, 1]], s=0)
        uvec = np.linspace(0, 1, self.nsamples)
        self._payload_x, self._payload_y = splev(uvec, tck)
        px, py = self._payload_x, self._payload_y
        dx, dy = splev(uvec, tck, der=1)
        self._payload_angle = np.arctan2(dy, dx)
        self._left_x = px - arm_length * np.sin(self._payload_angle)
        self._left_y = py + arm_length * np.cos(self._payload_angle)
        self._right_x = px + arm_length * np.sin(self._payload_angle)
        self._right_y = py - arm_length * np.cos(self._payload_angle)
        self._i = 0
    
    def send(self, *args, **kwargs) -> Tuple[Pose, Pose]:
        if self._i >= len(self._payload_angle) - 3:
            raise StopIteration
        rtn = (
            Pose(self._left_x[self._i], self._left_y[self._i], self._payload_angle[self._i]),
            Pose(self._right_x[self._i], self._right_y[self._i], self._payload_angle[self._i]),
        )
        self._i += 1
        return rtn

    def throw(self, *args, **kwargs):
        raise StopIteration

class SamplePath(Waypoints):
    def __init__(self) -> None:
        pathxy = np.loadtxt("samplepath.csv", delimiter=",", skiprows=1)
        super().__init__(pathxy)
        self._i = 1

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    gen = SamplePath()
    fig, ax = plt.subplots()
    poses_l = []
    poses_r = []

    def update_plot(frame):
        ax.clear()
        ax.grid(True)
        ax.set_xlabel("X (cm)")
        ax.set_ylabel("Y (cm)")
        colors = plt.rcParams["axes.prop_cycle"]()
        color = next(colors)["color"]
        ax.plot(gen.original_path[:, 0], gen.original_path[:, 1], color=color, label="Payload Path")
        try:
            pose_l, pose_r = next(gen)
        except StopIteration:
            return
        for label, pose, poses in zip(["Left", "Right"], [pose_l, pose_r], [poses_l, poses_r]):
            color = next(colors)["color"]
            poses.append(pose)
            ax.plot([p.x for p in poses], [p.y for p in poses], color=color, label=label)
            ax.scatter(pose.x, pose.y, color=color)
            ax.arrow(pose.x, pose.y,
                    10 * np.cos(pose.o), 10 * np.sin(pose.o),
                    color=color,
                    width=0.2, head_width=6, head_length=6,
                    )
        ax.legend(loc="upper right")

    ani = FuncAnimation(fig, update_plot, interval=500, cache_frame_data=False)
    plt.show()
