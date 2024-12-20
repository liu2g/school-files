#!venv/bin/python3

from pathlib import Path
import subprocess
import numpy as np
from matplotlib import pyplot as plt
from helpers import SimSession, MobileRobot, SimObject, wrap2pi, PidCtrl, signedsaturate
from helpers import REPO_ROOT
from mplpatch import *
from enum import Enum, auto

scene_path = Path(__file__).parent / "coppeliasim_files/youbot_scene.ttt"
plot_path =  REPO_ROOT / "figs/diagrams"
plot_path.resolve()
save_plot = False


class KukaYouBot(MobileRobot):
    wheel_rad = 0.05
    wheel_dist_x = 0.31702
    wheel_dist_y = 0.45601

    def get_orientation(self, handle=None) -> float | np.ndarray:
        a = self._sim.getObjectOrientation(
            handle or self._obj_handle, self._sim.handle_world
        )
        return np.array(np.array(a) if handle else a[1])

    def set_orientation(self, a: float):
        a_old = self.get_orientation(self._obj_handle)
        a_old[1] = a
        self._sim.setObjectOrientation(
            self._obj_handle, self._sim.handle_world, list(a_old)
        )

    def drive(self, linvel_x: float, linvel_y: float, angvel_z: float = 0.0):
        linvel_x, linvel_y = linvel_y, -1 * linvel_x
        angvel_z = -1 * angvel_z
        # https://ecam-eurobot.github.io/Tutorials/mechanical/mecanum.html
        angvel_fl = (
            linvel_x
            - linvel_y
            - (self.wheel_dist_x / 2 + self.wheel_dist_y / 2) * angvel_z
        ) / self.wheel_rad
        angvel_fr = (
            linvel_x
            + linvel_y
            + (self.wheel_dist_x / 2 + self.wheel_dist_y / 2) * angvel_z
        ) / self.wheel_rad
        angvel_rl = (
            linvel_x
            + linvel_y
            - (self.wheel_dist_x / 2 + self.wheel_dist_y / 2) * angvel_z
        ) / self.wheel_rad
        angvel_rr = (
            linvel_x
            - linvel_y
            + (self.wheel_dist_x / 2 + self.wheel_dist_y / 2) * angvel_z
        ) / self.wheel_rad
        self.actuate(
            {
                "rollingJoint_fl": angvel_fl,
                "rollingJoint_rl": angvel_rl,
                "rollingJoint_rr": angvel_rr,
                "rollingJoint_fr": angvel_fr,
            }
        )


class State(Enum):
    START = auto()
    LIFT = auto()
    NAVIGATE = auto()
    DROP = auto()
    LEAVE = auto()


class SimDemo(SimSession):

    def init(self):
        self.sim.cameraFitToView(self.sim.handleflag_camera, None, 0b00, 0.5)
        self.stop_flag = False
        self.state = State.START
        self.max_speed = 0.1
        self.pid_params = (0.8, 0.0, 0.8)
        self.simIK = self.client.require("simIK")
        self.payload = SimObject(self.client, self.sim, "Payload")
        self.target = SimObject(self.client, self.sim, "Target")
        self.youbot1 = KukaYouBot(self.client, self.sim, "youBot1")
        self.youbot2 = KukaYouBot(self.client, self.sim, "youBot2")
        self.ikenv1 = self.simIK.createEnvironment()
        self.ikgroup1 = self.simIK.createGroup(self.ikenv1)
        self.simIK.setGroupCalculation(
            self.ikenv1, self.ikgroup1, self.simIK.method_damped_least_squares, 0.3, 99
        )
        self.simIK.addElementFromScene(
            self.ikenv1,
            self.ikgroup1,
            self.youbot1.get_handle(),
            self.youbot1.get_handle("youBot_gripperOrientationTip"),
            self.youbot1.get_handle("tipTarget"),
            self.simIK.constraint_pose,
        )
        self.ikenv2 = self.simIK.createEnvironment()
        self.ikgroup2 = self.simIK.createGroup(self.ikenv2)
        self.simIK.setGroupCalculation(
            self.ikenv2, self.ikgroup2, self.simIK.method_damped_least_squares, 0.3, 99
        )
        self.simIK.addElementFromScene(
            self.ikenv2,
            self.ikgroup2,
            self.youbot2.get_handle(),
            self.youbot2.get_handle("youBot_gripperOrientationTip"),
            self.youbot2.get_handle("tipTarget"),
            self.simIK.constraint_pose,
        )
        self.distance = None
        self.pid_x1 = PidCtrl(
            self.target.get_position()[0],
            *self.pid_params,
            #   0.8, 0, 0,
            postprocess=lambda _: signedsaturate(_, self.max_speed),
        )
        self.pid_y1 = PidCtrl(
            self.target.get_position()[1],
            *self.pid_params,
            #   0.8, 0, 0,
            postprocess=lambda _: signedsaturate(_, self.max_speed),
        )
        self.pid_a1 = PidCtrl(
            0,
            *self.pid_params,
            #   0.8, 0, 0,
            postprocess=wrap2pi,
        )
        self.pid_x2 = PidCtrl(
            np.nan,
            *self.pid_params,
            #   0.8, 0, 0,
        )
        self.pid_y2 = PidCtrl(
            np.nan,
            *self.pid_params,
            #   0.8, 0, 0,
        )
        self.pid_a2 = PidCtrl(
            np.nan,
            *self.pid_params,
            #   0.8, 0, 0,
            postprocess=wrap2pi,
        )
        self.payload_pos_vec = []
        self.ee1_pos_vec = []
        self.ee2_pos_vec = []
        self.err_x1_vec = []
        self.err_y1_vec = []
        self.err_a1_vec = []
        self.err_x2_vec = []
        self.err_y2_vec = []
        self.err_a2_vec = []

    def step(self):
        payload_pos = self.payload.get_position(self.payload.get_handle())
        if self.state != State.LEAVE:
            self.payload_pos_vec.append(payload_pos)
        ee1_pos = self.youbot1.get_position(
            self.youbot1.get_handle("youBot_gripperPositionTip")
        )
        self.ee1_pos_vec.append(ee1_pos)
        ee2_pos = self.youbot2.get_position(
            self.youbot2.get_handle("youBot_gripperPositionTip")
        )
        self.ee2_pos_vec.append(ee2_pos)
        match self.state:
            case State.START:
                self.step_start()
            case State.LIFT:
                self.step_lift()
            case State.NAVIGATE:
                self.step_navigate()
            case State.DROP:
                self.step_drop()
            case State.LEAVE:
                self.step_leave()

    def stop_condition(self):
        return self.sim.getSimulationState() == 0 or self.stop_flag

    def step_start(self):
        youbot1_ready, youbot2_ready = False, False
        x1 = self.youbot1.get_pose().x
        x2 = self.youbot2.get_pose().x
        if np.abs(x1 - (-0.2886)) > 0.005:
            self.youbot1.drive(0.05, 0, 0)
        else:
            self.youbot1.drive(0, 0, 0)
            youbot1_ready = True
        if np.abs(x2 - 0.7376) > 0.005:
            self.youbot2.drive(-0.05, 0, 0)
        else:
            self.youbot2.drive(0, 0, 0)
            youbot2_ready = True
        if youbot1_ready and youbot2_ready:
            self.distance = abs(
                self.youbot1.get_position()[0] - self.youbot2.get_position()[0]
            )
            self.state = State.LIFT

    def step_lift(self):
        tip_handle = self.youbot1.get_handle("tipTarget")
        x, y, z = self.sim.getObjectPosition(tip_handle, self.sim.handle_world)
        self.sim.setObjectPosition(tip_handle, self.sim.handle_world, [x, y, z + 0.001])
        tip_handle = self.youbot2.get_handle("tipTarget")
        x, y, z = self.sim.getObjectPosition(tip_handle, self.sim.handle_world)
        self.sim.setObjectPosition(tip_handle, self.sim.handle_world, [x, y, z + 0.001])
        self.simIK.applyIkEnvironmentToScene(self.ikenv1, self.ikgroup1)
        self.simIK.applyIkEnvironmentToScene(self.ikenv2, self.ikgroup2)
        x, y, z = self.payload.get_position(self.payload.get_handle())
        if z > 0.35:
            self.sim.writeCustomDataBlock(
                self.youbot1.get_handle("youBot_gripper"), "activity", "close"
            )
            self.sim.writeCustomDataBlock(
                self.youbot2.get_handle("youBot_gripper"), "activity", "close"
            )
            self.state = State.NAVIGATE

    def step_navigate(self):
        dt = self.sim.getSimulationTimeStep()
        x1, y1, a1 = self.youbot1.get_pose().toarray()
        xp, yp = self.payload.get_position()
        linvel_x1 = self.pid_x1.go(xp, dt)
        linvel_y1 = self.pid_y1.go(yp, dt)
        angvel_z1 = self.pid_a1.go(a1, dt)
        self.err_x1_vec.append(self.pid_x1.cte_last)
        self.err_y1_vec.append(self.pid_y1.cte_last)
        self.err_a1_vec.append(self.pid_a1.cte_last)
        self.youbot1.drive(linvel_x1, linvel_y1, angvel_z1)
        self.pid_x2.ref_val = x1 + self.distance * np.cos(a1)
        self.pid_y2.ref_val = y1 + self.distance * np.sin(a1)
        self.pid_a2.ref_val = a1
        x2, y2, a2 = self.youbot2.get_pose().toarray()
        linvel_x2 = linvel_x1 + self.pid_x2.go(x2, dt)
        linvel_y2 = linvel_y1 + self.pid_y2.go(y2, dt)
        angvel_z2 = angvel_z1 + self.pid_a2.go(a2, dt)
        self.err_x2_vec.append(self.pid_x2.cte_last)
        self.err_y2_vec.append(self.pid_y2.cte_last)
        self.err_a2_vec.append(self.pid_a2.cte_last)
        self.youbot2.drive(linvel_x2, linvel_y2, angvel_z2)
        _ = self.target.get_position() - self.payload.get_position()
        if np.linalg.norm(_) < 0.05:
            self.state = State.DROP

    def step_drop(self):
        tip_handle = self.youbot1.get_handle("tipTarget")
        x, y, z = self.sim.getObjectPosition(tip_handle, self.sim.handle_world)
        self.sim.setObjectPosition(tip_handle, self.sim.handle_world, [x, y, z - 0.001])
        tip_handle = self.youbot2.get_handle("tipTarget")
        x, y, z = self.sim.getObjectPosition(tip_handle, self.sim.handle_world)
        self.sim.setObjectPosition(tip_handle, self.sim.handle_world, [x, y, z - 0.001])
        self.simIK.applyIkEnvironmentToScene(self.ikenv1, self.ikgroup1)
        self.simIK.applyIkEnvironmentToScene(self.ikenv2, self.ikgroup2)
        x, y, z = self.payload.get_position(self.payload.get_handle())
        if z < 0.27:
            self.sim.writeCustomDataBlock(
                self.youbot1.get_handle("youBot_gripper"), "activity", "open"
            )
            self.sim.writeCustomDataBlock(
                self.youbot2.get_handle("youBot_gripper"), "activity", "open"
            )
            self.state = State.LEAVE
            self.drop_counter = 0

    def step_leave(self):
        self.drop_counter += 1
        if 20 < self.drop_counter < 100:
            self.youbot1.drive(-0.1, 0, 0)
            self.youbot2.drive(0.1, 0, 0)
        elif self.drop_counter < 200:
            self.youbot1.drive(0, 0, 0)
            self.youbot2.drive(0, 0, 0)
        else:
            self.stop_flag = True

    def plot_err(self):
        def postprocess(fig, ax, ax2):
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Position error [$10^{-3}$ m]")
            ax2.set_ylabel("Orientation error [$10^{-3}$ rad]", color="darkblue")
            ax2.tick_params(axis="y", colors="darkblue")
            ylim1 = ax.get_ylim()
            len1 = ylim1[1] - ylim1[0]
            yticks1 = ax.get_yticks()
            rel_dist = [(y - ylim1[0]) / len1 for y in yticks1]
            ylim2 = ax2.get_ylim()
            len2 = ylim2[1] - ylim2[0]
            yticks2 = [ry * len2 + ylim2[0] for ry in rel_dist]
            ax2.set_yticks(yticks2)
            ax2.set_ylim(ylim2)
            # ax2.set_ylim([-np.pi, np.pi])
            # ax2.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
            #                [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$",
            #                r"$\pi$"])
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, frameon=True, loc="upper right")
            ax.grid(True)
            fig.tight_layout()

        t_vec = np.arange(len(self.err_x1_vec)) * self.sim.getSimulationTimeStep()
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(
            t_vec,
            np.array(self.err_x1_vec) * 1e3,
            label=r"$e_{a,x}$",
            color="darkorange",
        )
        ax.plot(
            t_vec,
            np.array(self.err_y1_vec) * 1e3,
            label=r"$e_{a,y}$",
            color="darkgreen",
        )
        ax2 = ax.twinx()
        ax2.plot(
            t_vec,
            np.array(self.err_a1_vec) * 1e3,
            label=r"$e_{a,\theta}$",
            color="darkblue",
        )
        postprocess(fig, ax, ax2)
        mpl_render(plot_path / "twinrobot-erra.pdf" if save_plot else None)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(
            t_vec,
            np.array(self.err_x2_vec) * 1e3,
            label=r"$e_{b,x}$",
            color="darkorange",
        )
        ax.plot(
            t_vec,
            np.array(self.err_y2_vec) * 1e3,
            label=r"$e_{b,y}$",
            color="darkgreen",
        )
        ax2 = ax.twinx()
        ax2.plot(
            t_vec,
            np.array(self.err_a2_vec) * 1e3,
            label=r"$e_{b,\theta}$",
            color="darkblue",
        )
        postprocess(fig, ax, ax2)
        mpl_render(plot_path / "twinrobot-errb.pdf" if save_plot else None)

    def plot_traj(self):
        payload_arr = np.array(self.payload_pos_vec)
        ee1_arr = np.array(self.ee1_pos_vec)
        ee2_arr = np.array(self.ee2_pos_vec)
        fig, ax = plt.subplots(
            subplot_kw=dict(projection="3d"),
            # layout='compressed'
        )
        ax.set_box_aspect((2, 2, 1))
        ax.view_init(elev=22, azim=-110, roll=0)
        ax.plot(
            payload_arr[:, 0], payload_arr[:, 1], payload_arr[:, 2], label="Payload"
        )
        ax.plot(
            payload_arr[:, 0],
            payload_arr[:, 1],
            np.zeros(len(payload_arr)) + 0.1,
            color="gray",
            alpha=0.8,
            label="Projection",
        )
        ax.plot(
            ee1_arr[:, 0],
            ee1_arr[:, 1],
            ee1_arr[:, 2],
            linestyle="dashed",
            color="darkorange",
            alpha=0.5,
            label="End-Effectors",
        )
        ax.plot(
            ee2_arr[:, 0],
            ee2_arr[:, 1],
            ee2_arr[:, 2],
            linestyle="dashed",
            color="darkorange",
            alpha=0.5,
        )
        ax.scatter(
            payload_arr[0, 0],
            payload_arr[0, 1],
            payload_arr[0, 2],
            s=40,
            label="Start",
            color="darkgreen",
            marker="^",
        )
        ax.scatter(
            *self.target.get_position(), 0.26, s=40, label="Goal", color="darkred"
        )
        ax.set_xlabel("$x$ [m]")
        ax.set_ylabel("$y$ [m]")
        ax.set_zlabel("$z$ [m]")
        ax.set_zlim([0.11, 0.4])
        ax.set_ylim([-2.5, 2.0])
        ax.tick_params(axis="y", which="major", pad=-3)
        ax.legend(loc="center left", bbox_to_anchor=(0.97, 0.5))
        p = plot_path / "twinrobot-trajectory.pdf"
        mpl_render(p if save_plot else None)
        if save_plot:
            subprocess.run(["pdfcrop", "--margins", "-1 -1 0 -1", str(p), str(p)])


if __name__ == "__main__":
    mpl_usetex(True)
    mpl_fontsize(10)
    sim = SimDemo(scene_path)
    sim.run()
    sim.plot_traj()
    mpl_style("seaborn")
    mpl_fontsize(13)
    sim.plot_err()
