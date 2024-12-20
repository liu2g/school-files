#!venv/bin/python3

import time
import numpy as np
from helpers import SimSession, SimObject, wrap2pi
from paho.mqtt import client as mqtt_client
from threading import Lock
from typing import Any
import msgpack
from pathlib import Path
from loguru import logger

scene_path = (Path(__file__).parent / "coppeliasim_files/romi_scene.ttt").resolve()
model_path = {
    "romi_front": (Path(__file__).parent / "coppeliasim_files/romi.ttm").resolve(),
    "romi_left": (Path(__file__).parent / "coppeliasim_files/romi_left.ttm").resolve(),
    "romi_right": (Path(__file__).parent / "coppeliasim_files/romi_right.ttm").resolve()
}
encoder_resolution = 1440  # ticks per revolution
encoder_min, encoder_max = -32768, 32767  # int16
wheel_diameter = 7.0  # cm
arm_length = 20.0  # cm
brick_size = 60.96  # cm

class RomiRobot(SimObject):
    def __init__(self, client, sim, name: str):
        super().__init__(client, sim, name)
        self._motor_handles = [
            self.get_handle("motor_left"),
            self.get_handle("motor_right")
        ]
        self._line_sensor_handles = [
            self.get_handle("linesensor_left"),
            self.get_handle("linesensor_right")
        ]
        self._motor_positions_last = [self._sim.getJointPosition(h) for h in self._motor_handles]
        self._motor_encoders_last = (0, 0)
        self.desired_motor_vel = (0, 0)
        self._sensor_readings = {
            "linesensor_left": -1,
            "linesensor_right": -1
        }
        try:
            self.gripper_left = self.get_handle("rr_joint_1/rr_arm_1/rr_joint_2/rr_arm_2/gripper/gripper_left")
            self.gripper_right = self.get_handle("rr_joint_1/rr_arm_1/rr_joint_2/rr_arm_2/gripper/gripper_left/gripper_joint/gripper_right")
        except Exception:
            self.gripper_left = self.gripper_right = None

    def set_motors(self, left: float, right: float):
        self.desired_motor_vel = (
            left / wheel_diameter / np.pi,
            right / wheel_diameter / np.pi
        )
    
    def get_encoders(self) -> tuple[float, float]:
        mpos_new_buffer = []
        menc_new_buffer = []
        for mhandle, mpos, menc in zip(
            self._motor_handles, self._motor_positions_last, self._motor_encoders_last
        ):
            mpos_new = self._sim.getJointPosition(mhandle)
            mpos_new_buffer.append(mpos_new)
            rev = wrap2pi(mpos_new - mpos) / (2 * np.pi)
            menc_new = round(menc + rev * encoder_resolution)
            if menc_new < encoder_min or menc_new > encoder_max:
                menc_new = ((menc_new - encoder_min) % (encoder_max - encoder_min)) + encoder_min
            menc_new_buffer.append(menc_new)
        self._motor_positions_last = mpos_new_buffer
        self._motor_encoders_last = menc_new_buffer
        return tuple(menc_new_buffer)

    def follow_line(self):
        linesensor_left = self._sensor_readings["linesensor_left"]
        linesensor_right = self._sensor_readings["linesensor_right"]
        if linesensor_left == 0 and linesensor_right == 0:
            self.set_motors(5, 5)
        elif linesensor_left == 0:
            self.set_motors(5, 10)
        elif linesensor_right == 0:
            self.set_motors(10, 5)
        else:
            self.set_motors(0, 0)
    
    def step(self):
        self._sim.setJointTargetVelocity(self._motor_handles[0], self.desired_motor_vel[0])
        self._sim.setJointTargetVelocity(self._motor_handles[1], self.desired_motor_vel[1])
        linesensor_left = self._sim.readVisionSensor(self._line_sensor_handles[0])
        if isinstance(linesensor_left, tuple):
            linesensor_left = linesensor_left[0]
        elif isinstance(linesensor_left, int):
            linesensor_left = linesensor_left
        self._sensor_readings["linesensor_left"] = linesensor_left
        linesensor_right = self._sim.readVisionSensor(self._line_sensor_handles[1])
        if isinstance(linesensor_right, tuple):
            linesensor_right = linesensor_right[0]
        elif isinstance(linesensor_right, int):
            linesensor_right = linesensor_right
        self._sensor_readings["linesensor_right"] = linesensor_right

class SimDemo(SimSession):

    def init(self):
        self.mutex = Lock()
        self.robots = {
            "romi_front": RomiRobot(self.client, self.sim, "romi_front"),
            "romi_left": RomiRobot(self.client, self.sim, "romi_left"),
            "romi_right": RomiRobot(self.client, self.sim, "romi_right")
        }
        self.mqtt_client = mqtt_client.Client()
        self._last_status_time = time.time()
        self.payload = self.sim.getObject("/payload")
    
    def populate_robots(self):
        pos_lut = {
            "romi_front": (brick_size / 2 / 100, 0),
            "romi_left": (0, arm_length / 100),
            "romi_right": (0, -arm_length / 100)
        }
        for name, pos in pos_lut.items():
            try:
                handle = self.sim.getObject(f"/{name}")
                self.sim.removeModel(handle)
            except Exception:
                ...
            handle = self.sim.loadModel(model_path[name].as_posix())
            robot = RomiRobot(self.client, self.sim, handle)
            robot.rename(name)
            xyz = self.sim.getObjectPosition(handle, self.sim.handle_world)
            self.sim.setObjectPosition(handle, self.sim.handle_world, [*pos, xyz[2]])

    def _mqtt_connect(self, host: str = "localhost", port: int = 1883):
        self.mqtt_client.connect(host, port)
        self.mqtt_client.message_callback_add(f"control/#", self._on_control)
        self.mqtt_client.subscribe(f"control/#")
        self.mqtt_client.loop_start()
        logger.info(f"Connected to MQTT broker at {host}:{port}")
    
    def _on_control(self, client: mqtt_client.Client, userdata: Any, msg: mqtt_client.MQTTMessage):
        with self.mutex:
            robot_id = msg.topic.split("/")[1]
            content: dict = msgpack.unpackb(msg.payload)
            robot = self.robots[robot_id]
            if motors := content.get("motors"):
                robot.set_motors(*motors)
            elif content.get("stop"):
                robot.set_motors(0, 0)
            else:
                robot.follow_line()
    
    def _send_status(self):
        for robot_id, robot in self.robots.items():
            encoders = robot.get_encoders()
            self.mqtt_client.publish(f"robots/{robot_id}", msgpack.packb({"encoders": encoders}))
    
    def step(self):
        if not self.mqtt_client.host:
            try:
                self._mqtt_connect()
            except Exception:
                return
        with self.mutex:
            if time.time() - self._last_status_time > 0.1:
                self._send_status()
                self._last_status_time = time.time()
            for robot in self.robots.values():
                robot.step()
            ee_left_xyz = self.robots["romi_left"].get_position(self.robots["romi_left"].gripper_left)
            ee_right_xyz = self.robots["romi_right"].get_position(self.robots["romi_right"].gripper_right)
            self.sim.setObjectPosition(
                self.payload, 
                [(ee_left_xyz[i] + ee_right_xyz[i]) / 2 for i in range(3)],
                self.sim.handle_world,
            )
            yaw = np.arctan2(ee_left_xyz[1] - ee_right_xyz[1], ee_left_xyz[0] - ee_right_xyz[0])
            self.sim.setObjectOrientation(
                self.payload,
                self.sim.handle_world,
                [0, 0, yaw]
            )
    
if __name__ == "__main__":
    sim = SimDemo(scene_path)
    # sim.populate_robots()
    # sim.save()
    sim.run()
