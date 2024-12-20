"""Implements odometric localization for DDR with Runge-Kutta integration"""

import math
import numpy as np
import romispecs as specs
from typing import NamedTuple
from scipy.spatial.transform import Rotation
from helpers import EncoderReading, Pose, wrap2pi

class Localizer:
    """
    Localization algorithm.
    """

    def __init__(self, x_init: float, y_init: float, orient_init: float):
        self.current_pose: Pose = Pose(x_init, y_init, orient_init)
        self.poses: list[Pose] = [self.current_pose]
        self.distance = 0
        self._accel_bias = np.zeros(3)
        self._gyro_bias = np.zeros(3)
        self._last_enc = EncoderReading(math.nan, math.nan)
        self._last_time = math.nan
        self._last_gyro = np.zeros(3)
        self._last_rot = Rotation.from_euler("z", orient_init)
        self._velx = 0
        self._vely = 0
        self._rot_mat = np.eye(3)
    
    def calibrate_imu(self, ax: float, ay: float, az: float, gx: float, gy: float, gz: float):
        """Calibrate IMU by providing initial readings"""
        self._accel_bias = (np.array([ax, ay, az]) + self._accel_bias) / 2
        self._gyro_bias = (np.array([gx, gy, gz]) + self._gyro_bias) / 2

    def run_encoders(
            self, enc_lwheel: int, enc_rwheel: int
    ) -> Pose:
        """
        Main stepping method by providing encoder ticks, using Runge-Kutta integration
        Args:
            enc_lwheel: encoder tick value on the left wheel
            enc_rwheel: encoder tick value on the right wheel

        Returns:
        """
        if not self._last_enc.is_valid():
            self._last_enc = EncoderReading(enc_lwheel, enc_rwheel)
            return Pose(0, 0, 0)
        # calculate revolutions by comparing current and last encoder readings
        rev = []
        for enc, last_enc in zip((enc_lwheel, enc_rwheel), self._last_enc):
            delta_enc = enc - last_enc
            if abs(delta_enc) > specs.encoder_range / 2:  # possible overflow
                delta_enc = delta_enc - math.copysign(specs.encoder_range, delta_enc)
            rev.append(delta_enc / specs.encoder_resolution)
        self._last_enc = EncoderReading(enc_lwheel, enc_rwheel)
        # calculate odometry with Runge-Kutta integration
        dist_lwheel = rev[0] * math.pi * specs.wheel_diameter
        dist_rwheel = rev[1] * math.pi * specs.wheel_diameter
        d_dist = (dist_lwheel + dist_rwheel) / 2
        d_orient = (dist_rwheel - dist_lwheel) / specs.wheel_track
        dx = d_dist * math.cos(self.current_pose.o + d_orient / 2)
        dy = d_dist * math.sin(self.current_pose.o + d_orient / 2)
        self._move_forward(dx, dy, d_orient)
        return Pose(dx, dy, d_orient)

    def run_imu(self, t: float, ax: float, ay: float, az: float, gx: float, gy: float, gz: float) -> Pose:
        """
        Main stepping method by providing IMU readings, using dead reckoning
        Args:
            t: current time in seconds
            ax: acceleration in x-axis, in m/s^2
            ay: acceleration in y-axis, in m/s^2
            az: acceleration in z-axis, in m/s^2
            gx: angular velocity in x-axis, in rad/s
            gy: angular velocity in y-axis, in rad/s
            gz: angular velocity in z-axis, in rad/s

        Returns:
        """
        if math.isnan(self._last_time):
            self._last_time = t
            return Pose(0, 0, 0)
        dt = t - self._last_time
        self._last_time = t
        delta_angle = dt * (np.array([gx, gy, gz]) - self._gyro_bias)
        rot = Rotation.from_rotvec(delta_angle)
        self._last_rot = rot * self._last_rot
        accel_vec = self._last_rot.apply(np.array([ax, ay, az]) - self._accel_bias)
        self._velx += accel_vec[0] * dt
        self._vely += accel_vec[1] * dt
        dx, dy, do = dt * self._velx, dt * self._vely, delta_angle[2]
        self._move_forward(dx, dy, do)
        return Pose(dx, dy, do)

    def _move_forward(self, dx: float, dy: float, do: float):
        """Move the robot forward by dx, dy and do"""
        self.current_pose = Pose(
            self.current_pose.x + dx,
            self.current_pose.y + dy,
            wrap2pi(self.current_pose.o + do)
        )
        self.poses.append(self.current_pose)
        self.distance += math.hypot(dx, dy)

if __name__ == "__main__":
    from typing import Any
    import csv
    from matplotlib.animation import FuncAnimation
    from matplotlib import pyplot as plt
    from paho.mqtt import client as mqtt_client
    import msgpack, json

    loc_enc = Localizer(0, 0, math.pi / 2)
    loc_imu = Localizer(0, 0, math.pi / 2)

    client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION2)
    def on_message(client: mqtt_client.Client, userdata: Any, msg: mqtt_client.MQTTMessage):
        global loc_enc, loc_imu
        sample: dict[str, Any] = msgpack.unpackb(msg.payload)
        delta_pose = loc_enc.run_encoders(
            sample["encl"],
            sample["encr"]
        )
        if delta_pose.x == 0 and delta_pose.y == 0 and delta_pose.o == 0:
            loc_imu.calibrate_imu(
                sample["accelx"],
                sample["accely"],
                sample["accelz"],
                sample["gyrox"],
                sample["gyroy"],
                sample["gyroz"],
            )
        else:
            loc_imu.run_imu(
                sample["time"],
                sample["accelx"],
                sample["accely"],
                sample["accelz"],
                sample["gyrox"],
                sample["gyroy"],
                sample["gyroz"],
            )
    client.connect("localhost", 1883)
    client.subscribe("robots/#")
    client.on_message = on_message
    client.loop_start()

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))

    # samples = []
    # with open("sampledata.csv", "r") as f:
    #     reader = csv.DictReader(f)
    #     for row in reader:
    #         samples.append(row)


    def update_plot(frame):
        # if not samples:
        #     return
        # sample = samples.pop(0)
        ax[0].clear()
        ax[1].clear()
        # loc_enc.run_encoders(
        #     int(sample["encl"]),
        #     int(sample["encr"])
        # )
        # loc_imu.run_imu(
        #     float(sample["time"]),
        #     float(sample["accelx"]),
        #     float(sample["accely"]),
        #     float(sample["accelz"]),
        #     math.radians(float(sample["gyrox"])),
        #     math.radians(float(sample["gyroy"])),
        #     math.radians(float(sample["gyroz"])),
        # )
        poses = loc_enc.poses.copy()
        ax[0].plot([p.x for p in poses], [p.y for p in poses])
        ax[0].set_title("Encoder Odometry")
        poses = loc_imu.poses.copy()
        ax[1].plot([p.x for p in poses], [p.y for p in poses])
        ax[1].set_title("IMU Dead Reckoning")

    ani = FuncAnimation(fig, update_plot, interval=100, cache_frame_data=False)
    plt.show()