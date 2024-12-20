#!venv/bin/python3

import csv
import math
import time
from localizer import Localizer
import romispecs as specs
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Any
from paho.mqtt import client as mqtt_client
import msgpack
import numpy as np
from waypoints import Waypoints
import kinematics
from controller import RobotCtrl, PointNavCtrl
from enum import Enum, auto
from loguru import logger
from helpers import wrap2pi, Pose

class State(Enum):
    LEADER = auto()
    FOLLOWER = auto()

localizers: dict[str, Localizer] = {
    "romi_left": Localizer(0, 1 * specs.arm_length, 0),
    "romi_right": Localizer(0, -1 * specs.arm_length, 0),
    "romi_front": Localizer(specs.brick_size/2, 0, 0),
}
controllers: dict[str, RobotCtrl] = {
    "romi_left": None,
    "romi_right": None,
}

fig, ax = plt.subplots(figsize=(12, 6))
mutex = threading.Lock()
stop_thread = False

def update_localizers(client: mqtt_client.Client, userdata: Any, msg: mqtt_client.MQTTMessage):
    with mutex:
        robot_id = msg.topic.split("/")[1]
        content: dict[str, Any] = msgpack.unpackb(msg.payload)
        if encoders := content.get("encoders"):
            localizers[robot_id].run_encoders(*encoders)

def update_plot(frame):
    ax.clear()
    ax.grid(True)
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_xlim([-50, 300])
    ax.set_ylim([-150, 150])
    colors = plt.rcParams["axes.prop_cycle"]()
    color = next(colors)["color"]
    for k, l in localizers.items():
        color = next(colors)["color"]
        poses = l.poses.copy()
        ax.plot([p.x for p in poses], [p.y for p in poses], color=color,
                label=f"{k} (x, y, \u03F4) = ({l.current_pose.x:.2f}, {l.current_pose.y:.2f}, {np.rad2deg(l.current_pose.o):.2f})")
        ax.scatter(l.current_pose.x, l.current_pose.y, color=color)
        ax.arrow(l.current_pose.x, l.current_pose.y,
                 10 * np.cos(l.current_pose.o), 10 * np.sin(l.current_pose.o),
                 color=color,
                 width=0.2, head_width=6, head_length=6,
                 )
    color = next(colors)["color"]
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0))
    fig.tight_layout()

def update_command(stop):
    i = 0
    while True:
        if stop():
            with mutex:
                for k in localizers.keys():
                    client.publish(f"control/{k}", msgpack.packb({"motors": [0, 0]}))
            break
        with mutex:
            client.publish("control/romi_front", msgpack.packb({"stop": False}))
            target_pose = localizers["romi_front"].current_pose
            left_x = target_pose.x - specs.arm_length * np.sin(target_pose.o)
            left_y = target_pose.y + specs.arm_length * np.cos(target_pose.o)
            left_ctrl = PointNavCtrl(left_x, left_y, 100.0, np.pi,
                                     gain_linear=(1.0, 0.0, 0.0),
                                     gain_angular=(2.0, 0.0, 0.0))
            v, w = left_ctrl.go(localizers["romi_left"].current_pose)
            motorl, motorr = kinematics.vw_to_motor(v, w)
            client.publish("control/romi_left", msgpack.packb({"motors": [motorl, motorr]}))
            right_x = target_pose.x + specs.arm_length * np.sin(target_pose.o)
            right_y = target_pose.y - specs.arm_length * np.cos(target_pose.o)
            right_ctrl = PointNavCtrl(right_x, right_y, 100.0, np.pi,
                                      gain_linear=(1.0, 0.0, 0.0),
                                      gain_angular=(2.0, 0.0, 0.0))
            v, w = right_ctrl.go(localizers["romi_right"].current_pose)
            motorl, motorr = kinematics.vw_to_motor(v, w)
            client.publish("control/romi_right", msgpack.packb({"motors": [motorl, motorr]}))
        time.sleep(0.1)


client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION2)
client.connect("localhost", 1883)
client.message_callback_add("robots/#", update_localizers)
client.subscribe("robots/#")

client.loop_start()
ani = FuncAnimation(fig, update_plot, interval=500, cache_frame_data=False)
cmd_thread = threading.Thread(target=update_command, args=(lambda: stop_thread,))
cmd_thread.start()
plt.show()
stop_thread = True
cmd_thread.join()
for k in localizers.keys():
    client.publish(f"control/{k}", msgpack.packb({"stop": True, "motors": [0, 0]}))
    time.sleep(0.1)

# for k in localizers.keys():
#     with open(f"sync_{k}.csv", "w+") as f:
#         for p in localizers[k].poses:
#             f.write(f"{p.x},{p.y},{p.o}\n")