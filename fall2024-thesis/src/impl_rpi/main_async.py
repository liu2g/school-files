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
        if k != "romi_front":
            if c := controllers[k]:
                ax.scatter(c.x_ref, c.y_ref, marker="x", c=color)
        ax.arrow(l.current_pose.x, l.current_pose.y,
                 10 * np.cos(l.current_pose.o), 10 * np.sin(l.current_pose.o),
                 color=color,
                 width=0.2, head_width=6, head_length=6,
                 )
    color = next(colors)["color"]
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0))
    fig.tight_layout()

def update_command(stop):
    waypoints = None
    i = 0
    total_dist = 0
    while True:
        if stop():
            with mutex:
                for k in localizers.keys():
                    client.publish(f"control/{k}", msgpack.packb({"motors": [0, 0]}))
            break
        with mutex:
            if waypoints is None:
                if total_dist >= 300.0:
                    break
                client.publish(f"control/romi_front", msgpack.packb({"stop": False}))
                if localizers["romi_front"].distance > 100.0:
                    total_dist += 100.0
                    poses = localizers["romi_front"].poses[i:].copy()
                    i = len(poses)
                    waypoints = Waypoints(np.array([[p.x, p.y] for p in poses]))
                    localizers["romi_front"].distance = 0.0
            else:
                client.publish(f"control/romi_front", msgpack.packb({"stop": True}))
                p = localizers["romi_front"].current_pose
                if controllers["romi_left"] is None and controllers["romi_right"] is None:
                    wpl, wpr = next(waypoints, (None, None))
                    if wpl is None or wpr is None:
                        waypoints = None
                        i = len(localizers["romi_front"].poses)
                        continue
                    controllers["romi_left"] = PointNavCtrl(wpl.x, wpl.y, 40.0, np.pi)
                    controllers["romi_right"] = PointNavCtrl(wpr.x, wpr.y, 40.0, np.pi)
                for k in ["romi_left", "romi_right"]:
                    if controllers[k] is not None:
                        l = localizers[k]
                        c = controllers[k]
                        v, w = c.go(l.current_pose)
                        motorl, motorr = kinematics.vw_to_motor(v, w)
                        client.publish(
                            f"control/{k}", 
                            msgpack.packb({"motors": [motorl, motorr]})
                        )
                        if v == 0 and w == 0:
                            controllers[k] = None
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
#     with open(f"async_{k}.csv", "w+") as f:
#         for p in localizers[k].poses:
#             f.write(f"{p.x},{p.y},{p.o}\n")