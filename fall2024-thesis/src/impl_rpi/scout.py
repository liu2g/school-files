#!venv/bin/python3
import threading
from transporter import Robot
from loguru import logger
from typing import Any

from paho.mqtt import client as mqtt_client
import msgpack, json
import sys
# from minimu import MinIMU
import time

if len(sys.argv) < 2:
    raise ValueError("Please provide a robot ID")
robot_id = sys.argv[1]
robot = Robot()
robot.reset()
robot.stopped = True
for i in range(3):
    logger.info(f"Robot {robot_id} is reset, commencing in {3 - i}...")
    time.sleep(1)
# imu = MinIMU()
# imu.enableAccel_Gyro()
client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION2)
client.connect("192.168.8.2", 1883)

def publish_info():
    global robot_id, client, robot, imu
    while True:
        encl, encr = robot.encoders
        # accelx, accely, accelz = imu.readAccel()
        # gyrox, gyroy, gyroz = imu.readGyro()
        client.publish(f"robots/{robot_id}",
                        msgpack.packb(
                    # json.dumps(
                            {
                                "time": time.time(),
                                "encoders": (encl, encr),
                                # "accelx": accelx,
                                # "accely": accely,
                                # "accelz": accelz,
                                # "gyrox": gyrox,
                                # "gyroy": gyroy,
                                # "gyroz": gyroz
                            }
                        )
                    )
        time.sleep(0.1)

def control_callback(client: mqtt_client.Client, userdata: Any, msg: mqtt_client.MQTTMessage):
    content: dict = msgpack.unpackb(msg.payload)
    if "stop" in content:
        robot.stopped = bool(content["stop"])

client.message_callback_add(f"control/{robot_id}", control_callback)
client.subscribe(f"control/{robot_id}")
client.loop_start()
threading.Thread(target=publish_info, daemon=True).start()

no_reading_count = 0

while True:
    if not robot.stopped:
        linel, liner = robot.linesensors
        if linel < 100 and liner < 100:
            no_reading_count += 1
            if no_reading_count > 20:
                robot.motors = (0, 0)
        elif linel > 100:
            no_reading_count = 0
            robot.motors = (2, 7)
        elif liner > 100:
            no_reading_count = 0
            robot.motors = (7, 2)
        else:
            no_reading_count = 0
            robot.motors = (2, 2)
    else:
        robot.motors = (0, 0)
    time.sleep(0.05)