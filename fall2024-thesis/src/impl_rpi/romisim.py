#!venv/bin/python3

from typing import Any
import uuid
from loguru import logger
import time
import json
from threading import Lock
import msgpack
from paho.mqtt import client as mqtt_client
import numpy
import romispecs as specs
import sys

class Sim:
    def __init__(self, label = None, update_peorid: float = 0.1):
        self.update_period = update_peorid  # seconds
        self.wheel_circum = specs.wheel_diameter * numpy.pi
        self.ticks_per_rev = specs.encoder_resolution
        self.wheel_track = specs.wheel_track
        self.motor_lin_speed = (0, 0)  # cm/s
        self.motor_enc_count = (0, 0)  # encoder ticks
        if label:
            self.label = label
        else:
            id_ = str(uuid.uuid4()).split("-")[0]
            self.label = f"sim-{id_}"
        logger.info(f"Robot ID is {self.label}")
        self.mqtt_client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION2)
        self.mutex = Lock()
    
    def _mqtt_connect(self, host: str = "localhost", port: int = 1883):
        self.mqtt_client.connect(host, port)
        self.mqtt_client.message_callback_add(f"control/{self.label}", self._on_control)
        self.mqtt_client.subscribe(f"control/{self.label}")
        self.mqtt_client.loop_start()
    
    def _on_control(self, client: mqtt_client.Client, userdata: Any, msg: mqtt_client.MQTTMessage):
        # content: dict = json.loads(msg.payload)
        content: dict = msgpack.unpackb(msg.payload)
        if motors := content.get("motors"):
            self.mutex.acquire()
            self.motor_lin_speed = tuple(motors)
            self.mutex.release()
    
    def _update_status(self):
        new_enc = [0, 0]
        for i in range(2):
            rev = self.motor_lin_speed[i] * self.update_period / self.wheel_circum
            new_count = int(self.motor_enc_count[i] + rev * self.ticks_per_rev)
            if new_count < specs.encoder_min or new_count > specs.encoder_max:
                new_count = ((new_count - specs.encoder_min) % specs.encoder_range) + specs.encoder_min
            new_enc[i] = new_count
        self.motor_enc_count = tuple(new_enc)
    
    def _send_status(self):
        msg = {
            "encoders": self.motor_enc_count,
        }
        self.mqtt_client.publish(
            f"robots/{self.label}",
            # json.dumps(msg),
            msgpack.packb(msg),
        )
    
    def mainloop(self):
        self._mqtt_connect()
        while True:
            try:
                if self.mutex.locked():
                    time.sleep(0.1)
                    continue
                self._update_status()
                self._send_status()
                time.sleep(self.update_period)
            except KeyboardInterrupt:
                logger.info("Goodbye")
                break


if __name__ == "__main__":
    if sys.argv[1:]:
        sim = Sim(sys.argv[1])
    else:
        sim = Sim()
    sim.mainloop()
