#!venv/bin/python3

import time
from typing import Generator, Dict, Any
import numpy as np
from paho.mqtt import client as mqtt_client
import msgpack
from loguru import logger


broker = "localhost"
port = 1883
topic = "robots/leader/main"
# username = 'emqx'
# password = 'public'

def mock() -> Generator[Dict[str, Any], None, None]:
    i = 0
    period = 5
    while True:
        yield {
            "motorl": float(np.sin(2 * np.pi * i / period)),
            "motorr": float(np.cos(2 * np.pi * i / period)),
        }
        i += 1

def main():
    client = mqtt_client.Client()
    # client.username_pw_set(username, password)
    client.connect(broker, port)
    client.loop_start()
    for obj in mock():
        logger.debug(f"[PUB] {topic}: {obj}")
        content = msgpack.packb(obj)
        client.publish(topic, content)
        time.sleep(1)
    client.loop_stop()

if __name__ == "__main__":
    main()
