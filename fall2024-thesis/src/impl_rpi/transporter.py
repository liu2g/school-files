#!venv/bin/python3

from smbus2 import SMBus
import struct
import time
from typing import Tuple, Any
import threading
from loguru import logger

class Robot:
    """
    Encapsulates the I2C communication with Pololu Romi 32U4 Robot

    Attributes:
        SERVODOWN: The value to send for the arm to be at the lowest position
        SERVOUP: The value to send for the arm to be at the highest position
    """
    SERVODOWN = 0
    SERVOUP = 100

    def __init__(self):
        self.bus = SMBus(1)
        self._lock = threading.Lock()
    
    def reset(self):
        """Resets the robot to a default state"""
        self.servo_down()
        self.motors = (0.0, 0.0)

    def _read_unpack(self, address: int, size: int, format: str) -> Tuple[Any, ...]:
        """Read and unpack data from the I2C bus

        Args:
            address: start N-th byte to read
            size: number of bytes to read
            format: format string for struct.unpack, see https://docs.python.org/3/library/struct.html#format-characters

        Returns:
            Tuple of unpacked values
        """
        with self._lock:
            self.bus.write_byte(20, address)
            time.sleep(0.0001)
            byte_list = [self.bus.read_byte(20) for _ in range(size)]
            return struct.unpack(format, bytes(byte_list))

        # Ideally we could do this:
        # > byte_list = self.bus.read_i2c_block_data(20, address, size)
        # But the AVR's TWI module can't handle a quick write->read transition,
        # since the STOP interrupt will occasionally happen after the START
        # condition, and the TWI module is disabled until the interrupt can
        # be processed.

    def _write_pack(self, address: int, format: str, *data: Any):
        """Pack and write data to the I2C bus

        Args:
            address: start N-th byte to write
            format: format string for struct.pack, see https://docs.python.org/3/library/struct.html#format-characters
            data: data to pack and write
        """
        with self._lock:
            data_array = list(struct.pack(format, *data))
            self.bus.write_i2c_block_data(20, address, data_array)
            time.sleep(0.0001)

    @property
    def leds(self):
        """
        Set the red, yellow, and green LEDs on the robot with three boolean values. Cannot be read.
        """
        raise NotImplementedError
    
    @leds.setter
    def leds(self, values: Tuple[bool, bool, bool]):
        red, yellow, green = values
        self._write_pack(0, 'BBB', red, yellow, green)

    @property
    def buttons(self) -> Tuple[bool, bool, bool]:
        """
        Read three push buttons on the robot. Cannot be written.

        Returns:
            Tuple of three boolean values representing the state of the buttons.
        """
        return self._read_unpack(3, 3, "???")
    
    @property
    def motors(self):
        """
        Set the left and right motor speeds by cm/s. Cannot be read.
        """
        raise NotImplementedError
    
    @motors.setter
    def motors(self, values: Tuple[float, float]):
        left, right = values
        self._write_pack(6, 'ff', left, right)

    @property
    def battery_millivolts(self) -> int:
        """
        Read the battery voltage in millivolts. Cannot be written.

        Returns:
            The battery voltage in millivolts.
        """
        return self._read_unpack(14, 2, "H")
    
    @property
    def analog(self) -> Tuple[int, int, int, int, int, int]:
        """
        Read the six analog readings on the robot. Cannot be written.

        Returns:
            Tuple of six integers representing the analog readings from 0 (0V) to 1023 (5V).
        """
        return self._read_unpack(16, 12, "HHHHHH")

    @property
    def encoders(self) -> Tuple[int, int]:
        """
        Read the left and right encoder counts. Cannot be written.

        Returns:
            Tuple of two integers representing the encoder counts from -32768 (backward) to 32767 (forward).
            The values can overflow and underflow.
        """
        return self._read_unpack(28, 4, 'hh')

    @property
    def linesensors(self) -> Tuple[int, int]:
        """
        Read the two line sensors on the robot. Cannot be written.

        Returns:
            Tuple of two integers representing the line sensor readings, always positive.
        """
        return self._read_unpack(32, 4, "HH")
    
    @property
    def servo(self):
        """
        Set the arm servo position by percent. Cannot be read.
        """
        raise NotImplementedError

    @servo.setter
    def servo(self, percent: int):
        self._write_pack(36, 'H', percent)
    
    def servo_up(self):
        """Set the arm servo position to the highest position."""
        self.servo = Robot.SERVOUP
    
    def servo_down(self):
        """Set the arm servo position to the lowest position."""
        self.servo = Robot.SERVODOWN

    def delaynormal(self):
        """Delay for 2 seconds."""
        time.sleep(2)

    def delayshort(self):
        """Delay for 0.5 seconds."""
        time.sleep(0.5)

    def delaylong(self):
        """Delay for 4 seconds."""
        time.sleep(4)

    # def test_read8(self):
    #     self._read_unpack(0, 8, 'cccccccc')

    # def test_write8(self):
    #     self.bus.write_i2c_block_data(20, 0, [0,0,0,0,0,0,0,0])
    #     time.sleep(0.0001)

if __name__ == "__main__":
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
        if motors := content.get("motors"):
            robot.motors = tuple(motors)

    client.message_callback_add(f"control/{robot_id}", control_callback)
    client.subscribe(f"control/{robot_id}")
    client.loop_start()
    # threading.Thread(target=publish_info, daemon=True).start()
    while True:
        try:
            publish_info()
        except KeyboardInterrupt:
            robot.reset()
            logger.info("Goodbye")
            break

