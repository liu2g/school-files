"""This module contains the physical dimensions and specs of the robot"""

wheel_diameter = 7.0  # cm
encoder_resolution = 1440  # ticks per revolution
encoder_min, encoder_max = -32768, 32767  # int16
encoder_range = encoder_max - encoder_min
wheel_track = 14.7  # cm
arm_length = 20.0  # cm
brick_size = 60.96  # cm
