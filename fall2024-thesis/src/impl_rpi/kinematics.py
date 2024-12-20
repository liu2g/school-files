import romispecs as specs
from helpers import Pose

def vw_to_motor(v: float, w: float) -> tuple[int, int]:
    """
    Calculate left and right wheel speeds from linear and angular speeds.
    Args:
        v: linear speed in cm/s
        w: angular speed in rad/s

    Returns:
        left and right wheel speeds in cm/s
    """
    return (
        (2 * v - w * specs.wheel_track) / (2 * specs.wheel_diameter),
        (2 * v + w * specs.wheel_track) / (2 * specs.wheel_diameter),
    )
