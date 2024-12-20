from typing import NamedTuple
import math
import numpy as np
from pathlib import Path
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
from loguru import logger

PORT = 23_000

class SimSession:
    def __init__(self, path: Path | str):
        self.scene_path = path = Path(path).resolve()
        if not (path := path.resolve()).exists():
            raise ValueError(f"Scene path at {path} is not found")
        logger.debug(f"Establishing remote client")
        self.client = RemoteAPIClient("localhost", PORT)
        logger.debug(f"Getting sim object handle")
        self.sim = self.client.require('sim')
        logger.debug(f"Loading scene at {str(path)}")
        self.sim.loadScene(str(path))

    def init(self):
        ...

    def step(self):
        ...

    def stop_condition(self):
        return self.sim.getSimulationState() == 0

    def save(self):
        self.sim.saveScene(str(self.scene_path))

    def run(self):
        self.client.setStepping(True)
        self.init()
        self.sim.startSimulation()
        while True:
            if self.stop_condition():
                break
            self.step()
            self.client.step()
        self.sim.stopSimulation()


class SimObject:
    def __init__(self, client, sim, obj_id: str | int):
        self._client = client
        self._sim = sim
        if isinstance(obj_id, int):
            self._obj_handle = obj_id
            self._obj_name = self._sim.getObjectAlias(self._obj_handle)
        else:
            self._obj_name = obj_id
            self._obj_handle = self._sim.getObject(f"/{obj_id}")
    
    def rename(self, new_name: str):
        self._sim.setObjectAlias(self._obj_handle, new_name)
        self._obj_name = new_name

    def get_handle(self, name: str = None):
        if name:
            return self._sim.getObject(f"/{self._obj_name}/{name}")
        else:
            return self._obj_handle

    def get_position(self, handle=None) -> np.ndarray:
        x, y, *_ = self._sim.getObjectPosition(
            handle or self._obj_handle, self._sim.handle_world
        )
        return np.array([x, y, *_] if handle else [x, y])

    def set_position(self, x: float, y: float):
        x_old, y_old, *_ = self.get_position(self._obj_handle)
        self._sim.setObjectPosition(
            self._obj_handle, self._sim.handle_world, [x, y, *_]
        )

    def get_orientation(self, handle=None) -> np.ndarray:
        *_, a = self._sim.getObjectOrientation(
            handle or self._obj_handle, self._sim.handle_world
        )
        return np.array([*_, a] if handle else a)

    def set_orientation(self, a: float):
        *_, a_old = self.get_orientation(self._obj_handle)
        self._sim.setObjectOrientation(
            self._obj_handle, self._sim.handle_world, [*_, a]
        )


class EncoderReading(NamedTuple):
    """A tuple to represent encoder reading"""
    left: int | float
    right: int | float

    def is_valid(self) -> bool:
        return not math.isnan(self.left) and not math.isnan(self.right)

class Pose(NamedTuple):
    """A tuple to represent robot pose"""
    x: float
    y: float
    o: float

    @classmethod
    def fromarray(cls, array: np.ndarray):
        return cls(*(array.flatten()))

    def toarray(self) -> np.ndarray:
        return np.array([self]).T

def wrap2pi(a: float) -> float:
    """Wrap a radiance angle between -pi to pi

    Args:
        a: angle in radiance

    Returns:
        wrapped angle
    """
    return (a + np.pi) % (2 * np.pi) - np.pi

def symmetrical_saturate(val: float, max_val: float) -> float:
    """Saturate a signed valued based on an abolute max value

    Args:
        val: input signed value
        max_val: absolute max value

    Returns:
        satuated value
    """
    return np.sign(val) * min(np.abs(val), np.abs(max_val))
