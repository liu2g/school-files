#!venv/bin/python3

from pathlib import Path
import inspect
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
from typing import Iterable
from attrs import define, field, astuple
from loguru import logger
import psutil

PORT = 23_000

REPO_ROOT = Path(__file__).parents[2]

class SimSession:
    def __init__(self, scene_path: Path | str):
        for proc in psutil.process_iter():
            if "coppeliasim" in proc.name().casefold():
                break
        else:
            raise ConnectionError("CoppeliaSim is not running")
        scene_path = Path(scene_path).resolve()
        if not (scene_path := scene_path.resolve()).exists():
            raise ValueError(f"Scene path at {scene_path} is not found")
        logger.debug(f"Establishing remote client")
        self.client = RemoteAPIClient("localhost", PORT)
        logger.debug(f"Getting sim object handle")
        self.sim = self.client.require('sim')
        logger.debug(f"Loading scene at {str(scene_path)}")
        self.sim.loadScene(str(scene_path))

    def init(self):
        ...

    def step(self):
        ...

    def stop_condition(self):
        return self.sim.getSimulationState() == 0

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


def get_module_dir(n_stack: int = 0) -> Path:
    """Get the directory containing current running module

    Args:
        n_stack: stack index if wish to go even upper layer. Defaults to 0.

    Returns:
        Directory containing currently running module that calls the function
    """
    frame = inspect.stack()[n_stack + 1]
    module = inspect.getmodule(frame[0])
    return Path(module.__file__).parent


def wrap2pi(a: float | np.ndarray) -> float | np.ndarray:
    """Wrap a radiance angle between -pi to pi

    Args:
        a: angle in radiance

    Returns:
        wrapped angle
    """
    return (a + np.pi) % (2 * np.pi) - np.pi


def signedsaturate(val: float, max_val: float) -> float:
    """Saturate a signed valued based on an abolute max value

    Args:
        val: input signed value
        max_val: absolute max value

    Returns:
        satuated value
    """
    return np.sign(val) * min(np.abs(val), np.abs(max_val))


@define
class Pose:
    x: float = field(default=0.0, converter=float)
    y: float = field(default=0.0, converter=float)
    a: float = field(default=0.0, converter=float)

    @classmethod
    def fromarray(cls, array: np.ndarray):
        return cls(*(array.flatten()))

    def toarray(self) -> np.ndarray:
        return np.array(astuple(self))


class SimObject:
    def __init__(self, client, sim, name: str):
        self._client = client
        self._sim = sim
        self._obj_name = name
        self._obj_handle = sim.getObject(f"/{name}")

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


class MobileRobot(SimObject):

    def get_pose(self) -> Pose:
        return Pose(*self.get_position(), self.get_orientation())

    def set_pose(self, pose: Pose):
        self.set_position(pose.x, pose.y)
        self.set_orientation(pose.a)

    def actuate(self, wheel_specs: dict[str | int, float]):
        for k, v in wheel_specs.items():
            if isinstance(k, str):
                motor_handle = self.get_handle(k)
            else:
                motor_handle = k
            self._sim.setJointTargetVelocity(motor_handle, v)

    def drive(self, *args, **kwargs):
        raise NotImplementedError


class PidCtrl:
    def __init__(
        self,
        ref_val: float | Iterable[float],
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.0,
        preprocess: callable = None,
        postprocess: callable = None,
    ):
        self.ref_val = ref_val
        self.kp, self.ki, self.kd = kp, ki, kd
        self.preprocess, self.postprocess = preprocess, postprocess
        self.reset()

    def go(self, val: float | Iterable[float], dt: float = None) -> float | None:
        cte = self.ref_val - val
        if self.preprocess:
            cte = self.preprocess(cte)
        self.cte_int += cte * dt
        pterm = cte
        iterm = self.cte_int
        dterm = 0 if self.cte_last is None else (cte - self.cte_last) / dt
        self.cte_last = cte
        out = self.kp * pterm + self.ki * iterm + self.kd * dterm
        if self.postprocess:
            return self.postprocess(out)
        else:
            return out

    def reset(self):
        self.cte_last = None
        self.cte_int = 0
