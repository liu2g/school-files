from typing import Iterable, Optional
import numpy as np
from helpers import wrap2pi, Pose
from abc import ABC, abstractmethod

class Pid:
    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.0,
    ):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.reset()

    def go(self, cte: float, dt: Optional[float] = None) -> float:
        """
        Return the control output for the given cross-track error (cte) and time step (dt).

        Args:
            cte: Cross-track error (reference - actual)
            dt: Time step (required for integral and derivative terms)
        
        Returns:
            Control output
        """
        if self.ki or self.kd:
            if dt is None:
                raise ValueError("dt is required for integral and derivative terms")
            else:
                self.cte_int += cte * dt
                iterm = self.cte_int
                dterm = 0 if self.cte_last is None else (cte - self.cte_last) / dt
                self.cte_last = cte
        else:
            iterm = 0
            dterm = 0
        pterm = cte
        out = self.kp * pterm + self.ki * iterm + self.kd * dterm
        return out

    def reset(self):
        self.cte_last = None
        self.cte_int = 0


class RobotCtrl(ABC):
    @abstractmethod
    def go(self, pose: Pose, dt: Optional[float] = None) -> tuple[float, float]:
        """Return (linear velocity, angular velocity) as per given pose and control logic"""
        ...


class PointNavCtrl(RobotCtrl):
    def __init__(self, x_ref: float, y_ref: float,
                 max_linear: float, max_angular: float,
                 gain_linear: Iterable[float] = (4.0, 0.0, 0.0),
                 gain_angular: Iterable[float] = (4.0, 0.0, 0.0),
                 done_tol: float = 2.0):
        self.x_ref, self.y_ref = x_ref, y_ref
        self.max_linear, self.max_angular = max_linear, max_angular
        self._pid_linear = Pid(*gain_linear)
        self._pid_angular = Pid(*gain_angular)
        self.done_tol = done_tol

    def go(self, pose: Pose, dt: Optional[float] = None) -> tuple[float, float]:
        e_x = self.x_ref - pose.x
        e_y = self.y_ref - pose.y
        e_d = np.linalg.norm([e_x, e_y])
        if e_d < self.done_tol:
            return 0, 0
        linear = self._pid_linear.go(e_d, dt)
        linear = np.clip(linear, -self.max_linear, self.max_linear)
        angular = self._pid_angular.go(wrap2pi(np.arctan2(e_y, e_x) - pose.o), dt)
        angular = np.clip(angular, -self.max_angular, self.max_angular)
        return linear, angular

class OrientationCtrl(RobotCtrl):
    def __init__(self, o_ref: float, 
                 max_angular: float,
                 gain_angular: Iterable[float] = (4.0, 0.0, 0.0),
                 done_tol: float = 15*np.pi/180):
        self.o_ref = o_ref
        self.max_angular = max_angular
        self._pid_angular = Pid(*gain_angular)
        self.done_tol = done_tol
    
    def go(self, pose: Pose, dt: Optional[float] = None) -> tuple[float, float]:
        e_o = wrap2pi(self.o_ref - pose.o)
        if abs(e_o) < self.done_tol:
            return 0, 0
        angular = self._pid_angular.go(e_o, dt)
        angular = np.clip(angular, -self.max_angular, self.max_angular)
        return 0, angular


class LineCtrl(RobotCtrl):
    def __init__(self, y_ref: float, 
                 max_angular: float, 
                 linear_vel: float,
                 gain_steer: Iterable[float] = (2.0, 0.0, 0.0),
                 gain_distance: Iterable[float] = (1.0, 0.0, 0.0)
                 ):
        self.y_ref = y_ref
        self.o_ref = 0
        self._pid_steer = Pid(*gain_steer)
        self._pid_distance = Pid(*gain_distance)
        self.max_angular = max_angular
        self.linear_vel = linear_vel

    def go(self, pose):
        distance_to_line = self.y_ref - pose.y
        angular = (self._pid_steer.go(self.o_ref - pose.o) + self._pid_distance.go(distance_to_line))
        angular = np.clip(wrap2pi(angular), -self.max_angular, self.max_angular)
        return self.linear_vel, angular

class PurPursuitCtrl(RobotCtrl):
    def __init__(self, node: Pose,
                 angular_vel: float = 10*np.pi/180):
        self.node = node
        self.angular_vel = angular_vel
    
    def go(self, pose: Pose, dt: Optional[float] = None) -> tuple[float, float]:
        e_x = self.node.x - pose.x
        e_y = self.node.y - pose.y
        dist = np.linalg.norm([e_x, e_y]) + 1e-6
        if dist < 5.0 and abs(wrap2pi(np.arctan2(e_y, e_x) - pose.o)) < 5*np.pi/180:
            return 0, 0
        e_o = wrap2pi(np.arctan2(e_y, e_x) - pose.o)
        curv = 2 * np.sin(e_o) / dist
        linear = self.angular_vel / curv
        # linear = self.angular_vel * e_x / (np.sin(self.node.o) - np.sin(pose.o))
        linear = np.clip(linear, 0.0, 40.0)
        return linear, self.angular_vel

