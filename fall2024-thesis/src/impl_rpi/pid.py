from typing import Iterable

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

    def go(self, val: float | Iterable[float], dt: float) -> float | None:
        cte = self.ref_val - val
        if self.preprocess:
            cte = self.preprocess(cte)
        self.cte_int += cte * dt
        pterm = cte
        iterm = self.cte_int
        dterm = 0 if self.cte_last is None else (cte - self.cte_last) / dt
        out = self.kp * pterm + self.ki * iterm + self.kd * dterm
        if self.postprocess:
            return self.postprocess(out)
        else:
            return out

    def reset(self):
        self.cte_last = None
        self.cte_int = 0

