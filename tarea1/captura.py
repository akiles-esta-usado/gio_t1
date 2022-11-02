from dataclasses import dataclass, field
from typing import Any
from tarea1.algorithm_state import State
from time import perf_counter_ns

import numpy as np

@dataclass
class Capture:
  k: int = 0
  point: Any = field(init=False)
  FO_optima: list = field(default_factory=list)
  start_time: float = field(default_factory=perf_counter_ns)
  stop_time: float = 0.0
  total_time: float = 0.0

  def __post_init__(self):
    self.point = np.array([0.0, 0.0])

  def __repr__(self):
    return f"iter: {self.k} | f{self.point} = {self.FO_optima[-1]} | time={self.total_time/1e6}[ms]"



def capture_update(state: State, capture: Capture):

  punto = state.punto
  f = state.f

  capture.k += 1
  capture.FO_optima.append(f(punto))


def capture_finish(state: State, capture: Capture):

  capture.point[0] = state.punto[0]
  capture.point[1] = state.punto[1]

  capture.stop_time = perf_counter_ns()
  capture.total_time = capture.stop_time - capture.start_time

