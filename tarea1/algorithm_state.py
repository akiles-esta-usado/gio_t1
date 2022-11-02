from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class State:
  punto: Any
  alpha: float
  beta: float
  f: Callable[..., Any]
  fd: Callable[..., Any]
  fdd: Callable[..., Any]