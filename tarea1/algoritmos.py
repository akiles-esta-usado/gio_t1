import random
import numpy as np
from dataclasses import dataclass
from typing import Any, Callable


import numpy as np


def hola():
  print ("hola")


def hola2():
  print ("hola")


def pointGenerator():
  return np.array([
    random.uniform(-10, 10),
    random.uniform(-20, 20),
  ])

def instanceGenerator():
  return np.array([
    pointGenerator() for _ in range(10)
  ])




@dataclass
class State:
  punto: Any
  alpha: float
  beta: float
  f: Callable[..., Any]
  fd: Callable[..., Any]


@dataclass
class Capture:
  k: int
  FO_optima: list


def state_update(state: State):
  punto = state.punto
  alpha = state.alpha
  beta = state.beta
  f = state.f
  fd = state.fd

  t = 1.0

  while True:
    LR = f(punto) + alpha * t * np.dot(np.transpose(fd(punto)), - fd(punto))
    LL = f(punto - t*fd(punto))
    t = t * beta

    if LR>LL:
      state.punto = punto - t * fd(punto)
      break


def capture_update(state: State, capture: Capture) -> Capture:

  punto = state.punto
  f = state.f

  capture.k += 1
  capture.FO_optima.append(f(punto))

  if capture.k % 1000 == 0:
    print("hola")


def metodo_gradiente(state: State, epsilon=0.001):
  
  capture = Capture(k=0, FO_optima=list())

  f = state.f
  fd = state.fd

  while True:
    state_update(state)

    if np.abs(np.linalg.norm(fd(state.punto))) <= epsilon:
      break

    capture_update(state, capture)


  capture_update(state, capture)
  print(f(state.punto), capture.k)
  
  return capture