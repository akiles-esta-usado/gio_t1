import random
import numpy as np
from typing import Any

import numpy as np

from tarea1.algorithm_state import State
from tarea1.captura import *


def pointGenerator():
  return np.array([
    random.uniform(-10, 10),
    random.uniform(-20, 20),
  ])


def instanceGenerator():
  return np.array([
    pointGenerator() for _ in range(10)
  ])


def backtracking_line_search(state: State, step: Any, max_iterations: int = 10_000_000):
  punto = state.punto
  alpha = state.alpha
  beta = state.beta
  f = state.f
  fd = state.fd

  t = 1.0

  while max_iterations:
    max_iterations = max_iterations -1

    L_Izq = f(punto + t*step)
    L_Der = f(punto) + alpha * t * np.dot(fd(punto), step)

    if L_Der > L_Izq:
      return t

    t = t * beta
  
  raise Exception("Backtracking does not finish")


def metodo_gradiente(state: State, epsilon=0.0001):
  
  capture = Capture()

  f = state.f
  fd = state.fd


  while np.abs(np.linalg.norm(fd(state.punto))) > epsilon:

    capture_update(state, capture)

    step = - fd(state.punto)
    t = backtracking_line_search(state, step)
    state.punto = state.punto + t * step

  capture_finish(state, capture)
  return capture


def metodo_newton(state: State, epsilon=0.0001):

  capture = Capture()

  f = state.f
  fd = state.fd
  fdd = state.fdd

  while True:

    l2 = np.linalg.multi_dot([
      np.transpose(fd(state.punto)),
      np.linalg.inv(fdd(state.punto)),
      fd(state.punto)
    ])

    l2 = np.linalg.multi_dot([
      fd(state.punto),
      np.linalg.inv(fdd(state.punto)),
      np.transpose(fd(state.punto))
    ])

    capture_update(state, capture)

    if l2/2 <= epsilon or capture.k > 10000:
      break

    step = -np.dot(
      np.linalg.inv(fdd(state.punto)),
      np.transpose(fd(state.punto))
    )
    t = backtracking_line_search(state, step)
    state.punto = state.punto + t * step

  capture_finish(state, capture)
  return capture