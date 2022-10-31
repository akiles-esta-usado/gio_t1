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
  fdd: Callable[..., Any]



@dataclass
class Capture:
  k: int
  FO_optima: list


def capture_update(state: State, capture: Capture) -> Capture:

  punto = state.punto
  f = state.f

  capture.k += 1
  capture.FO_optima.append(f(punto))

  if capture.k % 1000 == 0:
    print("hola")




def backtracking_line_search(state: State):
  punto = state.punto
  alpha = state.alpha
  beta = state.beta
  f = state.f
  fd = state.fd

  t = 1.0

  while True:
    L_Izq = f(punto - t*fd(punto))
    L_Der = f(punto) + alpha * t * np.dot(fd(punto), - fd(punto))

    if L_Der > L_Izq:
      return t

    t = t * beta


def newton(state: State):
  pass



def metodo_gradiente(state: State, epsilon=0.001):
  
  capture = Capture(k=0, FO_optima=list())

  f = state.f
  fd = state.fd

  while np.abs(np.linalg.norm(fd(state.punto))) > epsilon:

    capture_update(state, capture)

    t = backtracking_line_search(state)
    state.punto = state.punto - t * fd(state.punto)

  return capture







def metodo_newton(state: State, epsilon=0.001):
  
  capture = Capture(k=0, FO_optima=list())

  f = state.f
  fd = state.fd
  fdd = state.fdd

  while True:


    delta_x_nt = - np.dot(
      np.linalg.inv(fdd(state.punto)),
      np.transpose(fd(state.punto))
    )

    l2 = np.linalg.multi_dot([
      np.transpose(fd(state.punto)),
      np.linalg.inv(fdd(state.punto)),
      fd(state.punto)
    ])

    if l2/2 <= epsilon or capture.k > 10000:
      break

    capture_update(state, capture)

    # ESTO NO SE SI ESTÁ BIEN :(
    t = backtracking_line_search(state)
    state.punto = state.punto + t * delta_x_nt
  
  return capture



# Gradiente ocupa solo el gradiente, 
# newton toma el valor de gradiente y la tasa de cambio.

# Nosotros usamos un newton adaptado.