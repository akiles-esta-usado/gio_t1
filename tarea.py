from tarea1.algoritmos import *
from tarea1.graficos import *

import numpy as np


def f(X):
  x1 = X[0]
  x2 = X[1]

  return (4 - 2.1 * x1**2 + (x1**4)/3) * x1**2 + x1*x2 + (-4 + 4*x2**2) * x2**2

def fd(X):
  x1 = X[0]
  x2 = X[1]

  return np.array([
    2 * (x1**5 - 4.2 * x1**3 + 4 * x1 + 0.5 * x2),
    x1 + 16*x2**3 - 8*x2
  ])


def fdd(X):
  x1 = X[0]
  x2 = X[1]

  return np.array([
      [
        8 - 25.2 * x1**2 + 10 * x1**4,
        1
      ], [
        1,
        -8 + 48 * x2**2
      ]
  ])

for point in instanceGenerator():
  initial_state = State(
    point,
    alpha=0.3,
    beta=0.6,
    f=f,
    fd=fd,
    fdd=fdd
  )

  #results: Capture = metodo_gradiente(initial_state)
  results: Capture = metodo_newton(initial_state)

  print(results)
  print()
  break