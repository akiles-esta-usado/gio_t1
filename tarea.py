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


initial_state = State(
  np.array([1,1]),
  alpha=0.3,
  beta=0.6,
  f=f,
  fd=fd
)

results: Capture = metodo_gradiente(initial_state)

print(results)