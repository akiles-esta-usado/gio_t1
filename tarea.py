from tarea1.algoritmos import *
from tarea1.graficos import *

from tarea1.algorithm_state import State

import numpy as np


def f(X):
  x1 = X[0]
  x2 = X[1]

  return (4 - 2.1 * x1**2 + (x1**4)/3) * x1**2 + x1*x2 + (-4 + 4*x2**2) * x2**2

def fd(X):
  x1 = X[0]
  x2 = X[1]

  return np.array([
    2*x1**5 - 8.4*x1**3 + 8*x1 + x2,
    x1 + 16*x2**3 - 8*x2
  ])


def fdd(X):
  x1 = X[0]
  x2 = X[1]

  return np.array([
      [
        8 - 25.2 * x1**2 + 10*x1**4,
        1
      ], [
        1,
        -8 + 48*x2**2
      ]
  ])


random_points = instanceGenerator()




# cols = list()

# for i in np.linspace(1, 5, 5):
#   cols.append([ [i, j] for j in np.linspace(1, 5, 5) ])


# cols = np.array(cols)

# print(cols)


# exit()

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


for i, point in enumerate(random_points):
  alpha = 0.3
  beta = 0.7

  state_gradiente = State(
    point,
    alpha,
    beta,
    f,
    fd,
    fdd
  )

  state_newton = State(
    point,
    alpha,
    beta,
    f,
    fd,
    fdd
  )

  epsilon = 1e-6
  
  results_gradiente: Capture = metodo_gradiente(state_gradiente, epsilon)
  results_newton: Capture = metodo_newton(state_newton, epsilon)

  print(f"punto de inicio: {point}")
  print(f"gradiente | {results_gradiente}")
  print(f"newton    | {results_newton}")

  print()

  

  fig, (ax1) = plt.subplots(nrows=1, figsize=(8,5))
  ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
  #ax1.set_yscale('log')
  ax1.axhline(y = -1.0316, color = 'r', linestyle = '--')
  ax1.set_ylim(-5, 10)
  ax1.plot(
    range(results_gradiente.k),
    results_gradiente.FO_optima,
    # s=0.7,
    label=rf'método gradiente, time={results_gradiente.total_time/1e6}[ms]'
  )
  ax1.plot(
    range(results_newton.k),
    results_newton.FO_optima,
    # s=0.7,
    label=rf'método newton, time={results_newton.total_time/1e6}[ms]'
  )
  ax1.set_xlabel("Número de iteraciones")
  #ax1.set_ylabel("Valor de FO en iteración $k$")
  ax1.set_ylabel("Valor de log FO en iteración $k$")
  ax1.set_title(rf'$\alpha = {alpha}$, $\beta = {beta}$')
  ax1.legend(loc="best")

  
  fig.savefig(f"hola_{i}.png", format="png")

