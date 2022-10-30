def hola():
  print ("hola")


def hola2():
  print ("hola")

import random
import numpy as np

def pointGenerator():
  return np.array([
    random.uniform(-10, 10),
    random.uniform(-20, 20),
  ])

def instanceGenerator():
  return np.array([
    pointGenerator() for _ in range(10)
  ])
