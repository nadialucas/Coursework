# Nadia Lucas

################################################################################


import numpy as np 
import pandas as pd 
import scipy.integrate as integrate
import math
import gurobipy as gp
from gurobipy import GRB

D = np.array([.12, .29, .48, .78])
D = D.reshape((len(D), 1))
Z = np.array([1, 2, 3, 4])
Z = Z.reshape((len(Z), 1))


