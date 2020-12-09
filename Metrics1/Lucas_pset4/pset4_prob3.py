# Nadia Lucas
# ECON 31720
# Problem Set 4, Question 3
# Replication Figures 10 and 11 in (MASC)
# December 12, 2020
# 
# I would like to thank Yixin Sun and George Vojta for helpful comments
# 
# Using the general method outlined in Maxwell Kellogg's github
# 
# running on Python version 3.8.6
################################################################################
import pandas as pd
import math
import numpy as np
import copy
import gurobipy as gp
from gurobipy import *
import matplotlib.pyplot as plt

# change this directory accordingly to match personal machine
psdir = "/Users/nadialucas/Dropbox/Second year/ECON 31720/ps4_files/"

df = pd.read_csv(psdir + "basque.csv")

# nearest neighbor weight vector

def nn_match(donors, treated, treatment, m):
	donors = donors[donors.index <= treatment]
	treated = treated[treated.index <= treatment]

def synthetic_control(donors, treated, treatment):
	donors = donors[donors.index <= 1969]
	treated = treated[treated.index<=1969]
	synthetic = -2*np.dot(treated.T, donors)
	print(synthetic)
	objcon = np.dot(treated.T, treated)
	Q = np.dot(donors.T, donors)
	model = gp.Model()

	numdonors = donors.shape[1]

	weights = model.addMVar(numdonors, lb = 0, ub = 1)
	# weights = model.addVars(numdonors, 1, lb = 0, ub = 1)
	model.addConstr((sum(weights[i] for i in range(numdonors)) == 1), name = 'weights')
	# extrapolation bias

	# model.addMQCons(Q, None, '=')
	# model.addObjCon(objcon)
	print((synthetic).shape)
	print(weights)
	model.setObjective((objcon + synthetic @ weights + weights @ Q @ weights ), GRB.MINIMIZE)

	model.optimize()
	print(model.objVal)
	for v in model.getVars():
		print('%s %g' % (v.varName, v.x))



def solve_masc(donors, treated, treatment):
	hi = 1

treatment = 1969
# get donors
df_donors = df[df.regionno != 17]
df_donors = df_donors[['regionno', 'year', 'gdpcap']]
df_donors = df_donors.pivot(index = 'year', columns = 'regionno', values = 'gdpcap')
print(df_donors.head())
df_treated = df[df.regionno == 17]
df_treated = df_treated[['regionno', 'year', 'gdpcap']]
df_treated = df_treated.pivot(index = 'year', columns = 'regionno', values = 'gdpcap')
print(df_treated.head())

treated = df_treated[df_treated.index<=1969]

synthetic_control(df_donors, df_treated, 1969)
