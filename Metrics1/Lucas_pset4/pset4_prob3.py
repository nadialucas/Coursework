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

def nn_match(donors, treated, treatment, m=5):
	# using dictionaries, not the most efficient method
	# but I'm sticking with this orientation of my dataframes
	donors = donors[donors.index <= treatment]
	treated = treated[treated.index <= treatment]
	# print(donors.sub(pd.Series(treated), axis = 0))
	dist = {}
	for (columnName, columnData) in donors.iteritems():
		if columnName != 18:
			dist[np.sum(np.power((donors[columnName] - treated[17]), 2))] = columnName
		else:
			dist[np.sum(np.power((donors[columnName] - treated[17]), 2))] = columnName-1

	sorted_dist = sorted(dist.items())
	neighbors = []
	for i in range(m):
		neighbors.append(sorted_dist[i][1])
	weights = np.zeros(donors.shape[1])
	weights[neighbors] = 1.0/m
	return weights
	

def synthetic_control(donors, treated, treatment):
	donors = donors[donors.index <= treatment]
	treated = treated[treated.index <= treatment]
	synthetic = -2*np.dot(treated.T, donors)
	objcon = np.dot(treated.T, treated)
	Q = np.dot(donors.T, donors)
	model = gp.Model()

	numdonors = donors.shape[1]

	weights = model.addMVar(numdonors, lb = 0, ub = 1)
	# weights = model.addVars(numdonors, 1, lb = 0, ub = 1)
	model.addConstr((sum(weights[i] for i in range(numdonors)) == 1), name = 'weights')
	# extrapolation bias
	model.setObjective((objcon + synthetic @ weights + weights @ Q @ weights ), GRB.MINIMIZE)

	model.optimize()
	weights = []
	for v in model.getVars():
		weights.append(v.x)
	print(weights)
	return(weights)



def solve_masc(donors, treated, treatment):
	hi = 1

treatment = 1969
# get donors
df_donors = df[df.regionno != 17]
df_donors = df_donors[['regionno', 'year', 'gdpcap']]
df_donors = df_donors.pivot(index = 'year', columns = 'regionno', values = 'gdpcap')
df_treated = df[df.regionno == 17]
df_treated = df_treated[['regionno', 'year', 'gdpcap']]
df_treated = df_treated.pivot(index = 'year', columns = 'regionno', values = 'gdpcap')

weights_sc = synthetic_control(df_donors, df_treated, 1969)
weights_match = nn_match(df_donors, df_treated, 1969)


# not it's time to get phi



