# Nadia Lucas
# ECON 31720
# Problem Set 4, Question 3
# Replication Figures 10 and 11 in (MASC)
# December 12, 2020
# 
# I would like to thank Yixin Sun and George Vojta for helpful comments
# 
# Using the general method outlined in Maxwell Kellogg's github but tailored
# to replicating figures 10 and 11
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
df['year'] = df['year'] - 1954

# nearest neighbor weight vector

def nn_match(donors, treated, treatment, m):
	# using dictionaries, not the most efficient method
	# but I'm sticking with this orientation of my dataframes
	donors = donors[donors.index < treatment]
	treated = treated[treated.index < treatment]
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
		neighbors.append(sorted_dist[i][1]-1)
	weights = np.zeros(donors.shape[1])
	weights[neighbors] = 1.0/m
	return weights
	

def synthetic_control(donors, treated, treatment):
	donors = donors[donors.index < treatment]
	treated = treated[treated.index < treatment]
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
	return(np.array(weights))



def solve_masc(donors, treated, treatment):
	hi = 1

treatment = 16
# get donors
df_donors = df[df.regionno != 17]
df_donors = df_donors[['regionno', 'year', 'gdpcap']]
df_donors = df_donors.pivot(index = 'year', columns = 'regionno', values = 'gdpcap')
df_treated = df[df.regionno == 17]
df_treated = df_treated[['regionno', 'year', 'gdpcap']]
df_treated = df_treated.pivot(index = 'year', columns = 'regionno', values = 'gdpcap')


weights_sc = synthetic_control(df_donors, df_treated, treatment)
weights_match = nn_match(df_donors, df_treated, treatment, 8)


# now it's time to get phi
m_tune_params = list(range(1,11))



min_preperiods = math.ceil(treatment/2)


#what is phis idk 



def cv_masc(treated, donors, treatment, m, min_preperiods, phis=[]):
	print(treatment)
	weights_f = np.ones(treatment-2-min_preperiods)
	set_f = list(range(min_preperiods, treatment-1))

	minlength = 1
	maxlength = 1


	# solve matching and sc estimators for each fold
	gamma_sc = []
	gamma_match = []
	Y_treated = []
	for fold in set_f:
		interval = list(range(fold + minlength, min(fold+maxlength, treatment-1) + 1))
		temp_treatment = fold+1



		sc_weights = synthetic_control(df_donors, df_treated, temp_treatment)
		sc_weights = np.expand_dims(sc_weights, axis = 1)
		match_weights = nn_match(df_donors, df_treated, temp_treatment, m)
		match_weights = np.expand_dims(match_weights, axis = 1)

		# double check that this is the correct forecasting
		control_donors = df_donors[df_donors.index.isin(interval)].to_numpy()
		gamma_match.append(np.dot(control_donors, match_weights)[0][0])
		gamma_sc.append(np.dot(control_donors, sc_weights)[0][0])
		Y_treated.append(df_treated[df_treated.index.isin(interval)].to_numpy()[0][0])
		
	gamma_match = np.array(gamma_match)
	gamma_sc = np.array(gamma_sc)
	Y_treated = np.array(Y_treated)
	phi = np.sum(np.multiply((gamma_match - gamma_sc), (Y_treated - gamma_sc)))/np.sum(np.multiply((gamma_match - gamma_sc),(gamma_match - gamma_sc)))
	phi = max(0,phi)
	phi = min(1,phi)

	error = np.sum(np.power((Y_treated - phi*gamma_match - (1-phi)*gamma_sc), 2))
	# error = np.power(Y_treated - phi * Y_match - (1-phi)*Y_sc, 2)
	# print(error)
	# store error, phi
	return error, phi

def masc(treated, donors, treatment, m_tune_params, min_preperiods, phis=[]):
	error_list = []
	phi_list = []
	for m in m_tune_params:
		error, phi = cv_masc(treated, donors, treatment, m, min_preperiods)
		error_list.append(error)
		phi_list.append(phi)

	min_index = error_list.index(min(error_list))
	
	# return m and phi

	return m_tune_params[min_index], phi_list[min_index]




opt_m, opt_phi = masc(df_treated, df_donors, treatment, m_tune_params, min_preperiods)

# finally construct the masc for basque data
ma_weights = nn_match(df_donors, df_treated, treatment, opt_m)
sc_weights = synthetic_control(df_donors, df_treated, treatment)

masc_weights = np.array(opt_phi * ma_weights + (1-opt_phi) * sc_weights)
masc_weights = np.expand_dims(masc_weights, axis = 1)


Y_control = np.dot(masc_weights.T, df_donors.to_numpy().T)

Y_treated = df_treated.to_numpy().T

Ys = (Y_treated - Y_control).T

X = list(range(len(Ys)))

print(X)


fig, ax = plt.subplots()

ax.plot(X, Ys)

plt.show()




