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
	print(synthetic.shape)
	print(objcon.shape)
	print(Q.shape)

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
	return error, phi

def masc(treated, donors, treatment, m_tune_params, min_preperiods):
	error_list = []
	phi_list = []
	for m in m_tune_params:
		error, phi = cv_masc(treated, donors, treatment, m, min_preperiods)
		error_list.append(error)
		phi_list.append(phi)

	min_index = error_list.index(min(error_list))
	
	# return m and phi
	return m_tune_params[min_index], phi_list[min_index]



def weights_from_pi(treated, donors, treatment, pi):
	# given pi, modifies the original synthetic control weights
	donors = donors[donors.index < treatment].to_numpy().T
	treated = treated[treated.index < treatment].to_numpy().T
	
	objcon = (1-pi) * np.dot(treated, treated.T)
	Q = (1-pi) * np.dot(donors, donors.T)

	print(objcon.shape)
	print(Q.shape)


	distance_norm = []
	for i in range(donors.shape[0]):
		distance_norm.append(np.linalg.norm(donors[i] - treated.flatten())**2)
	synthetic = -2*(1-pi) * np.dot(treated, donors.T) + pi*np.array(distance_norm)
	print(synthetic.shape)
	print(synthetic)

	model = gp.Model()

	numdonors = donors.shape[0]


	
	weights = model.addMVar(numdonors, lb = 0, ub = 1)

	model.addConstr((sum(weights[i] for i in range(numdonors)) == 1), name = 'weights')
	model.setObjective( (synthetic @ weights + weights @ Q @ weights), GRB.MINIMIZE)

	model.optimize()

	weights = []
	for v in model.getVars():
		weights.append(v.x)
	return np.array(weights)
	# model.setObjective(((1-pi) * (objcon + synthetic @ weights + weights @ Q @ weights) + pi * (ones @ (distance @ weights))), GRB.MINIMIZE)
	# interpolation bias

def loss_from_pi(treated, donors, treatment, min_preperiods, pi):
	weights_f = np.ones(treatment-2-min_preperiods)
	set_f = list(range(min_preperiods, treatment-1))

	minlength = 1
	maxlength = 1

	# solve matching and sc estimators for each fold
	gamma_sc_pi = []
	Y_treated = []
	for fold in set_f:
		interval = list(range(fold + minlength, min(fold+maxlength, treatment-1) + 1))
		temp_treatment = fold+1

		pi_weights = weights_from_pi(treated, donors, temp_treatment, pi)
		pi_weights = np.expand_dims(pi_weights, axis = 1)

		# double check that this is the correct forecasting
		control_donors = df_donors[df_donors.index.isin(interval)].to_numpy()
		gamma_sc_pi.append(np.dot(control_donors, pi_weights)[0][0])
		Y_treated.append(df_treated[df_treated.index.isin(interval)].to_numpy()[0][0])
		
	gamma_sc_pi = np.array(gamma_sc_pi)
	Y_treated = np.array(Y_treated)

	error = np.linalg.norm(Y_treated - gamma_sc_pi)
	return error

def penalized_sc(treated, donors, treatment, min_preperiods):
	pis = np.arange(0, .2, .005).tolist()
	errors = []
	for pi in pis:
		errs = loss_from_pi(treated, donors, treatment, min_preperiods, pi)
		errors.append(errs)

	min_index = errors.index(min(errors))
	pi_opt = pis[min_index]
	pi_opt = .005

	print(pi_opt)

	weights = weights_from_pi(treated, donors, treatment, pi_opt)

	print(pi_opt)
	return weights

synthetic_control(df_donors, df_treated, treatment)

penalized_sc(df_treated, df_donors, treatment, min_preperiods)



opt_m, opt_phi = masc(df_treated, df_donors, treatment, m_tune_params, min_preperiods)
print(opt_m, opt_phi)

# finally construct the masc for basque data
ma_weights = nn_match(df_donors, df_treated, treatment, opt_m)
ma_weights = np.expand_dims(ma_weights, axis = 1)
sc_weights = synthetic_control(df_donors, df_treated, treatment)
sc_weights = np.expand_dims(sc_weights, axis = 1)
pensc_weights = penalized_sc(df_treated, df_donors, treatment, min_preperiods)
pensc_weights = np.expand_dims(pensc_weights, axis = 1)

masc_weights = np.array(opt_phi * ma_weights + (1-opt_phi) * sc_weights)

Y_control_ma = np.dot(ma_weights.T, df_donors.to_numpy().T)
Y_control_sc = np.dot(sc_weights.T, df_donors.to_numpy().T)
Y_control_psc = np.dot(sc_weights.T, df_donors.to_numpy().T)


Y_control = np.dot(masc_weights.T, df_donors.to_numpy().T)

Y_treated = df_treated.to_numpy().T

Ys = 1000*(Y_treated - Y_control).T

Ytreated = Ys[15:]

X = list(range(len(Ys)))
X = [x+1954 for x in X]


fig, ax = plt.subplots()

ax.plot(X, Ys, color = 'black')
ax.axhline(y=0, xmin=0.0, xmax=1.0, linestyle = '-', color='black', linewidth = .6)
ax.axhline(y=np.mean(Ytreated), xmin = 0.0, xmax = 1.0, linestyle = '-', color = 'dodgerblue', label = 'Mean effect', linewidth = .6)
ax.axvline(x=1969.5, ymin = 0.0, ymax = 1.0, linestyle = '-', color = 'coral', linewidth = .6, label = 'Treatment period')
ax.legend(loc = 'lower left', fontsize = 8)
ax.set(xlabel = 'Year', ylabel = 'Difference in Effect (per capita GDP)')
fig.savefig("fig10.png")

plt.show()

fig, ax = plt.subplots()
Y1 = 1000*(Y_control - Y_control_ma).T
Y2 = 1000*(Y_control - Y_control_sc).T
Y3 = 1000*(Y_control - Y_control_psc).T
ax.plot(X, Y1, color = 'coral', label = 'Matching')
ax.plot(X, Y2, color = 'dodgerblue', label = 'Synthetic Control and Penalized synthetic control')
ax.axhline(y=0, xmin = 0.0, xmax = 1.0, linestyle = '-', linewidth = .6, color = 'black')
ax.plot(X, Y3, color = 'dodgerblue')
ax.legend(loc = 'lower left', fontsize = 8)
ax.set(xlabel = 'Year', ylabel = 'Difference in Effect (per capita GDP)')
fig.savefig("fig11.png")

plt.show()






