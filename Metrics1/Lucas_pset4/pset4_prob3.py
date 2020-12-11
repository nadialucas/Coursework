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
# clean up the years for the event study to make sense
df['year'] = df['year'] - 1954

# establish certain features of the dataset
treatment = 16
# get donors and reshape for ease of use
df_donors = df[df.regionno != 17]
df_donors = df_donors[['regionno', 'year', 'gdpcap']]
df_donors = df_donors.pivot(index='year', columns='regionno', values='gdpcap')
# get treated and reshape for ease of use
df_treated = df[df.regionno == 17]
df_treated = df_treated[['regionno', 'year', 'gdpcap']]
df_treated = df_treated.pivot(index='year', columns='regionno', values='gdpcap')



################################################################################
# MASC
################################################################################

# nearest neighbor weight vector
def nn_match(donors, treated, treatment, m):
	# using dictionaries
	donors = donors[donors.index < treatment]
	treated = treated[treated.index < treatment]
	dist = {}
	# append distances
	for (columnName, columnData) in donors.iteritems():
		if columnName != 18:
			dist[np.sum(np.power((donors[columnName] -
				treated[17]),2))] = columnName
		else:
			dist[np.sum(np.power((donors[columnName] - 
				treated[17]), 2))] = columnName-1

	sorted_dist = sorted(dist.items())
	neighbors = []
	# cycle through distances in order for the m-nearest neighbors
	for i in range(m):
		neighbors.append(sorted_dist[i][1]-1)
	weights = np.zeros(donors.shape[1])
	# weight the m nearest neighbors evenly
	weights[neighbors] = 1.0/m
	return weights
	
# get the synthetic control weights using gurobi for python
def synthetic_control(donors, treated, treatment):
	# parse out the donors and treated before the treatment date
	donors = donors[donors.index < treatment]
	treated = treated[treated.index < treatment]
	constant = np.dot(treated.T, treated)
	linear = -2*np.dot(treated.T, donors)
	quadratic = np.dot(donors.T, donors)
	# set up gurobi
	model = gp.Model()
	numdonors = donors.shape[1]
	# instantiate a vector of weights to solve
	weights = model.addMVar(numdonors, lb = 0, ub = 1)
	# require all weights to sum to 1
	model.addConstr((sum(weights[i] for i in range(numdonors)) == 1))
	# the quadratic equation that anlytically is extrapolation bias
	# which we want to minimize
	model.setObjective((constant + linear@weights + weights@quadratic@weights), 
		GRB.MINIMIZE)
	model.optimize()
	# extract the weight vector to return
	weights = []
	for v in model.getVars():
		weights.append(v.x)
	return(np.array(weights))


# helper function that does cross validation
def cv_masc(treated, donors, treatment, m, min_preperiods, phis=[]):
	# on a high level, the function will take in m, the matching parameter
	# and returns te optimal phi and error associated with the masc
	weights_f = np.ones(treatment-2-min_preperiods)
	set_f = list(range(min_preperiods, treatment-1))
	# in line with the paper, set the length of the folds
	minlength = 1
	maxlength = 1
	# solve matching and sc estimators for each fold
	gamma_sc = []
	gamma_match = []
	Y_treated = []
	# step through each fold and perform the cross validation
	for fold in set_f:
		# this comes out to an interval size of 1 in our case
		# but is in theory flexible to other interval lengths for cv
		interval = list(range(fold+minlength,min(fold+maxlength,treatment-1)+1))
		# set the 'treatment' we use in the cross validation
		temp_treatment = fold+1
		# get both the synthetic control and matching weights
		sc_weights = synthetic_control(df_donors, df_treated, temp_treatment)
		sc_weights = np.expand_dims(sc_weights, axis = 1)
		match_weights = nn_match(df_donors, df_treated, temp_treatment, m)
		match_weights = np.expand_dims(match_weights, axis = 1)
		# construct the predicted counterfactuals and the associated
		# data to compare against
		control_donors = df_donors[df_donors.index.isin(interval)].to_numpy()
		gamma_match.append(np.dot(control_donors, match_weights)[0][0])
		gamma_sc.append(np.dot(control_donors, sc_weights)[0][0])
		Y_treated.append(df_treated[df_treated.index.isin(interval)]
			.to_numpy()[0][0])
	# using the stored values, calculate f
	gamma_match = np.array(gamma_match)
	gamma_sc = np.array(gamma_sc)
	Y_treated = np.array(Y_treated)
	# analytically solve for phi
	phi = np.sum(np.multiply((gamma_match - gamma_sc),(Y_treated - gamma_sc)))/\
	np.sum(np.multiply((gamma_match - gamma_sc),(gamma_match - gamma_sc)))
	phi = max(0,phi)
	phi = min(1,phi)
	# calculate the error
	error = np.sum(np.power((Y_treated - phi*gamma_match-(1-phi)*gamma_sc), 2))
	return error, phi

def masc(treated, donors, treatment, m_tune_params, min_preperiods):
	# cycle through all m tuning parameters and return the optimal
	# m and phi to construct the masc with
	error_list = []
	phi_list = []
	for m in m_tune_params:
		error, phi = cv_masc(treated, donors, treatment, m, min_preperiods)
		error_list.append(error)
		phi_list.append(phi)
	min_index = error_list.index(min(error_list))
	# return m and phi
	return m_tune_params[min_index], phi_list[min_index]

################################################################################
# PENALIZED SYNTHETIC CONTROL
################################################################################

def weights_from_pi(treated, donors, treatment, pi):
	# given pi, modifies the original synthetic control weights
	donors = donors[donors.index < treatment].to_numpy().T
	treated = treated[treated.index < treatment].to_numpy().T
	# similarly, construct the coefficients for the analytically 
	# quadratic loss function
	constant = (1-pi) * np.dot(treated, treated.T)
	quadratic = (1-pi) * np.dot(donors, donors.T)
	# get the extra interpolation bias term
	distance_norm = []
	for i in range(donors.shape[0]):
		distance_norm.append(np.linalg.norm(donors[i] - treated.flatten())**2)
	linear = -2*(1-pi) * np.dot(treated, donors.T)+pi*np.array(distance_norm)
	# now set up gurobi
	model = gp.Model()
	# initialize the vector of weights and ensure they sum to 1
	numdonors = donors.shape[0]
	weights = model.addMVar(numdonors, lb = 0, ub = 1)
	# all weights sum to 1
	model.addConstr((sum(weights[i] for i in range(numdonors)) == 1))
	# set objective to the new quadratic function
	model.setObjective((constant + linear@weights + weights@quadratic@weights), 
		GRB.MINIMIZE)
	model.optimize()
	# extract the weights and return
	weights = []
	for v in model.getVars():
		weights.append(v.x)
	return np.array(weights)

def loss_from_pi(treated, donors, treatment, min_preperiods, pi):
	# helper function to the get the sum of squared errors for any given pi
	weights_f = np.ones(treatment-2-min_preperiods)
	set_f = list(range(min_preperiods, treatment-1))

	minlength = 1
	maxlength = 1
	# solve matching and sc estimators for each fold
	gamma_sc_pi = []
	Y_treated = []
	for fold in set_f:
		interval = list(range(fold+minlength,min(fold+maxlength,treatment-1)+1))
		temp_treatment = fold+1
		# get weights
		pi_weights = weights_from_pi(treated, donors, temp_treatment, pi)
		pi_weights = np.expand_dims(pi_weights, axis = 1)
		# constructed predicted counterfactual for cross validation
		control_donors = df_donors[df_donors.index.isin(interval)].to_numpy()
		gamma_sc_pi.append(np.dot(control_donors, pi_weights)[0][0])
		Y_treated.append(df_treated[df_treated.index.isin(interval)]
			.to_numpy()[0][0])
		
	gamma_sc_pi = np.array(gamma_sc_pi)
	Y_treated = np.array(Y_treated)
	# solve error and return
	error = np.linalg.norm(Y_treated - gamma_sc_pi)
	return error

def penalized_sc(treated, donors, treatment, min_preperiods):
	# grid search over pi and return the weights associated with
	# the pi that gives the lowest error
	pis = np.arange(0, .2, .005).tolist()
	errors = []
	for pi in pis:
		errs = loss_from_pi(treated, donors, treatment, min_preperiods, pi)
		errors.append(errs)
	# find min pi
	min_index = errors.index(min(errors))
	pi_opt = pis[min_index]
	# fina optimal weights and return
	weights = weights_from_pi(treated, donors, treatment, pi_opt)
	return weights


################################################################################
# CALLING ALL FUNCTIONS AND PLOTTING
################################################################################
# set tuning parameters according to best practicesin the paper
m_tune_params = list(range(1,11))
min_preperiods = math.ceil(treatment/2)
# call MASC
opt_m, opt_phi = masc(df_treated, df_donors, 
	treatment, m_tune_params, min_preperiods)

# finally construct the masc for basque data
ma_weights = nn_match(df_donors, df_treated, treatment, opt_m)
ma_weights = np.expand_dims(ma_weights, axis = 1)
sc_weights = synthetic_control(df_donors, df_treated, treatment)
sc_weights = np.expand_dims(sc_weights, axis = 1)
# and penalized synthetic control
pensc_weights = penalized_sc(df_treated, df_donors, treatment, min_preperiods)
pensc_weights = np.expand_dims(pensc_weights, axis = 1)
masc_weights = np.array(opt_phi * ma_weights + (1-opt_phi) * sc_weights)
# find all the predicted counterfactuals are
Y_control_ma = np.dot(ma_weights.T, df_donors.to_numpy().T)
Y_control_sc = np.dot(sc_weights.T, df_donors.to_numpy().T)
Y_control_psc = np.dot(pensc_weights.T, df_donors.to_numpy().T)
# find the masc counterfactual
Y_control = np.dot(masc_weights.T, df_donors.to_numpy().T)
# compare to treated for treatment effect
Y_treated = df_treated.to_numpy().T
Ys = 1000*(Y_treated - Y_control).T
# pull out just treated time periods
Ytreated = Ys[15:]
# plot years
X = list(range(len(Ys)))
X = [x+1954 for x in X]
# plot figure 10
fig, ax = plt.subplots()
ax.plot(X, Ys, color = 'black')
ax.axhline(y=0, xmin=0.0, xmax=1.0, linestyle = '-', color='black', 
	linewidth = .6)
ax.axhline(y=np.mean(Ytreated), xmin = 0.0, xmax = 1.0, linestyle = '-', 
	color = 'dodgerblue', label = 'Mean effect', linewidth = .6)
ax.axvline(x=1969.5, ymin = 0.0, ymax = 1.0, linestyle = '-', 
	color = 'coral', linewidth = .6, label = 'Treatment period')
ax.legend(loc = 'lower left', fontsize = 8)
ax.set(xlabel = 'Year', ylabel = 'Difference in Effect (per capita GDP)')
fig.savefig("fig10.png")

# plot figure 11
fig, ax = plt.subplots()
Y1 = 1000*(Y_control - Y_control_ma).T[14:]
Y2 = 1000*(Y_control - Y_control_sc).T[14:]
Y3 = 1000*(Y_control - Y_control_psc).T[14:]
ax.plot(X[14:], Y1, color = 'coral', label = 'Matching', linewidth = 0.75)
ax.plot(X[14:], Y2, color = 'dodgerblue', label = 'Synthetic Control', 
	linewidth = 0.75, linestyle = ':')
ax.axhline(y=0, xmin = 0.0, xmax = 1.0, linestyle = '-', linewidth = .6, 
	color = 'black')
ax.plot(X[14:], Y3, color = 'purple', linestyle= '--',
 label = 'Penalized Synthetic Control',linewidth = 0.75)
ax.legend(loc = 'upper left', fontsize = 8)
ax.set(xlabel = 'Year', ylabel = 'Difference in Effect (per capita GDP)')
fig.savefig("fig11.png")





