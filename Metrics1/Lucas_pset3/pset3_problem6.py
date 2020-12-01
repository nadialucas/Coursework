# Nadia Lucas
# ECON 31720
# Problem Set 3, Question 6
# December 1, 2020
# 
# I would like to thank Yixin Sun and George Vojta for helpful comments
# 
# running on Python version 3.8.6
################################################################################


import numpy as np 
import pandas as pd 
import scipy.integrate as integrate
import math
import gurobipy as gp
from gurobipy import *
import matplotlib.pyplot as plt

# setting up what the problem gives us
D = np.array([.12, .29, .48, .78])
D = D.reshape((len(D), 1))
Z = np.array([1, 2, 3, 4])
Z = Z.reshape((len(Z), 1))


# class for calculating and storing all weights and betas
class estimand:
	def __init__(self, D, Z):
		self.D = D
		self.Z = Z
		# sanity check that these match up with figure 4
		self.w_iv = self.w_iv()
		self.w_tsls = self.w_tsls()
		self.w_att = self.w_att()
		# uses all the helper function to return the estimands
		self.beta_iv = self.beta_s(self.w_iv)
		self.beta_tsls = self.beta_s(self.w_tsls)
		self.beta_att = self.beta_s(self.w_att, True)

	def m0(self, u):
		return 0.9 - 1.1*u + 0.3*u**2

	def m1(self, u):
		return 0.35 - 0.3*u - 0.05*u**2

	# TSLS weights derived from the jth component
	def w_tsls(self):
		weights0 = []
		weights1 = []
		# using the definition of the instrument in TSLS
		# as a matrix of indicator vectors
		Ztilde = np.diag(np.ones(len(self.Z)))
		# adding a constant to demean treatment
		Xtilde = np.hstack((np.ones((len(self.D), 1)),self.D))
		# calculate expectations and solve
		EZX = np.dot(Ztilde.T, Xtilde)/len(self.Z)
		EZZ = np.dot(Ztilde.T, Ztilde)/len(self.Z)
		Pi = np.dot(EZX.T, np.linalg.inv(EZZ))
		s = np.dot(np.dot(np.linalg.inv(np.dot(Pi, EZX)), Pi), Ztilde)[1]
		s = np.insert(s, 0, 0)
		weights = np.cumsum(s)/len(self.D)
		return weights

	# IV weights coming from table 3
	def w_iv(self):
		s_list = []
		weights0 = []
		weights1 = []
		EDZ = np.dot(self.D.T, self.Z)[0]/len(self.Z)
		EZ = np.sum(self.Z)/len(self.Z)
		ED = np.sum(self.D)/len(self.D)
		CovDZ = (EDZ - EZ*ED)[0] 
		for z in self.Z:
			s=(z[0]-EZ)/CovDZ
			s_list.append(s)
		s = np.insert(s_list, 0, 0)
		weights = np.cumsum(s)/len(self.D)
		return weights

	def w_att(self):
		ED = np.sum(self.D)/len(self.D)
		weights = (-1.0/ED)*np.array([1, .75, .5, .25, 0])
		return weights

	def beta_s(self, weights, att = False):
		beta_s = 0
		# construct a list that includes all the bounds for integration
		bounds = self.D.flatten()
		bounds = np.insert(bounds, 0, 0)
		bounds = np.insert(bounds, len(self.D)+1, 1)
		# using weights and appropriate MTE functions, solve for beta
		# as outlined in the paper
		for i in list(range(len(bounds)-1)):
			w0 = weights[i]
			w1 = -1.0*weights[i]
			integral0 = integrate.quad(self.m0, bounds[i], bounds[i+1])[0]
			integral1 = integrate.quad(self.m1, bounds[i], bounds[i+1])[0]
			beta_s += (w0*integral0 + w1*integral1)
		return beta_s


# construct the gammas
class gamma:
	def __init__(self, D, Z, K):
		self.D = D
		self.Z = Z
		self.K = K
		# constructing a list that includes all the bounds for integration
		bounds = self.D.flatten()
		bounds = np.insert(bounds, 0, 0)
		self.bounds = np.insert(bounds, len(self.D)+1, 1)
		# now create intervals from the bounds
		self.intervals = [self.bounds[i+1] - self.bounds[i] 
			for i in range(len(self.bounds)-1)]
		# store an estimand object to reference weights
		self.estimate = estimand(D, Z)

	# Bernstein polynomial
	def bernstein(self, u, k):
		return math.comb(self.K, k)*(u**k)*((1-u)**(self.K-k))

	def gamma_star_k(self, k):
		gamma_s0 = 0
		gamma_s1 = 0
		# similar to constructing beta_s, cycle through weights and
		# integrate the bernstein polynomial at each relevant interval
		for i in list(range(1, len(self.bounds))):
			w0 = self.estimate.w_att[i-1]
			w1 = -1.0*self.estimate.w_att[i-1]
			integral = integrate.quad(self.bernstein, self.bounds[i-1], 
				self.bounds[i], args = k)[0]
			gamma_s0 += w0*integral
			gamma_s1 += w1*integral
		return [gamma_s0, gamma_s1]

	def gamma_iv_k(self, k):
		gamma_s0 = 0
		gamma_s1 = 0
		# using IV weights this time, similarly construct the gamma_s for IV
		for i in list(range(1, len(self.bounds))):
			w0 = self.estimate.w_iv[i-1]
			w1 = -1.0*self.estimate.w_iv[i-1]
			integral = integrate.quad(self.bernstein, self.bounds[i-1], 
				self.bounds[i], args = k)[0]
			gamma_s0 += w0*integral
			gamma_s1 += w1*integral
		return [gamma_s0, gamma_s1]

	def gamma_tsls_k(self, k):
		gamma_s0 = 0
		gamma_s1 = 0
		# using TSLS weights this time, similarly construct the gamma_s for TSLS
		for i in list(range(1, len(self.bounds))):
			w0 = self.estimate.w_tsls[i-1]
			w1 = -1.0*self.estimate.w_tsls[i-1]
			integral = integrate.quad(self.bernstein, self.bounds[i-1], 
				self.bounds[i], args=k)[0]
			gamma_s0 += w0*integral
			gamma_s1 += w1*integral
		return [gamma_s0, gamma_s1]

	def gamma_star_spline(self):
		# for the constant spline, we integrate over an indicator function
		# so relevan weights are just multiplied by the length of the interval
		gamma_s0 = np.multiply(self.intervals, self.estimate.w_att)
		gamma_s1 = np.multiply(self.intervals, -1.0*self.estimate.w_att)
		return [gamma_s0, gamma_s1]

	def gamma_iv_spline(self):
		gamma_s0 = np.multiply(self.intervals, self.estimate.w_iv)
		gamma_s1 = np.multiply(self.intervals, -1.0*self.estimate.w_iv)
		return [gamma_s0, gamma_s1]

	def gamma_tsls_spline(self):
		gamma_s0 = np.multiply(self.intervals, self.estimate.w_tsls)
		gamma_s1 = np.multiply(self.intervals, -1.0*self.estimate.w_tsls)
		return [gamma_s0, gamma_s1]

def solver(K, D, Z, max, monotonic):
	# call all relevant weights and constraints
	iv_estimand = estimand(D, Z)
	gammaK = gamma(D, Z, K)
	gamma_star = []
	gamma_iv = []
	gamma_tsls = []
	for k in range(K+1):
		gamma_star.append(gammaK.gamma_star_k(k))
		gamma_iv.append(gammaK.gamma_iv_k(k))
		gamma_tsls.append(gammaK.gamma_tsls_k(k))
	print('gamas')
	print(gamma_star)
	print(gamma_iv)
	print(gamma_tsls)

	# add monotonicity into the solver (decreasing)
	mono_constraint = np.zeros((K, K+1))
	for k in range(K):
		mono_constraint[k, k] = 1
		mono_constraint[k, k+1] = -1

	model = gp.Model('M')
	# Create decision variables for the model
	theta = model.addVars(K+1, 2, lb = 0, ub = 1)
	# once we have our beta_s and gamma_s, we can
	# use the Gurobi API to construct the two constraints
	model.addConstr((sum(theta[i,j] * gamma_iv[i][j] for i in range(K+1) 
		for j in range(2)) == iv_estimand.beta_iv), name = 'IV')
	model.addConstr((sum(theta[i,j] * gamma_tsls[i][j] for i in range(K+1) 
		for j in range(2)) == iv_estimand.beta_tsls), name = 'TSLS')
	# we can additionally impose the monotonically decreasing constraints
	if monotonic:
		for k in range(K):
			model.addConstr((sum(theta[i, 0] * mono_constraint[k, i] 
				for i in range(K+1)) <= 0))
			model.addConstr((sum(theta[i, 1] * mono_constraint[k, i] 
				for i in range(K+1)) <= 0))
	# using our beta*, gamma*, we can set the relevant objective function
	if max:
		model.setObjective((sum(theta[i,j] * gamma_star[i][j] 
			for i in range(K+1) for j in range(2))), GRB.MAXIMIZE)
	else:
		model.setObjective((sum(theta[i,j] * gamma_star[i][j] 
			for i in range(K+1) for j in range(2))), GRB.MINIMIZE)
	
	# and solve
	model.optimize()
	return model.objVal

def splines(max, monotonic):
	K = 4 # for monotonicity to reuse the same method as above
	iv_estimand = estimand(D, Z)
	gamma_spline = gamma(D, Z, K)
	gamma_star_spline = gamma_spline.gamma_star_spline()
	gamma_iv_spline = gamma_spline.gamma_iv_spline()
	gamma_tsls_spline = gamma_spline.gamma_tsls_spline()

	print(gamma_star_spline)
	print(gamma_iv_spline)

	# add monotonicity into the solver (decreasing)
	mono_constraint = np.zeros((K, K+1))
	for k in range(K-1):
		mono_constraint[k, k] = 1
		mono_constraint[k, k+1] = -1

	model = gp.Model('M')
	# Create decision variables for the model
	theta = model.addVars(K, 2, lb = 0, ub = 1)
	# once we have our beta_s (spline) and gamma_s (spline, we can
	# use the Gurobi API to construct the two constraints
	model.addConstr((sum(theta[i,j] * gamma_iv_spline[j][i] for i in range(K) 
		for j in range(2)) == iv_estimand.beta_iv), name = 'IV')
	model.addConstr((sum(theta[i,j] * gamma_tsls_spline[j][i] for i in range(K) 
		for j in range(2)) == iv_estimand.beta_tsls), name = 'TSLS')
	# we can additionally impose the monotonically decreasing constraints
	if monotonic:
		for k in range(K):
			model.addConstr((sum(theta[i, 0] * mono_constraint[k, i] 
				for i in range(K)) <= 0))
			model.addConstr((sum(theta[i, 1] * mono_constraint[k, i] 
				for i in range(K)) <= 0))
	# using our beta*, gamma*, we can set the relevant objective function
	if max:
		model.setObjective((sum(theta[i,j] * gamma_star_spline[j][i] 
			for i in range(K) for j in range(2))), GRB.MAXIMIZE)
	else:
		model.setObjective((sum(theta[i,j] * gamma_star_spline[j][i] 
			for i in range(K) for j in range(2))), GRB.MINIMIZE)
	# and solve
	model.optimize()
	return(model.objVal)

def plot_polys(maxK, D, Z):
	# get the ATT 
	att = estimand(D, Z).beta_att
	X = list(range(1, maxK+1))
	maxYpoly = []
	minYpoly = []
	maxYmono = []
	minYmono = []
	# call the solver for all Ks
	for K in list(range(1, maxK+1)):
		maxYpoly.append(solver(K, D, Z, True, False))
		minYpoly.append(solver(K, D, Z, False, False))
		maxYmono.append(solver(K, D, Z, True, True))
		minYmono.append(solver(K, D, Z, False, True))
	# construct the line of ATTs and the line of NP bounds
	attY = att*np.ones(maxK)
	maxNPY = splines(True, False)*np.ones(maxK)
	minNPY = splines(False, False)*np.ones(maxK)
	maxNPYmono = splines(True, True)*np.ones(maxK)
	minNPYmono = splines(False, True)*np.ones(maxK)

	fig, ax = plt.subplots()
	# create the style for each line and label legend
	ax.plot(X, maxYpoly, '-o', markersize = 2.5, linewidth = .4, 
		color='dodgerblue', label = 'Polynomial')
	ax.plot(X, minYpoly, '-o', markersize = 2.5, linewidth = .4, 
		color='dodgerblue')
	ax.plot(X, maxYmono, '-s', markersize = 2.5, linewidth = .4, 
		color='coral', label = 'Polynomial and decreasing')
	ax.plot(X, minYmono, '-s', markersize = 2.5, linewidth = .4, 
		color='coral')
	ax.plot(X, attY, ':*', markersize = 2.5, linewidth = .4, 
		color = 'black', label = 'ATT')
	ax.plot(X, maxNPY, '--o', markersize = 2.5, linewidth = .4, 
		color = 'dodgerblue', label = 'Nonparametric')
	ax.plot(X, minNPY, '--o', markersize = 2.5, linewidth = .4, 
		color = 'dodgerblue')
	ax.plot(X, maxNPYmono, '--s', markersize = 2.5, linewidth = .4, 
		color = 'coral', label = 'Nonparametric and decreasing')
	ax.plot(X, minNPYmono, '--s', markersize = 2.5, linewidth = .4, 
		color = 'coral')
	ax.legend(loc='lower left', fontsize = 8)
	plt.ylim([-1, .3])
	plt.xticks(np.arange(min(X), max(X)+1, 1.0))
	plt.xlim(1, 19)
	ax.set(xlabel='Polynomial degree (K)', ylabel='Upper and Lower Bounds')
	fig.savefig("fig6.png")

# call the plotting function for all 19 Ks
plot_polys(19, D, Z)











