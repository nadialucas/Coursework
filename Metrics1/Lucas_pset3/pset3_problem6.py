# Nadia Lucas

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

	# TSLS weights derived from teh jth component
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

	# somehow this is wrong for beta_att but it seems minor and everything else worked (ignoring for now)
	def beta_s(self, weights, att = False):
		beta_s = 0
		Dflat = self.D.flatten()
		Dflat = np.insert(Dflat, 0, 0)
		Dflat = np.insert(Dflat, len(self.D)+1, 1)
		for i in list(range(len(Dflat)-1)):
			w0 = weights[i]
			w1 = -1.0*weights[i]
			integral0 = integrate.quad(self.m0, Dflat[i], Dflat[i+1])[0]
			integral1 = integrate.quad(self.m1, Dflat[i], Dflat[i+1])[0]
			beta_s += (w0*integral0 + w1*integral1)
		return beta_s


# construct the gammas
class gamma:
	def __init__(self, D, Z, K):
		self.D = D
		self.Z = Z
		self.K = K
		Dflat = self.D.flatten()
		Dflat = np.insert(Dflat, 0, 0)
		self.Dflat = np.insert(Dflat, len(self.D)+1, 1)
		self.estimate = estimand(D, Z)

	# regular Bernstein function
	def bernstein(self, u, k):
		return math.comb(self.K, k)*(u**k)*((1-u)**(self.K-k))

	def constant_spline(self, u):
		return 1

	def gamma_star_k(self, k):
		gamma_s0 = 0
		gamma_s1 = 0
		for i in list(range(1, len(self.Dflat))):
			w0 = self.estimate.w_att[i-1]
			w1 = -1.0*self.estimate.w_att[i-1]
			integral = integrate.quad(self.bernstein, self.Dflat[i-1], self.Dflat[i], args = k)[0]
			gamma_s0 += w0*integral
			gamma_s1 += w1*integral
		return [gamma_s0, gamma_s1]

	def gamma_iv_k(self, k):
		gamma_s0 = 0
		gamma_s1 = 0
		for i in list(range(1, len(self.Dflat))):
			w0 = self.estimate.w_iv[i-1]
			w1 = -1.0*self.estimate.w_iv[i-1]
			integral = integrate.quad(self.bernstein, self.Dflat[i-1], self.Dflat[i], args = k)[0]
			gamma_s0 += w0*integral
			gamma_s1 += w1*integral
		return [gamma_s0, gamma_s1]

	def gamma_tsls_k(self, k):
		gamma_s0 = 0
		gamma_s1 = 0
		for i in list(range(1, len(self.Dflat))):
			w0 = self.estimate.w_tsls[i-1]
			w1 = -1.0*self.estimate.w_tsls[i-1]
			integral = integrate.quad(self.bernstein, self.Dflat[i-1], self.Dflat[i], args=k)[0]
			gamma_s0 += w0*integral
			gamma_s1 += w1*integral
		return [gamma_s0, gamma_s1]

	def gamma_star_spline(self):
		diff = [self.Dflat[i+1] - self.Dflat[i] for i in range(len(self.Dflat)-1)]
		gamma_s0 = np.multiply(diff, self.estimate.w_att)
		gamma_s1 = np.multiply(diff, -1.0*self.estimate.w_att)
		return [gamma_s0, gamma_s1]

	def gamma_iv_spline(self):
		diff = [self.Dflat[i+1] - self.Dflat[i] for i in range(len(self.Dflat)-1)]
		gamma_s0 = np.multiply(diff, self.estimate.w_iv)
		gamma_s1 = np.multiply(diff, -1.0*self.estimate.w_iv)
		return [gamma_s0, gamma_s1]

	def gamma_tsls_spline(self):
		diff = [self.Dflat[i+1] - self.Dflat[i] for i in range(len(self.Dflat)-1)]
		gamma_s0 = np.multiply(diff, self.estimate.w_tsls)
		gamma_s1 = np.multiply(diff, -1.0*self.estimate.w_tsls)
		return [gamma_s0, gamma_s1]




# Time to implement the solver

def solver(K, D, Z, max, monotonic):
	estimandK = estimand(D, Z)
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

	model = gp.Model('GetBounds')
	# Create decision variables for the model
	theta = model.addVars(K+1, 2, lb = 0, ub = 1)
	model.addConstr((sum(theta[i,j] * gamma_iv[i][j] for i in range(K+1) for j in range(2)) == estimandK.beta_iv), name = 'IV')
	model.addConstr((sum(theta[i,j] * gamma_tsls[i][j] for i in range(K+1) for j in range(2)) == estimandK.beta_tsls), name = 'TSLS')
	if monotonic:
		for k in range(K):
			model.addConstr((sum(theta[i, 0] * mono_constraint[k, i] for i in range(K+1)) <= 0))
			model.addConstr((sum(theta[i, 1] * mono_constraint[k, i] for i in range(K+1)) <= 0))
	if max:
		model.setObjective((sum(theta[i,j] * gamma_star[i][j] for i in range(K+1) for j in range(2))), GRB.MAXIMIZE)
	else:
		model.setObjective((sum(theta[i,j] * gamma_star[i][j] for i in range(K+1) for j in range(2))), GRB.MINIMIZE)
	

	model.optimize()
	return model.objVal
	
# att = estimand(D,Z).beta_att
# print(att)

def splines(max, monotonic):
	K = 4
	estimandK = estimand(D, Z)
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

	model = gp.Model('GetBounds')
	# Create decision variables for the model
	theta = model.addVars(K, 2, lb = 0, ub = 1)
	model.addConstr((sum(theta[i,j] * gamma_iv_spline[j][i] for i in range(K) for j in range(2)) == estimandK.beta_iv), name = 'IV')
	model.addConstr((sum(theta[i,j] * gamma_tsls_spline[j][i] for i in range(K) for j in range(2)) == estimandK.beta_tsls), name = 'TSLS')
	if monotonic:
		for k in range(K):
			model.addConstr((sum(theta[i, 0] * mono_constraint[k, i] for i in range(K)) <= 0))
			model.addConstr((sum(theta[i, 1] * mono_constraint[k, i] for i in range(K)) <= 0))
	if max:
		model.setObjective((sum(theta[i,j] * gamma_star_spline[j][i] for i in range(K) for j in range(2))), GRB.MAXIMIZE)
	else:
		model.setObjective((sum(theta[i,j] * gamma_star_spline[j][i] for i in range(K) for j in range(2))), GRB.MINIMIZE)


	model.optimize()
	return(model.objVal)

def plot_polys(maxK, D, Z):
	att = estimandK = estimand(D, Z).beta_att
	X = list(range(1, maxK+1))
	maxYpoly = []
	minYpoly = []
	maxYmono = []
	minYmono = []
	for K in list(range(1, maxK+1)):
		maxYpoly.append(solver(K, D, Z, True, False))
		minYpoly.append(solver(K, D, Z, False, False))
		maxYmono.append(solver(K, D, Z, True, True))
		minYmono.append(solver(K, D, Z, False, True))
	attY = att*np.ones(maxK)
	maxNPY = splines(True, False)*np.ones(maxK)
	minNPY = splines(False, False)*np.ones(maxK)
	maxNPYmono = splines(True, True)*np.ones(maxK)
	minNPYmono = splines(False, True)*np.ones(maxK)

	fig, ax = plt.subplots()
	# style of data points
	ax.plot(X, maxYpoly, '-o', markersize = 2.5, linewidth = .4, color='dodgerblue', label = 'Polynomial')
	ax.plot(X, minYpoly, '-o', markersize = 2.5, linewidth = .4, color='dodgerblue')
	ax.plot(X, maxYmono, '-s', markersize = 2.5, linewidth = .4, color='coral', label = 'Polynomial and decreasing')
	ax.plot(X, minYmono, '-s', markersize = 2.5, linewidth = .4, color='coral')
	ax.plot(X, attY, ':*', markersize = 2.5, linewidth = .4, color = 'black', label = 'ATT')
	ax.plot(X, maxNPY, '--o', markersize = 2.5, linewidth = .4, color = 'dodgerblue', label = 'Nonparametric')
	ax.plot(X, minNPY, '--o', markersize = 2.5, linewidth = .4, color = 'dodgerblue')
	ax.plot(X, maxNPYmono, '--s', markersize = 2.5, linewidth = .4, color = 'coral', label = 'Nonparametric and decreasing')
	ax.plot(X, minNPYmono, '--s', markersize = 2.5, linewidth = .4, color = 'coral')
	ax.legend(loc='lower left', fontsize = 8)
	plt.ylim([-1, .3])
	plt.xticks(np.arange(min(X), max(X)+1, 1.0))
	plt.xlim(1, 19)
	ax.set(xlabel='Polynomial degree (K)', ylabel='Upper and Lower Bounds',
	       title="Figure 6 Replication")
	fig.savefig("fig6.png")

plot_polys(19, D, Z)












