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
		self.w_iv0, self.w_iv1 = self.w_iv()
		self.w_tsls0, self.w_tsls1 = self.w_tsls()
		self.w_att0, self.w_att1 = self.w_att()
		# uses all the helper function to return the estimands
		self.beta_iv = self.beta_s(self.w_iv0, self.w_iv1)
		self.beta_tsls = self.beta_s(self.w_tsls0, self.w_tsls1)
		self.beta_att = self.beta_s(self.w_att0, self.w_att1, True)

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
		# transform s into weights
		for j in range(len(self.D)):
			weights0.append(sum(s[:j+1])/len(self.D))
			weights1.append(sum(s[j:])/len(self.D))
		return weights0, weights1

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
		for j in range(len(self.D)):
			weights0.append(sum(s_list[:j+1])/len(self.D))
			weights1.append(sum(s_list[j:])/len(self.D))
		return weights0, weights1

	def w_att(self):
		ED = np.sum(self.D)/len(self.D)
		s = (1.0/ED)*np.ones(len(self.D))
		weights1 = []
		for j in range(len(self.D)):
			weights1.append(sum(s[j:])/len(self.D))
		weights0 = np.multiply(-1.0, weights1)
		return weights0, weights1

	# somehow this is wrong for beta_att but it seems minor and everything else worked (ignoring for now)
	def beta_s(self, weights0, weights1, att = False):
		beta_s = 0
		Dflat = self.D.flatten()
		Dflat = np.insert(Dflat, 0, 0)
		Dflat = np.insert(Dflat, len(self.D)+1, 1)
		for i in list(range(1, len(self.D)+1)):
			w0 = weights0[i-1]
			w1 = weights1[i-1]
			if att:
				integral0 = integrate.quad(self.m0, Dflat[i-1], Dflat[i])[0]
			else:
				integral0 = integrate.quad(self.m0, Dflat[i], Dflat[i+1])[0]
			integral1 = integrate.quad(self.m1, Dflat[i-1], Dflat[i])[0]
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

	def gamma_star_k(self, k):
		gamma_s0 = 0
		gamma_s1 = 0
		for i in list(range(1, len(self.D)+1)):
			w0 = self.estimate.w_att0[i-1]
			w1 = self.estimate.w_att1[i-1]
			integral = integrate.quad(self.bernstein, self.Dflat[i-1], self.Dflat[i], args = k)[0]
			gamma_s0 += w0*integral
			gamma_s1 += w1*integral
		return [gamma_s0, gamma_s1]

	def gamma_iv_k(self, k):
		gamma_s0 = 0
		gamma_s1 = 0
		for i in list(range(1, len(self.D)+1)):
			w0 = self.estimate.w_iv0[i-1]
			w1 = self.estimate.w_iv1[i-1]
			integral0 = integrate.quad(self.bernstein, self.Dflat[i], self.Dflat[i+1], args=k)[0]
			integral1 = integrate.quad(self.bernstein, self.Dflat[i-1], self.Dflat[i], args = k)[0]
			gamma_s0 += w0*integral0
			gamma_s1 += w1*integral1
		return [gamma_s0, gamma_s1]

	def gamma_tsls_k(self, k):
		gamma_s0 = 0
		gamma_s1 = 0
		for i in list(range(1, len(self.D)+1)):
			w0 = self.estimate.w_tsls0[i-1]
			w1 = self.estimate.w_tsls1[i-1]
			integral0 = integrate.quad(self.bernstein, self.Dflat[i], self.Dflat[i+1], args=k)[0]
			integral1 = integrate.quad(self.bernstein, self.Dflat[i-1], self.Dflat[i], args = k)[0]
			gamma_s0 += w0*integral0
			gamma_s1 += w1*integral1
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
	
att = estimand(D,Z).beta_att
print(att)

def plot_polys(maxK, D, Z):
	att = estimand(D,Z).beta_att
	X = list(range(1, maxK+1))
	maxYpoly = []
	minYpoly = []
	maxYmono = []
	minYmono = []
	attY = att*np.ones(maxK)
	for K in list(range(1, maxK+1)):
		maxYpoly.append(solver(K, D, Z, True, False))
		minYpoly.append(solver(K, D, Z, False, False))
		maxYmono.append(solver(K, D, Z, True, True))
		minYmono.append(solver(K, D, Z, False, True))

	fig, ax = plt.subplots()
	# style of data points
	ax.plot(X, maxYpoly, '-o', markersize = 2, color='dodgerblue')
	ax.plot(X, minYpoly, '-o', markersize = 2, color='dodgerblue')
	ax.plot(X, maxYmono, '-x', markersize = 2, color='red')
	ax.plot(X, minYmono, '-x', markersize = 2, color='red')
	ax.plot(X, attY, '-v', markersize = 2, color = 'black')
	# predicted means
	#ax.plot(plot_x, means_y, color = 'dodgerblue', alpha = 0.7)
	# shade in the standard deviation around the means
	ax.set(xlabel='X', ylabel='Y',
	       title="Figure 6")
	fig.savefig("fig6.png")
	plt.show()
	print(att)

#plot_polys(19, D, Z)

# debug my ATT issues
print(solver(1, D, Z, True, False))
