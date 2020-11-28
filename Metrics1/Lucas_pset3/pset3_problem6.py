# Nadia Lucas

################################################################################


import numpy as np 
import pandas as pd 
import scipy.integrate as integrate
import math
import gurobipy as gp
from gurobipy import *

D = np.array([.12, .29, .48, .78])
D = D.reshape((len(D), 1))
Z = np.array([1, 2, 3, 4])
Z = Z.reshape((len(Z), 1))



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

	def w_tsls(self):
		weights0 = []
		weights1 = []
		Ztilde = np.diag(np.ones(len(self.Z)))
		Xtilde = np.hstack((np.ones((len(self.D), 1)),self.D))

		EZX = np.dot(Ztilde.T, Xtilde)/len(self.Z)
		EZZ = np.dot(Ztilde.T, Ztilde)/len(self.Z)
		Pi = np.dot(EZX.T, np.linalg.inv(EZZ))
		s = np.dot(np.dot(np.linalg.inv(np.dot(Pi, EZX)), Pi), Ztilde)[1]
		for j in range(len(self.D)):
			weights0.append(sum(s[:j+1])/len(self.D))
			weights1.append(sum(s[j:])/len(self.D))
		return weights0, weights1


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
		weights0 = np.multiply((-1.0/ED), np.array([1,.75, .5, .25]))
		weights1 = np.multiply((1.0/ED), np.array([1,.75, .5, .25]))
		return weights0, weights1

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

	# uses list of coefficients and powers to construct a function
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

def solver(K, D, Z, max = True):
	estimandK = estimand(D, Z)
	gammaK = gamma(D, Z, K)
	gamma_star = []
	gamma_iv = []
	gamma_tsls = []
	for k in range(K+1):
		gamma_star.append(gammaK.gamma_star_k(k))
		gamma_iv.append(gammaK.gamma_iv_k(k))
		gamma_tsls.append(gammaK.gamma_tsls_k(k))

	# add monotonicity into the solver
	mono_constraint = np.zeros((K, K+1))
	for k in range(K):
		mono_constraint[k, k] = 1
		mono_constraint[k, k+1] = -1


	print(mono_constraint)
	print(len(gamma_iv))



	model = gp.Model('GetBounds')
	# Create decision variables for the model
	theta = model.addVars(K+1, 2, lb = 0, ub = 1)
	model.addConstr((sum(theta[i,j] * gamma_iv[i][j] for i in range(K+1) for j in range(2)) == estimandK.beta_iv), name = 'IV')
	model.addConstr((sum(theta[i,j] * gamma_tsls[i][j] for i in range(K+1) for j in range(2)) == estimandK.beta_tsls), name = 'TSLS')
	for k in range(K):
		model.addConstr((sum(theta[i, 0] * mono_constraint[k, i] for i in range(K+1)) <= 0))
		model.addConstr((sum(theta[i, 1] * mono_constraint[k, i] for i in range(K+1)) <= 0))
	if max:
		model.setObjective((sum(theta[i,j] * gamma_star[i][j] for i in range(K+1) for j in range(2))), GRB.MAXIMIZE)
	else:
		model.setObjective((sum(theta[i,j] * gamma_star[i][j] for i in range(K+1) for j in range(2))), GRB.MINIMIZE)
	

	model.optimize()

	print(model.objVal)
	return model.objVal
	

solver(19, D, Z)

