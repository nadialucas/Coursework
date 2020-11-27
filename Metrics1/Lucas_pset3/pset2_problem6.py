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


yo = estimand(D,Z)
print(yo.beta_iv)
print(yo.beta_tsls)
print(yo.beta_att)

# construct the gammas
class gamma:
	def __init__(self, D, Z, K):
		self.D = D
		self.Z = Z
		self.K = K
		Dflat = self.D.flatten()
		Dflat = np.insert(Dflat, 0, 0)
		self.Dflat = np.insert(Dflat, len(self.D)+1, 1)
		coeflist = []
		powerlist = []
		for k in range(self.K):
			for i in list(range(k, self.K+1)):
				coeflist.append((-1)**(i-k) * math.comb(self.K, i) * math.comb(i, k))
				powerlist.append(i)
		self.coeflist = np.array(coeflist)
		self.powerlist = np.array(powerlist)
		self.estimate = estimand(D, Z)

	# uses list of coefficients and powers to construct a function
	def bernstein(self, u, k):
		#return np.sum(np.multiply(self.coeflist, np.power(u, self.powerlist)))
		return math.comb(self.K, k)*(u**k)*((1-u)**(self.K-k))


	def gamma_star(self, k):
		gamma_s0 = 0
		gamma_s1 = 0
		for i in list(range(1, len(self.D)+1)):
			w0 = self.estimate.w_att0[i-1]
			w1 = self.estimate.w_att1[i-1]
			integral = integrate.quad(self.bernstein, self.Dflat[i-1], self.Dflat[i], args = k)[0]
			gamma_s0 += w0*integral
			gamma_s1 += w1*integral
		return(gamma_s0, gamma_s1)

	def gamma_iv(self, k):
		gamma_s0 = 0
		gamma_s1 = 0
		for i in list(range(1, len(self.D)+1)):
			w0 = self.estimate.w_iv0[i-1]
			w1 = self.estimate.w_iv1[i-1]
			integral0 = integrate.quad(self.bernstein, self.Dflat[i], self.Dflat[i+1], args=k)[0]
			integral1 = integrate.quad(self.bernstein, self.Dflat[i-1], self.Dflat[i], args = k)[0]
			gamma_s0 += w0*integral0
			gamma_s1 += w1*integral1
		return(gamma_s0, gamma_s1)

	def gamma_tsls(self, k):
		gamma_s0 = 0
		gamma_s1 = 0
		for i in list(range(1, len(self.D)+1)):
			w0 = self.estimate.w_tsls0[i-1]
			w1 = self.estimate.w_tsls1[i-1]
			integral0 = integrate.quad(self.bernstein, self.Dflat[i], self.Dflat[i+1], args=k)[0]
			integral1 = integrate.quad(self.bernstein, self.Dflat[i-1], self.Dflat[i], args = k)[0]
			gamma_s0 += w0*integral0
			gamma_s1 += w1*integral1
		return(gamma_s0, gamma_s1)




hi = gamma(D, Z, 1)
print(hi.gamma_star(0))
print(hi.gamma_star(1))
print(hi.gamma_iv(0))
print(hi.gamma_iv(1))
print(hi.gamma_tsls(0))
print(hi.gamma_tsls(1))

