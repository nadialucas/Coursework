# Nadia Lucas

################################################################################


import numpy as np 
import pandas as pd 
import scipy.integrate as integrate
import math


D = np.array([.12, .29, .48, .78])
D = D.reshape((len(D), 1))
Z = np.array([1, 2, 3, 4])
Z = Z.reshape((len(Z), 1))



class IV_estimand:
	def __init__(self, D, Z):
		self.D = D
		self.Z = Z

	def m0(self, u):
		return 0.9 - 1.1*u + 0.3*u**2

	def m1(self, u):
		return 0.35 - 0.3*u - 0.05*u**2

	def s_tsls(self):
		Ztilde = np.diag(np.ones(len(self.Z)))
		Xtilde = np.hstack((np.ones((len(self.D), 1)),self.D))

		EZX = np.dot(Ztilde.T, Xtilde)/len(self.Z)
		EZZ = np.dot(Ztilde.T, Ztilde)/len(self.Z)
		Pi = np.dot(EZX.T, np.linalg.inv(EZZ))
		return(np.dot(np.dot(np.linalg.inv(np.dot(Pi, EZX)), Pi), Ztilde)[1])

	def s_iv(self):
		s_list = []
		EDZ = np.dot(self.D.T, self.Z)[0]/len(self.Z)
		EZ = np.sum(self.Z)/len(self.Z)
		ED = np.sum(self.D)/len(self.D)
		CovDZ = (EDZ - EZ*ED)[0] 
		for z in self.Z:
			s=(z[0]-EZ)/CovDZ
			s_list.append(s)
		return s_list

	def weights(self):
		s0 = self.s_tsls()[0]
		s1 = self.s_tsls()[1]
		w0 = [sum(s0)/4.0, sum(s0[1:])/3.0, sum(s0[2:])/2.0]

	def w_att(self):
		w_list = []
		Dflat = self.D.flatten()
		for i in range(len(self.D)):
			w1 = sum(Dflat[i:])/len(self.D)
			w_list.append(1/Dflat[i])
		return w_list

	def beta_s0(self, s):
		beta_s = 0
		weights0 = []
		for i in range(len(self.D)):
			w0 = sum(s[:i+1])/len(self.D)
			integral = integrate.quad(self.m0, D[i], 1)
			beta_s += w0*integral[0]
			weights0.append(w0)
		# sanity check that these weights line up with figure 4
		print(weights0)
		return beta_s

	def beta_s1(self, s):
		beta_s = 0
		weights1 = []
		for i in range(len(self.D)):
			w1 = sum(s[i:])/len(self.D)
			weights1.append(w1)
			integral = integrate.quad(self.m1, 0, D[i])
			beta_s += w1*integral[0]
		# sanity check that these weights lineup with figure 4
		print(weights1)
		return beta_s

	def beta_tsls(self):
		s = self.s_tsls()
		beta_s0 = self.beta_s0(s)
		beta_s1 = self.beta_s1(s)		
		return beta_s0 + beta_s1

	def beta_iv(self):
		s = self.s_iv()
		print(s)
		print("hi")
		beta_s0 = self.beta_s0(s)
		beta_s1 = self.beta_s1(s)		
		return beta_s0 + beta_s1

	def beta_att(self):
		weights_att = self.w_att()
		yo = self.beta_s1(weights_att)
		print("yoo")
		print(yo)
		beta_s = 0
		for i in range(len(self.D)):
			w1 = weights_att[i]
			integral = integrate.quad(self.m1, 0, D[i])
			beta_s += w1*integral[0]
		print(weights_att)
		return beta_s

yo = IV_estimand(D,Z)
# print(yo.beta_iv())
# print(yo.beta_tsls())
# print(yo.beta_att())

# construct the gammas
class gamma:
	def __init__(self, D, Z, K):
		self.D = D
		self.Z = Z
		self.K = K

	# returns list of coefficients
	def bernstein(self):
		yo = []
		for k in range(self.K):
			for i in list(range(k, self.K+1)):
				yo.append((-1)**(i-k) * math.comb(self.K, i) * math.comb(i, k))
		print(yo)


hi = gamma(D, Z, 3)
hi.bernstein()
