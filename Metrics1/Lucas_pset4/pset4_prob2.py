# Nadia Lucas
# ECON 31720
# Problem Set 4, Question 2
# December 12, 2020
# 
# I would like to thank Yixin Sun and George Vojta for helpful comments
# 
# running on Python version 3.8.6
################################################################################


import numpy as np 
import math
import matplotlib.pyplot as plt
from scipy.stats import randint


# Construct the DGP
N = 1000
E = randint.rvs(2, 6, size = N).reshape(N, 1)
T=5
rho = .5
theta = -2


epsilon = np.array([ [np.random.normal() for i in range(T)]    for j in range(N) ])
U = epsilon[:,0].reshape(N,1)
for i in range(T-1):
	Unext = rho*U[:,i] + epsilon[:,i+1]
	U = np.insert(U, i+1, Unext, axis = 1)
	
time_matrix = np.array([ [ t for t in range(T) ] for j in range(N)])

V = np.array([ [np.random.normal() for i in range(T)]    for j in range(N) ])


Y0 = -.2 + .5*E + U
Y1 = -.2 + .5*E + np.sin(time_matrix - theta*E) + U + V


# an object that sets up and runs OLS and TSLS
class estimate:
	def __init__(self, X,Y,Z):
		self.N = len(Y)
		self.Z = Z
		self.Y = Y
		self.X = X
		self.k = X.shape[1]
		# initialize the object [X'X]^-1. This is the "bread" of the sandwich
		# in any sandwich estimator and will be used in each error calculation
		self.XpXi = np.linalg.inv(np.dot(self.X.T, self.X))
		self.ZpZi = np.linalg.inv(np.dot(self.Z.T, self.Z))
		# solve for the coefficients
		self.theta = np.dot(self.XpXi, np.dot(self.X.T, self.Y))
		self.pi = np.dot(self.ZpZi, np.dot(self.Z.T, self.X))

	def regress(self):
		# returns the regular OLS estimator
		return self.theta

	def tsls(self):
		# 2SLS estimator - following the logic of the classic sandwich estimator
		pi = self.pi
		PZZPinv = np.linalg.inv(np.dot(np.dot(pi.T, np.dot(self.Z.T, self.Z)),\
		 pi)) 
		beta = np.dot(np.dot(PZZPinv, pi.T), np.dot(self.Z.T, self.Y))
		return beta

# construct Y
Epanel = [E for i in range(T)]
print(Epanel.shape)

# construct a very saturated regression


