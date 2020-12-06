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

# for exact replication
np.random.seed(1234)

# an object that sets up and runs OLS and TSLS
class estimate:
	def __init__(self, X,Y):
		self.N = len(Y)
		self.Y = Y
		self.X = X
		self.k = X.shape[1]
		# initialize the object [X'X]^-1. This is the "bread" of the sandwich
		# in any sandwich estimator and will be used in each error calculation
		self.XpXi = np.linalg.inv(np.dot(self.X.T, self.X))
		# solve for the coefficients
		self.theta = np.dot(self.XpXi, np.dot(self.X.T, self.Y))

	def regress(self):
		# returns the regular OLS estimator
		return self.theta


def monte_carlo(N, M, theta, rho = 0.5):
	rel_betas = []
	T=5
	rho = .5
	rlist = [-3, -2, 0, 1, 2, 3]
	for m in range(M):
		# Construct the DGP
		E = randint.rvs(2, 6, size = N).reshape(N, 1)


		epsilon = np.array([ [np.random.normal() for i in range(T)]    for j in range(N) ])
		# construct the AR1
		U = epsilon[:,0].reshape(N,1)
		for i in range(T-1):
			Unext = rho*U[:,i] + epsilon[:,i+1]
			U = np.insert(U, i+1, Unext, axis = 1)
			
		Tpanel = np.array([ [ t for t in range(T) ] for j in range(N)])

		V = np.array([ [np.random.normal() for i in range(T)]    for j in range(N) ])


		Y0 = -.2 + .5*E + U
		Y1 = -.2 + .5*E + np.sin(Tpanel - theta*E) + U + V


		# construct Y first by constructing treatment
		Epanel = np.repeat(E, T, axis = 1)
		D = np.array([ [1 if Epanel[i,t] <= t+1 else 0 for t in range(T)] for i in range(N) ]) 

		# Then construct using potential outcomes framework
		Y = np.multiply(D, Y1) + np.multiply(1-D, Y0)

		# now we essentially 'reshape long' and stack everything
		Ylong = Y.reshape(len(Y.flatten()), 1)

		# now construct all the dummies
		Elong = Epanel.reshape(len(Epanel.flatten()), 1)
		Tlong = Tpanel.reshape(len(Tpanel.flatten()), 1)

		cohortDums = np.array([ [1 if Elong[i] == t+2 else 0 for t in range(T-1)] for i in range(len(Elong)) ]) 
		timeDums = np.array([ [1 if Tlong[i] == t else 0 for t in range(T)] for i in range(len(Tlong)) ]) 
		
		relDums = np.array( [ [1 if Tlong[i]+1 - Elong[i] == r else 0 for r in rlist] for i in range(len(Tlong)) ])

		# cohort fixed effects, time fixed effects, and relative time dummies are regressors
		X = np.hstack([cohortDums, timeDums, relDums])

		ols_beta = estimate(X, Ylong).regress()
		rel_betas.append(ols_beta[9:])

	betas = np.array(rel_betas)
	mean = np.squeeze(np.mean(betas, axis = 0))
	low_quantile = np.squeeze(np.quantile(betas, .025, axis = 0))
	hi_quantile = np.squeeze(np.quantile(betas, .975, axis=0))
	return low_quantile, mean, hi_quantile



thetalist = [-2, 0, 1]

for theta in thetalist:
	# do we have to impute the 
	lo1, mean1, hi1 = monte_carlo(1000, 60, theta)

	lo2, mean2, hi2 = monte_carlo(10000, 60, theta)

	rlist = [-3, -2, 0, 1, 2, 3]

	# time to plot
	fig, ax = plt.subplots()

	ax.plot(rlist, mean1, '-o', markersize = 2.5, linewidth = 1, 
			color = 'coral', label = 'N=1000 with 2.5'+'%'+' and 97.5'+'%'+' quantiles reported')
	ax.plot(rlist, lo1, alpha = 0)
	ax.plot(rlist, hi1, alpha = 0)
	ax.fill_between(rlist, lo1, hi1, color = 'coral', alpha = 0.3)

	ax.plot(rlist, mean2, '-o', markersize = 2.5, linewidth = 1, 
			color = 'dodgerblue', label = 'N=10000 with 2.5'+'%'+' and 97.5'+'%'+' quantiles reported')
	ax.plot(rlist, lo2, alpha = 0)
	ax.plot(rlist, hi2, alpha = 0)
	ax.fill_between(rlist, lo2, hi2, color = 'dodgerblue', alpha = 0.3)
	ax.legend(loc='upper left', fontsize = 8)
	ax.set(xlabel='Relative time', ylabel='Coefficient', title = 'Theta: '+str(theta))

	fig.savefig("partb_theta"+str(theta)+".png")
