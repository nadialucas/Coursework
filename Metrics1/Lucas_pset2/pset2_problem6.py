# Nadia Lucas
# ECON 31720
# Problem Set 2, Problem 6
# 
# running on Python version 3.8.6
################################################################################
import pandas as pd
import math
import numpy as np
import copy
import scipy.stats

# for purposes of recreating the writeup
np.random.seed(1234)

# Creates an object, estimate, that will give options for what kind of 
# estimate to run 
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
		return self.theta

	def tsls(self):
		# for TSLS go through the standard matrix algebra and return estimate
		pi = self.pi
		PZZPinv = np.linalg.inv(np.dot(np.dot(pi.T, np.dot(self.Z.T, self.Z)), \
		 pi)) 
		beta = np.dot(np.dot(PZZPinv, pi.T), np.dot(self.Z.T, self.Y))
		return beta

	def jackknife(self):
		# for jackknife, following the Angrist notes closely for the analytical
		# way of doing it
		pi_hat = np.dot(self.pi.T, self.Z.T)
		h_1 = np.dot(self.Z, self.ZpZi)
		h = np.sum(np.multiply(h_1, self.Z), 1)
		h = h.reshape(self.N,1)
		Zpi = np.divide((pi_hat.T - np.multiply(h, self.X)), 1-h)
		beta = np.dot(np.linalg.inv(np.dot(Zpi.T, self.X)), np.dot(Zpi.T, \
			self.Y))
		return beta


def coverage(data, coverage=0.95):
	# helper function to get the 95% coverage 
	lo_quant = (1.0-coverage)/2
	hi_quant = coverage + (1.0 - coverage)/2
	lo = np.quantile(data, lo_quant)
	hi = np.quantile(data, hi_quant)
	return lo, hi


def monte_carlo (M, N):
	# initialize all 5 list of estimates and fill them in 
	OLS_estimates = np.empty(M)
	TSLS1_estimates = np.empty(M)
	JIVEs1 = np.empty(M)
	TSLS20_estimates = np.empty(M)
	JIVEs20 = np.empty(M)
	for i in range(M):
		# DGP for each run
		mean = [0, 0]
		cov = [[.25, .2],[.2, .25]]
		U, V = np.random.multivariate_normal(mean, cov, N).T
		U = U.reshape(len(U),1)
		V = V.reshape(len(V),1)
		Z = np.random.normal(0, 1, (N,1,20))
		Z1 = Z[:,:,0]
		X = .3*Z1 + V
		Y = X+U
		# add a constant
		ones = np.ones((N,1))
		X = np.append(X, ones, 1)
		# make Z easier to work with
		Z1 = Z1.reshape(N,1)
		Z = Z.reshape(N,20)
		Z1 = np.append(Z1, ones, 1)
		Z = np.append(Z, ones, 1)
		# initialize each regression object for one and many instruments
		regression1= estimate(X, Y, Z1)
		regression20 = estimate(X, Y, Z)
		# generate all 5 estimates 
		OLS_estimate = regression1.regress()[0]
		TSLS1_estimate = regression1.tsls()[0]
		JIVE1 = regression1.jackknife()[0]
		TSLS20_estimate = regression20.tsls()[0]
		JIVE20 = regression20.jackknife()[0]
		# and append the estimates accordingly
		OLS_estimates[i] = OLS_estimate
		TSLS1_estimates[i] = TSLS1_estimate
		JIVEs1[i] = JIVE1
		TSLS20_estimates[i] = TSLS20_estimate
		JIVEs20[i] = JIVE20
	# find the standard errors of each list of estimates
	OLSstd = np.std(OLS_estimates)
	TSLS1std = np.std(TSLS1_estimates)
	TSLS20std = np.std(TSLS20_estimates)
	JIVEs1std = np.std(JIVEs1)
	JIVEs20std = np.std(JIVEs20)
	# find the medians of each list of estimates
	OLSmed = np.median(OLS_estimates)
	TSLS1med = np.median(TSLS1_estimates)
	TSLS20med = np.median(TSLS20_estimates)
	JIVEs1med = np.median(JIVEs1)
	JIVEs20med = np.median(JIVEs20)
	# find the bias of each list of estimates
	OLSbias = np.mean(OLS_estimates - 1.0)
	TSLS1bias = np.mean(TSLS1_estimates - 1.0)
	TSLS20bias = np.mean(TSLS20_estimates - 1.0)
	JIVEs1bias = np.mean(JIVEs1 - 1.0)
	JIVEs20bias = np.mean(JIVEs20 - 1.0)
	# find the coverage rate of each list of estimates
	OLSlo, OLShi = coverage(OLS_estimates)
	TSLS1lo, TSLS1hi = coverage(TSLS1_estimates)
	TSLS20lo, TSLS20hi = coverage(TSLS20_estimates)
	JIVE1lo, JIVE1hi = coverage(JIVEs1)
	JIVE20lo, JIVE20hi = coverage(JIVEs20)
	# combine all estimates
	meds = [OLSmed, TSLS1med, TSLS20med, JIVEs1med, JIVEs20med]
	bias = [OLSbias, TSLS1bias, TSLS20bias, JIVEs1bias, JIVEs20bias]
	stds = [OLSstd, TSLS1std, TSLS20std, JIVEs1std, JIVEs20std]
	# I took coverage to be the actual space of coverage
	coverage_list = [OLShi - OLSlo, TSLS1hi-TSLS1lo, TSLS20hi-TSLS20lo, \
	JIVE1hi - JIVE1lo, JIVE20hi-JIVE20lo]
	# generate a dataframe of estimates to print out easily to latex
	titles = ["OLS", "IV (1)", "IV (20)", "Jackknife (1)", "Jackknife(20)"]
	data = {' ': titles, 'Median': meds, 'Bias': bias, 'Standard deviation': \
	stds, '95 coverage': coverage_list}
	df = pd.DataFrame.from_dict(data)
	print(df.to_latex(index = False))


# Run all specifications asked of in the problem
monte_carlo(1000, 100)
monte_carlo(1000, 200)
monte_carlo(1000, 400)
monte_carlo(1000, 800)





