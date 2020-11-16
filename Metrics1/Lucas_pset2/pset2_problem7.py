# Nadia Lucas
# ECON 31720
# Problem Set 2, Problem 7
# 
# Thanks to George Vojta for helpful comments
#
# running on Python version 3.8.6
################################################################################
from __future__ import division
import pandas as pd
import math
import numpy as np
import copy
import scipy.stats
import random

# for purposes of recreating exactly the values in my writeup
np.random.seed(1234)
path = "/Users/nadialucas/Dropbox/Second year/ECON 31720/ps2_files/"
df = pd.read_csv(path+"abadie.csv")
# Creates an object, estimate, that gives options for what type of estimate
# to run
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

	def weighted_ols(self, weights):
		# returns the propensity score weighted OLS estimator
		# corresponding to least squares treated
		XwXi = np.linalg.inv(np.dot(np.multiply(self.X, weights).T, self.X))
		return np.dot(XwXi, np.dot(np.multiply(self.X, weights).T, self.Y))

	def tsls(self):
		# 2SLS estimator - following the logic of the classic sandwich estimator
		pi = self.pi
		PZZPinv = np.linalg.inv(np.dot(np.dot(pi.T, np.dot(self.Z.T, self.Z)),\
		 pi)) 
		beta = np.dot(np.dot(PZZPinv, pi.T), np.dot(self.Z.T, self.Y))
		return beta

	def jackknife(self):
		# jackknife estimator
		pi_hat = np.dot(self.pi.T, self.Z.T)
		h_1 = np.dot(self.Z, self.ZpZi)
		h = np.sum(np.multiply(h_1, self.Z), 1)
		h = h.reshape(self.N,1)
		Zpi = np.divide((pi_hat.T - np.multiply(h, self.X)), 1-h)
		beta = np.dot(np.linalg.inv(np.dot(Zpi.T, self.X)), np.dot(Zpi.T, \
			self.Y))
		return beta

class errors:
	def __init__(self, df, beta, controls, yvar, exog, endog, clustervars = [] \
		, constant = True):
		# This initialization looks quite messy but it gives a structure to 
		# every possible type of data you might need for calculating errors
		# as long as the fields are filled out
		newdf = df.copy()
		allvars = controls + yvar + exog + endog + clustervars
		xvars = endog + controls 
		if constant == False:
			allvars = ['C'] + controls + yvar + exog + endog + clustervars
			xvars = endog + ['C'] + controls
		zvars = exog + controls
		self.df = newdf[allvars]
		self.beta = beta
		self.X = self.df[xvars].to_numpy()
		self.k = len(xvars)
		self.Y = self.df[yvar].to_numpy()
		self.Z = self.df[zvars].to_numpy()
		self.N = len(self.Y)
		self.D = self.df[endog].to_numpy()
		self.Z1 = self.df[exog].to_numpy()
		self.Xcontrols = self.df[controls].to_numpy()
		self.xvars = xvars
		self.yvar = yvar
		self.zvars = zvars
		self.exog = exog
		self.endog = endog
		self.controls = controls
		self.clustervars = clustervars
		# Do some pre-calculations to help us in multiple helper functions
		self.df.loc[:,'Yhat'] = np.sum(np.multiply(self.beta.T, self.X), 1)
		self.df.loc[:,'e'] = self.df[self.yvar[0]] - self.df['Yhat']
		self.XpXi = np.linalg.inv(np.dot(self.X.T, self.X))
		self.ZpZi = np.linalg.inv(np.dot(self.Z.T, self.Z))
		self.e = self.df['e'].to_numpy()

	def std(self):
		# returns regular standard errors with no heteroskedasticity adjustment
		s2 = np.dot(self.e.T, self.e)/(self.N - self.k)
		sd = np.sqrt(np.multiply(s2, np.diag(self.XpXi)))
		return sd

	def anderson_rubin(self):
		# Anderson-Rubin confidence interval
		beta_exo = self.beta[1:].reshape((len(self.beta)-1, 1))
		# the beta of interest
		beta_endo = self.beta[0]
		# generate a Y-variable 
		Y = self.Y.reshape((len(self.Y), 1))
		Y_tilde = Y - np.dot(beta_exo.T, self.Xcontrols.T).T
		# don't necessarily have to iterate through all possible bounds, just 
		# a fine number of betas around the bounds (using trial and error)
		beta_range_low = np.arange(6.4,6.5,.001)
		beta_range_hi = np.arange(12.4, 12.5, .001)
		beta_range = np.concatenate((beta_range_low, beta_range_hi), axis = 0)
		# initialize list of t-stats
		ts = np.zeros(len(beta_range))
		# iterate through all possible betas and find the t-stat
		for b in range(len(beta_range)):
			newdf = self.df.copy()
			# create Y - X(beta) and regress on Z
			U = Y_tilde - np.multiply(beta_range[b], self.D)
			newdf.loc[:,'U'] = U
			beta_to_test = estimate(self.Z1, U, self.Z1).regress()
			beta_error = errors(newdf, beta_range[b], self.exog, ['U'] , [], [])
			# get the variance and construct the t-stat
			omega = beta_error.robust()**2
			ts[b] = (beta_to_test*beta_to_test)/omega
		# the 95% chi-squared t-stat cutoff 
		filtered = ts < 3.481
		filtered_betas = filtered*beta_range
		# get the CI from the betas that are left
		betas = [i for i in filtered_betas if i!=0]
		low = np.around(min(betas), 3)
		high = np.around(max(betas), 3)
		return low, high

	def robust(self):
		# create the error term for heteroskedasticity adjustment
		e2 = np.multiply(self.e, self.e)
		esigma = np.diagflat(e2)
		X = self.X
		# create the "meat" 
		robust_sum = np.matmul(np.matmul(X.T, esigma), X)
		# construct the sandwich with the "bread"
		V_rob = np.matmul(np.matmul(self.XpXi, robust_sum), self.XpXi)
		sd = np.sqrt(np.diag(V_rob))
		return sd

	def bootstrap(self, M, fctn):
		# returns the standard errors from running either least squares
		# treated or jackknife M times
		estimates = []
		if fctn == "LST":
			# for Least Squares Treated
			for i in range(M):
				indices = random.choices(list(range(self.N)), k = self.N)	
				# reminder for logit, no constant
				newX = self.Xcontrols[indices, :]
				newY = self.Y[indices]
				newD = self.D[indices, :]
				newZ = self.Z1[indices, :]
				log = logit(newX, newZ)
				fitted_vals = log.descent()
				pred_probs = log.pred_values(False)
				# go through the same way as below in getting LST estimates
				# straight from the Abadie paper
				DxnotZ = np.multiply(newD, 1-newZ)
				notDxZ = np.multiply(1-newD, newZ)
				Z1 = pred_probs.reshape(len(newY), 1)
				Z0 = (1-pred_probs).reshape(len(newY), 1)

				X_ols = self.X[indices, :]

				kappa = 1 - np.divide(DxnotZ, Z0) - np.divide(notDxZ, Z1)
				kappa = kappa.reshape((len(kappa),1)) 
				weightedX = np.multiply(kappa, X_ols)
				newZ = newZ.reshape(len(newZ),1)
				yo = estimate(X_ols, newY, newZ)
				beta = yo.weighted_ols(kappa)
				estimates.append(beta)
		elif fctn == "JK":
			# for Jackknife
			for i in range(M):
				indices = random.choices(list(range(self.N)), k = self.N)
				# reminder for logit, no constant
				newX = self.X[indices, :]
				newY = self.Y[indices]
				newZ = self.Z[indices, :]
				JIVE_obj = estimate(newX, newY, newZ)
				beta = JIVE_obj.jackknife()
				estimates.append(beta)
		return np.std(estimates, axis = 0)


class logit:
	# a class to find propensity scores used in least squares treated
	def __init__(self, X, Z):
		self.X = X
		self.Z = Z.reshape(len(Z), 1)
		# initialize weights here
		self.weights = np.zeros((X.shape[1], 1))
		std_dev = np.std(self.X, axis = 0).reshape((len(self.weights), 1))
		# working with no constant since it doesn't make sense for the standard
		# deviation correction so we de-mean the Xs
		self.X_demeaned = np.divide((self.X - np.mean(self.X, axis=0)), \
			std_dev.T)
	def logistic(self):
		return 1.0/(1.0 + np.exp(np.dot(-self.X_demeaned, self.weights))).T[0]
	def gradient(self):
		# for gradient descent
		inner = self.logistic() - np.squeeze(Z)
		grad = np.dot(inner.T, self.X_demeaned)
		return grad.reshape((len(grad),1))
	def cost(self):
		yo = self.logistic()
		cost1 = np.dot(self.Z.T, np.log(yo))
		cost2 = np.dot((1-self.Z.T), np.log(1-yo))
		return np.mean(-1.0*(cost1+cost2))
	def descent(self, lr = .001, converge_change = .001):
		# the 
		cost = self.cost()
		delta_cost = 1
		i = 1
		while delta_cost > converge_change:
			old_cost = cost
			self.weights = self.weights - (lr * self.gradient())
			cost = self.cost()
			delta_cost = old_cost - cost
			i+=1
		return self.weights
	def pred_values(self, hard = True):
		pred_prob = self.logistic()
		pred_val = np.where(pred_prob >= .5, 1, 0)
		if hard:
			return pred_val
		return pred_prob

# Start by cleaning the dataset
df['C'] = 1
df = df.assign(age_minus25 = lambda x: x.age - 25)
df = df.assign(age_minus25_sqrd = lambda x: x.age_minus25**2)
Y = df['nettfa'].to_numpy()

# Create all necessary objects and run what is asked in the problem
# using all the above helper classes
X_ols = df[['p401k', 'C', 'inc', 'age_minus25', 'age_minus25_sqrd', 'marr', \
'fsize']].to_numpy()
Z = df[['e401k', 'C', 'inc', 'age_minus25', 'age_minus25_sqrd', 'marr', \
'fsize']].to_numpy()

# For OLS we initiate the estimator object and errors and return a list
# of coefficients and standard errors
ols = estimate(X_ols,Y, Z)
beta_ols = ols.regress()
yvar = ['nettfa']
controls = ['C', 'inc', 'age_minus25', 'age_minus25_sqrd', 'marr', 'fsize']
exog = ['e401k']
endog = ['p401k']
error_ols = errors(df, beta_ols, controls, yvar, exog, endog)
# this works!
print("OLS coefficients: ", beta_ols)
print("OLS errors: ", error_ols.robust())

# First stage is similar, we jsut need to change around the Xs and independent
# paramters of interest to match
X_fs = df[['C', 'inc', 'age_minus25', 'age_minus25_sqrd', 'marr', 'fsize', \
'e401k']].to_numpy()
Y_fs = df['p401k'].to_numpy()
first_stage = estimate(X_fs, Y_fs, Z)
gamma_hat = first_stage.regress()
yvar = ['p401k']
controls = ['C', 'inc', 'age_minus25', 'age_minus25_sqrd', 'marr', 'fsize', \
'e401k']
exog = []
endog = []
error_fs = errors(df, gamma_hat, controls, yvar, exog, endog)
print("First stage coefficients", gamma_hat)
print("First stage errors: ", error_fs.robust())

# Second stage we also set it up similarly, but make sure to be careful about
# what is defined as exogenous or endogenous
X_ss = df[['p401k', 'C', 'inc', 'age_minus25', 'age_minus25_sqrd', 'marr', \
'fsize']].to_numpy()
Y_ss = df['nettfa'].to_numpy()
second_stage = estimate(X_ss, Y_ss, Z)
beta_tsls = second_stage.tsls()
controls = ['C', 'inc', 'age_minus25', 'age_minus25_sqrd', 'marr', 'fsize']
exog = ['e401k']
endog = ['p401k']
yvar = ['nettfa']
error_iv = errors(df, beta_tsls, controls, yvar, exog, endog)
print("Second stage coefficients: ", beta_tsls)
print("Second stage errors: ", error_iv.robust())
# call the Anderson-Rubin CI test while we are here
print("95 AR CI: ", error_iv.anderson_rubin())

# There's a little extra work to be done for Least Squares Treated
# First we initialize the propensity scores
D = df['p401k'].to_numpy()
Z = df['e401k'].to_numpy()
X = df[['inc', 'age_minus25', 'age_minus25_sqrd', 'marr', 'fsize']].to_numpy()
log = logit(X, Z)
fitted_vals = log.descent()
pred_probs = log.pred_values(False)
# Then plug the p-scores into the equation from Abadie's paper
DxnotZ = np.multiply(D, 1-Z)
notDxZ = np.multiply(1-D, Z)
Z1 = pred_probs
Z0 = 1-pred_probs
X_exo = df[['C', 'inc', 'age_minus25', 'age_minus25_sqrd', 'marr', 'fsize']]
kappa = 1 - np.divide(DxnotZ, Z0) - np.divide(notDxZ, Z1)
kappa = kappa.reshape((len(kappa),1)) 
weightedX = np.multiply(kappa, X_ols)
Z = Z.reshape(len(Z),1)
least_squares_treated = estimate(X_ols, Y, Z)
# Run the weighted OLS using kappa as weights
beta = least_squares_treated.weighted_ols(kappa)
controls = ['inc', 'age_minus25', 'age_minus25_sqrd', 'marr', 'fsize']
error_lst = errors(df, beta, controls, yvar, exog, endog, constant = False)
D = D.reshape((len(D), 1))
print("Least squares treated coefficients: ", beta)
# and bootstrap standard errors
print("Least squares treated errors: ", error_lst.bootstrap(50, "LST"))

# Finally run the jackknife
Z = df[['e401k', 'C', 'inc', 'age_minus25', 'age_minus25_sqrd', 'marr', \
'fsize']].to_numpy()
JIVEs = estimate(X_ols, Y, Z)
JIVE_beta = JIVEs.jackknife()
print("Jackknife coefficient: ", JIVE_beta)
controls = ['C', 'inc', 'age_minus25', 'age_minus25_sqrd', 'marr', 'fsize']
error_JIVE = errors(df, JIVE_beta, controls, yvar, exog, endog)
# And bootstrap standard errors
print("Jackknife errors: ", error_JIVE.bootstrap(50, "JK"))


