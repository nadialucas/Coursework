# Nadia Lucas
# ECON 31720
# Problem Set 1, Problem 5
# Replicate table VI in the paper
# and perform propensity score matching estimation
# October 12, 2020
#
# I would like to thank George Vojta for helpful comments and also 
# Fiona Burlig's blog: https://www.fionaburlig.com/teaching/are212
# and Tom Roth's blog: https://tomroth.com.au/logistic/
# for the conceptual help needed to create this
# 
# running on Python version 3.8.6
################################################################################
import pandas as pd
import math
import numpy as np
import copy
from scipy.optimize import minimize, rosen, rosen_der
# change this directory accordingly to match personal machine
psdir = "/Users/nadialucas/Dropbox/Second year/ECON 31720/"

df = pd.read_stata(psdir + 
	"Persecution_Perpetuated_QJE_Replicate"+
	"/Dataset_QJE_Replicate_with_Cities.dta")

# clean the data to match the replication code
df['c25pop'] = df['c25pop'].astype(float)
df['c25juden'] = df['c25juden'].astype(float)
df['c25prot'] = df['c25prot'].astype(float)
df = df.assign(logpop25c = lambda x: np.log(x.c25pop))
df = df.assign(perc_JEW25 = lambda x: 100 * x.c25juden / x.c25pop)
df = df.assign(perc_PROT25 = lambda x: 100 * x.c25prot / x.c25pop)

# subset the data based on whether the village was around in 1349
exist1349 = np.where((df.judaica==1) | (df.comm1349==1), True, False)
cleaned_df = df[exist1349]

# trim the dataset to just the variables of interest and drop missing values
yvar = ['pog20s']
xvars =  ['pog1349', 'logpop25c', 'perc_JEW25', 'perc_PROT25']
xvars_geo = ['pog1349', 'Latitude', 'Longitude']
cluster_var = ['kreis_nr']
allvars = yvar + xvars + xvars_geo[1:] + cluster_var
cleaned_df = cleaned_df[allvars]
cleaned_df = cleaned_df.dropna()

# generate datasets for each specification
panel_a_vars = yvar + xvars + cluster_var
panel_a_df = cleaned_df[panel_a_vars]

panel_b_vars = yvar + xvars
panel_b_df = cleaned_df[panel_b_vars]

panel_c_vars = yvar + xvars_geo
panel_c_df = cleaned_df[panel_c_vars]

# add a constant for the propensity score calculations
pscore_b_df = copy.copy(panel_b_df)
pscore_b_df['C'] = 1
pscore_b_xvars = xvars + ['C']

pscore_c_df = copy.copy(panel_c_df)
pscore_c_df['C'] = 1
pscore_c_xvars = xvars_geo + ['C']


# Creates an object, OLS, that will automatically run a regression and
# gives options for what error you would like to return
class OLS:
	def __init__(self, yvar, xvars, df, constants = True):
		self.orig_df = df
		df_copy = copy.copy(df)
		xvars_copy = copy.copy(xvars)
		if constants == True:
			df_copy['C'] = 1
			xvars_copy.append('C')
		self.yvar = yvar[0]
		self.xvars = xvars_copy
		self.df = df_copy
		self.N = len(self.df)
		self.k = len(xvars_copy)
		self.Y = self.df[yvar]
		self.X = self.df[xvars_copy]
		# initialize the object [X'X]^-1. This is the "bread" of the sandwich
		# in any sandwich estimator and will be used in each error calculation
		self.XpXi = np.linalg.inv(np.dot(self.X.T, self.X))
		# solve for the coefficients
		self.theta = np.dot(self.XpXi, np.dot(self.X.T, self.Y))
		# calculate the errors
		predictedX = np.sum(np.multiply(self.theta.T, self.X), 1).to_frame()
		predictedX = predictedX.rename(columns = {0:yvar[0]})
		self.e = (self.Y - predictedX).to_numpy()

	def regress(self):
		return self.theta

	def std(self):
		# returns regular standard errors with no heteroskedasticity adjustment
		s2 = np.dot(self.e.T, self.e)/(self.N - self.k)
		sd = np.sqrt(np.multiply(s2, np.diag(self.XpXi)))
		return sd

	def robust(self):
		# create the error term for heteroskedasticity adjustment
		e2 = np.multiply(self.e, self.e)
		esigma = np.diagflat(e2)
		X = self.X.to_numpy()
		# create the "meat" 
		robust_sum = np.matmul(np.matmul(X.T, esigma), X)
		# construct the sandwich with the "bread"
		V_rob = np.matmul(np.matmul(self.XpXi, robust_sum), self.XpXi)
		sd = np.sqrt(np.diag(V_rob))
		return sd

	def cluster_robust(self, clustervars):
		# first add the error term to the dataset
		newdf = self.df
		predictedX = np.sum(np.multiply(self.theta.T, self.X), 1).to_frame()
		predictedX = predictedX.rename(columns = {0:"Yhat"})
		# add the residual to the dataframe
		df_withresid = newdf.assign(e = lambda x: x[self.yvar] - \
			predictedX.Yhat)
		df_withresid[clustervars] = self.orig_df[clustervars]
		# group by the cluster
		groups = df_withresid.groupby(clustervars)
		G = len(groups)
		robust_sum = 0
		# cycle through each cluster and create cluster-specific "meat"
		for key,item in groups:
			Xgroup = item[self.xvars].to_numpy()
			egroup = item['e'].to_numpy()
			egroup = egroup.reshape(len(egroup), 1)
			cluster_sum = np.matmul(np.matmul(np.matmul(Xgroup.T, egroup), \
				egroup.T), Xgroup)
			robust_sum+=cluster_sum
		# correct for degrees of freedom
		deg_freedom = (G/(G-1)) * ((self.N-1)/(self.N-self.k))
		# sandwich together with the bread defined in the class initialization
		V = deg_freedom * np.matmul(np.matmul(self.XpXi.T, robust_sum), \
			self.XpXi)
		return np.sqrt(np.diag(V))

# Object that initializes score calculations and gives the option for 
# nearest-neighbor calculations
class matching:
	def __init__(self, Y, X, df, yvar, xvars, weights):
		self.yvar = yvar[0]
		self.df = df
		self.N = len(self.df)
		self.k = len(xvars)
		self.Y = Y
		self.X = X
		# initialize the score which is weighting each X attribute
		# and then summing over them
		self.weighted_X = np.multiply(self.X, weights)
		self.score = np.sum(self.weighted_X, 1)
		self.df = self.df.assign(score = lambda x: self.score)

	def nearest_neighbors(self, k, treatmentvar):
		# takes in k which is the number of neighbors to match on and a binary
		# treatment variable to split the data on
		scores = self.df.score.to_numpy()
		treated = self.df[self.df[treatmentvar] == 1]
		treatedY = treated[treatmentvar].to_numpy()
		untreated = self.df[self.df[treatmentvar] == 0]
		untreatedY = untreated[treatmentvar].to_numpy()
		yhat0 = []
		# for each treated unit construct the counterfactual Y_hat(0)
		for i, row in treated.iterrows():
			score_i = row['score']
			distances = np.absolute(untreated.score - score_i)
			nsmallest = distances.nsmallest(k, keep = 'last')
			window = untreated.iloc[nsmallest, :]
			yhat0.append(window.mean(axis=0)[self.yvar])
		yhat1 = []
		# for each untreated unit construct the counterfactual Y_hat(1)
		for i, row in untreated.iterrows():
			score_i = row['score']
			distances = np.absolute(treated.score - score_i)
			nsmallest = distances.nsmallest(k, keep = 'last')
			window = treated.iloc[nsmallest, :]
			yhat1.append(window.mean(axis=0)[self.yvar])
		treated = treated.assign(yhat0 = yhat0)
		untreated = untreated.assign(yhat1 = yhat1)
		# the ATT comes from just the treated units and counterfactual
		att = (treated[self.yvar] - treated.yhat0).mean(axis = 0)
		# the ATE comes from all units and their counterfactuals
		ate = (treated[self.yvar] - treated.yhat0).mean(axis = 0) * \
		(len(treated[self.yvar])/len(scores)) + \
		(untreated.yhat1 - untreated[self.yvar]).mean(axis = 0) * \
		(len(untreated[self.yvar])/len(scores))

		return att, ate

# The logit class sets up an object for use in calculating weights based
# on logistic regression
class logit:
	def __init__(self, iterations, threshold, X, Y):
		self.iterations = iterations
		self.X = X
		self.Y = Y.reshape(len(Y), 1)
		self.threshold = threshold

	def sigmoid(self, z):
		return np.divide(np.exp(z), (1+np.exp(z)))

	def logistic_regression(self):
		# initialize weights
		weights = np.zeros((self.X.shape[1], 1))
		for iter in range(self.iterations):
			# calcualte predictions
			scores = np.matmul(self.X, weights)
			predictions = self.sigmoid(scores).reshape(len(scores), 1)
			# set up the scores and errors in a way such that the matrix
			# multiplication works out
			W = np.diag(np.diag(np.multiply(predictions, (1-predictions).T)))
			errors = (self.Y - predictions).reshape(len(predictions), 1)
			XWXinv = np.linalg.inv(np.dot(np.dot(self.X.T, W), self.X))
			# create the incremental change in weights from gradient descent
			delta = np.matmul(np.matmul(XWXinv, self.X.T), errors)
			weights = weights + delta
			diff = sum(delta**2)
			if diff < self.threshold:
				return weights

		return weights


# this function takes in a dataframe, reps, and variables of interest
# to bootstrap the standard errors from a matching estimator
def bootstrap(df, reps, yvar, xvars, k, run_logit = False):
	estimates = []
	# always samples/resamples the same number of observations in the dataframe
	M = len(df[yvar])
	for i in range(reps):
		# construct sample
		sample_i = df.sample(M, replace = True)
		sample_i = sample_i.reset_index(drop = True)
		X = sample_i[xvars].to_numpy()
		Y = sample_i[yvar].to_numpy()
		# run the appropriate matching estimator
		if run_logit:
			log = logit(100, 1e-10, X, Y)
			weights = log.logistic_regression().flatten()
		else:
			variance = X.var(0)
			weights = 1.0/variance
		# store the match
		match = matching(Y, X, sample_i, yvar, xvars, weights)
		estimate = match.nearest_neighbors(k, 'pog1349')
		estimates.append(estimate)
	# report mean and standard deviation of stored matches
	estimate_mean = np.mean(estimates)
	se = np.sqrt(np.var(estimates))
	return se

# the following code uses the cleaned dataframes and calls each appropriate 
# routine for replication and propensity score matching

# replicate panel A
panel_a = OLS(yvar, xvars, panel_a_df)
panel_a_estimates = panel_a.regress()
panel_a_errors = panel_a.cluster_robust(['kreis_nr'])
print("Panel A Estimates: ", panel_a_estimates)
print("Panel A Errors: ", panel_a_errors)

# replicate panel B
panel_b_Y = panel_b_df[yvar]
panel_b_X = panel_b_df[xvars]
# generate Mahalanobis weights
variance_b = panel_b_df[xvars].to_numpy().var(0)
panel_b_weights = 1.0/variance_b
panel_b = matching(panel_b_Y, panel_b_X, panel_b_df, yvar, \
	xvars, panel_b_weights)
panel_b_match = panel_b.nearest_neighbors(4, 'pog1349')
print("Panel B ATT Estimator: ", panel_b_match[0])

# replicate panel C
panel_c_Y = panel_c_df[yvar]
panel_c_X = panel_c_df[xvars_geo]
# generate Mahalnobis weights
variance_c = panel_c_df[xvars_geo].to_numpy().var(0)
panel_c_weights = 1.0/variance_c
panel_c = matching(panel_c_Y, panel_c_X, panel_c_df, yvar, xvars_geo, \
	panel_c_weights)
panel_c_estimate = panel_c.nearest_neighbors(2, 'pog1349')
print("Panel C ATT Estimator: ", panel_c_estimate[0])

# do propensity score matching
# first set up the data to input
pscore_b_Y = pscore_b_df[yvar].to_numpy()
pscore_b_X = pscore_b_df[pscore_b_xvars].to_numpy()
# set up the logit class and get the logit weights
pscore_b_log = logit(1000, 1e-10, pscore_b_X, pscore_b_Y)
pscore_b_weights = pscore_b_log.logistic_regression().flatten()
# perform nearest neighbors matching
pscore_b_match = matching(pscore_b_Y, pscore_b_X, pscore_b_df, yvar, \
	pscore_b_xvars, pscore_b_weights)
pscore_b_estimate, bate = pscore_b_match.nearest_neighbors(4, 'pog1349')
# pscore_b_errors = bootstrap(pscore_b_df, 20, yvar, pscore_b_xvars, 4, True)
print("ATT p-score panel B: ", pscore_b_estimate)
print("ATE p-score panel B: ",bate)
# print(pscore_b_errors)

# repeat for the geograhic matching
# first set up the data to input
pscore_c_Y = pscore_c_df[yvar].to_numpy()
pscore_c_X = pscore_c_df[pscore_c_xvars].to_numpy()
# set up the logit class and get the logit weights
pscore_c_log = logit(1000, 1e-10, pscore_c_X, pscore_c_Y)
pscore_c_weights = pscore_c_log.logistic_regression().flatten()
# perform nearest neighbors matching
pscore_c_match = matching(pscore_c_Y, pscore_c_X, pscore_c_df, yvar, \
	pscore_c_xvars, pscore_c_weights)
pscore_c_estimate, cate = pscore_c_match.nearest_neighbors(40, 'pog1349')
# pscore_c_errors = bootstrap(pscore_c_df, 20, yvar, pscore_c_xvars, 2, True)
print("ATT p-score panel C: ", pscore_c_estimate)
print("ATE p-score panel C: ", cate)
# print(pscore_c_errors)

