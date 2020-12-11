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
import pandas as pd
import math
import copy
import matplotlib.pyplot as plt
from scipy.stats import randint
# for exact replication
np.random.seed(1)

################################################################################
# HELPER FUNCTIONS AND CLASSES
################################################################################

# an object that sets up and runs OLS and returns standard errors
class estimate:
	def __init__(self, Xvars, Yvar, df):
		self.df = copy.copy(df)
		self.yvar = Yvar
		self.xvars = Xvars
		self.N = len(df[Yvar])
		self.Y = df[Yvar]
		self.X = df[Xvars]
		self.k = self.X.shape[1]
		# initialize the object [X'X]^-1. This is the "bread" of the sandwich
		# in any sandwich estimator and will be used in each error calculation
		X = self.X.to_numpy()
		Y = self.Y.to_numpy()
		self.XpXi = np.linalg.inv(np.dot(X.T, X))
		# solve for the coefficients
		self.theta = np.dot(self.XpXi, np.dot(X.T, Y))
		self.theta = np.linalg.solve(np.dot(self.X.T, self.X), \
			np.dot(self.X.T, self.Y))
		# calculate the errors
		predictedX = np.sum(np.multiply(self.theta.T, self.X), 1).to_frame()
		predictedX = predictedX.rename(columns = {0:Yvar[0]})
		self.e = (self.Y - predictedX).to_numpy()

	def regress(self):
		# returns the regular OLS estimator
		return self.theta

	def std(self):
		# returns regular standard errors under homoskedasticity assumption
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
		newdf = copy.copy(self.df)
		predictedX = np.sum(np.multiply(self.theta.T, self.X), 1).to_frame()
		predictedX = predictedX.rename(columns = {0:"Yhat"})
		# add the residual to the dataframe
		df_withresid = newdf
		newdf['e']= self.e
		df_withresid[clustervars] = self.df[clustervars]
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

	def wild_bootstrap(self, beta_null, var):
		# beta_null is the null hypothesis for var
		X1 = self.df[var].to_numpy()
		# get a list of all variables that are not rel time 1 dummy
		Xvars_no1 = [v for v in self.xvars if v!=var[0]]
		# perform a regression without rel time 1 dummy
		Y1 = self.Y - beta_null*X1
		Xno1 = self.df[Xvars_no1].to_numpy()
		beta1 = np.linalg.solve(Xno1.T.dot(Xno1), Xno1.T.dot(Y1))
		# use the beta to construct the Us
		U = Y1 - np.dot(Xno1, beta1)
		# boostrap the Us
		rand_sign = 2*randint.rvs(0, 2, size = self.N).reshape(self.N, 1) - 1
		newU = np.multiply(U, rand_sign)
		# construct the wild Y
		Ywild = np.dot(Xno1, beta1) + X1 * beta_null + newU
		# get the new beta from the wild Y
		beta_wild = np.dot(self.XpXi, np.dot(self.X.T, Ywild))

		error = Ywild - np.dot(self.X, beta_wild)
		# and now the clustered-robust std error with similar procedure
		# as above
		clustervars = ['index']
		newdf = copy.copy(self.df)
		predictedX = np.sum(np.multiply(self.theta.T, self.X), 1).to_frame()
		predictedX = predictedX.rename(columns = {0:"Yhat"})
		# add the residual to the dataframe
		df_withresid = newdf
		newdf['e']= error

		# .assign(e = lambda x: x[self.yvar] - \
		# 	predictedX.Yhat)
		df_withresid[clustervars] = self.df[clustervars]
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

# the data generating process in a function to make it easily callable
# in all the monte carlo routines
def dgp(N, T, rho, theta, rlist, Nsmall = False):
	if Nsmall:
		# for small sized datasets, we want to evenly assign Es so as to avoid
		# collinearity issues
		E = np.array([ i%4 + 2 for i in range(N)]).reshape(N,1)
	else:
		E = randint.rvs(2, 6, size = N).reshape(N, 1)
	# the error term on the AR1
	epsilon = np.array([[np.random.normal() for i in range(T)] 
		for j in range(N)])
	# construct the AR1
	U = epsilon[:,0].reshape(N,1)
	for i in range(T-1):
		Unext = rho*U[:,i] + epsilon[:,i+1]
		U = np.insert(U, i+1, Unext, axis = 1)
	# construct the panel of time
	Tpanel = np.array([ [ t for t in range(T) ] for j in range(N)])
	# Serially independent standard normal
	V = np.array([ [np.random.normal() for i in range(T)]  for j in range(N) ])
	# Construct the potential outcomes
	Y0 = -.2 + .5*E + U
	Y1 = -.2 + .5*E + np.sin(Tpanel + 1 - theta*E) + U + V

	# construct Y first by constructing treatment in an event studies framework
	Epanel = np.repeat(E, T, axis = 1)
	D = np.array([ [1 if Epanel[i,t] <= t+1 else 0 for t in range(T)]
		for i in range(N) ]) 

	# Then construct using potential outcomes framework
	Y = np.multiply(D, Y1) + np.multiply(1-D, Y0)

	# now we essentially 'reshape long' and put it into a dataframe
	Ylong = Y.reshape(len(Y.flatten()), 1)

	# now construct all the dummies
	Elong = Epanel.reshape(len(Epanel.flatten()), 1)
	Tlong = Tpanel.reshape(len(Tpanel.flatten()), 1)
	Timelong = Tlong+1
	data = {'Y': Ylong.flatten(), 'T': Timelong.flatten(), 'E': Elong.flatten()}
	df = pd.DataFrame(data  = data)
	df['C'] = 1
	# use list comprehensions to get all the dummies and add them to dataframe
	cohortDums = np.array([ [1 if Elong[i] == t+2 else 0 for t in range(T-1)] 
		for i in range(len(Elong)) ])
	df['E2'] = cohortDums[:,0]
	df['E3'] = cohortDums[:,1]
	df['E4'] = cohortDums[:,2]
	df['E5'] = cohortDums[:,3]
	timeDums = np.array([ [1 if Tlong[i] == t else 0 for t in range(T)] 
		for i in range(len(Tlong)) ]) 
	df['t1'] = timeDums[:,0]
	df['t2'] = timeDums[:,1]
	df['t3'] = timeDums[:,2]
	df['t4'] = timeDums[:,3]
	df['t5'] = timeDums[:,4]
	relDums = np.array([ [1 if Tlong[i]+1 - Elong[i] == r else 0 for r in rlist] 
		for i in range(len(Tlong)) ])
	df['rt-3'] = relDums[:,0]
	df['rt-2'] = relDums[:,1]
	df['rt0'] = relDums[:,2]
	df['rt1'] = relDums[:,3]
	df['rt2'] = relDums[:,4]
	df['rt3'] = relDums[:,5]

	# add this in as the cluster variable
	df['index'] = list(range(N*T))
	return df

def monte_carlo(N, M, theta, rho = 0.5):
	# for part b and c we want to run a monte carlo and get a distribution of
	# the coefficients on the relative time dummies
	rel_betas = []
	T=5
	rlist = [-3, -2, 0, 1, 2, 3]
	for m in range(M):
		df = dgp(N, T, rho, theta, rlist)
		# run the regression on a constant with 
		# 1 cohort dummy dropped and 1 time dummy dropped
		Xvars = ['C', 'E2', 'E3', 'E4', 't1', 't2', 't3', 't4', 'rt-3', 'rt-2', 
		'rt0', 'rt1', 'rt2', 'rt3']
		Yvar = ['Y']
		ols_beta = estimate(Xvars, Yvar, df).regress()
		rel_betas.append(ols_beta[8:])
	# tease out and return the parameters of interest to plot
	betas = np.array(rel_betas)
	mean = np.squeeze(np.mean(betas, axis = 0))
	low_quantile = np.squeeze(np.quantile(betas, .025, axis = 0))
	hi_quantile = np.squeeze(np.quantile(betas, .975, axis=0))
	return low_quantile, mean, hi_quantile

################################################################################
# PART B and C
################################################################################

# thetas we want to plot
thetalist = [-2, 0, 1]
for theta in thetalist:
	# plot a monte carlo for both N = 1000 and N = 10,000
	lo1, mean1, hi1 = monte_carlo(1000, 50, theta)
	lo2, mean2, hi2 = monte_carlo(10000, 50, theta)

	rlist = [-3, -2, 0, 1, 2, 3]
	# now plot
	fig, ax = plt.subplots()
	# all on the same axis
	ax.plot(rlist, mean1, '-o', markersize = 2.5, linewidth = 1, 
			color = 'coral', 
			label = 'N=1000 with 95'+'%'+' confidence interval reported')
	ax.plot(rlist, lo1, alpha = 0)
	ax.plot(rlist, hi1, alpha = 0)
	ax.fill_between(rlist, lo1, hi1, color = 'coral', alpha = 0.3)

	ax.plot(rlist, mean2, '-o', markersize = 2.5, linewidth = 1, 
			color = 'dodgerblue', 
			label = 'N=10000 with 95'+'%'+' confidence interval reported')
	ax.plot(rlist, lo2, alpha = 0)
	ax.plot(rlist, hi2, alpha = 0)
	ax.fill_between(rlist, lo2, hi2, color = 'dodgerblue', alpha = 0.3)
	if theta == 1:
		ax.legend(loc='upper left', fontsize = 9)
	else:
		ax.legend(loc='lower left', fontsize = 9)
	ax.set(xlabel='Relative time', ylabel='Coefficient')

	fig.savefig("partb_theta"+str(theta)+".png")

################################################################################
# PART D
################################################################################

# for part d we calculate the ATE at time 3 for cohort 2
# see writeup for analytical counterpart of each variable
def ATE2(N, theta, rho=0.5):
	T=5
	rlist = [-3, -2, 0, 1, 2, 3]
	df = dgp(N, T, rho, theta, rlist)
	# construct the trend from T=1 to T=2
	trend1 = df.loc[(df['T'] == 2) & (df['E'] >= 3), 'Y'].mean() - \
	  df.loc[(df['T'] == 1) & (df['E'] >= 3), 'Y'].mean()
	 # construct the trend from T=2 to T=3
	trend2 = df.loc[(df['T'] == 3) & (df['E'] >= 4), 'Y'].mean() - \
	  df.loc[(df['T'] == 2) & (df['E'] >= 4), 'Y'].mean()
	# solve for the ATE using the trends
	ATE2 = df.loc[(df['T'] == 3) & (df['E'] == 2), 'Y'].mean() - \
	  df.loc[(df['T'] == 1) & (df['E'] == 2), 'Y'].mean() + trend1 + trend2
	return ATE2

# returns parameters of interest for problem 2 part d
for theta in thetalist:
	print("ATE2 for theta = ", theta)
	print(ATE2(10000, theta))

################################################################################
# PART E
################################################################################

# the workhorse function for part e
def get_errors(N, M, rho, theta=1, 
	true_beta = (math.sin(1)-math.sin(-1)-math.sin(-4))):
	rel_betas = []
	T=5
	rlist = [-3, -2, 0, 1, 2, 3]
	# initialize storing all standard errors
	stdlist = []
	robustlist = []
	clustlist = []
	bootstraplist = []
	wildlist = []

	for m in range(M):
		df = dgp(N, T, rho, theta, rlist, Nsmall = True)
		# run the regression on a constant with 
		# 1 cohort dummy dropped and 1 time dummy dropped
		Xvars = ['C', 'E2', 'E3', 'E4', 't1', 't2', 't3', 't4', 'rt-3', 'rt-2', 
		'rt0', 'rt1', 'rt2', 'rt3']
		Yvar = ['Y']
		# initialize an object of the estimate class
		est = estimate(Xvars, Yvar, df)
		# get estimator and all corresponding errors
		betas = est.regress()
		std_errors = est.std().flatten()
		robust_errors = est.robust()
		clust_robust_errors = est.cluster_robust(['index'])
		wild_errors = est.wild_bootstrap(true_beta, ['rt1'])
		# coefficient on rt1 is the 12th 
		beta = betas[11]
		std = std_errors[11]
		rob = robust_errors[11]
		clus = clust_robust_errors[11]
		wild = wild_errors[11]
		# find out if this simulation would have rejected the null
		# for each error structure
		if np.abs((beta-true_beta)/std) > 1.96:
			stdlist.append(1)
		else:
			stdlist.append(0)
		if np.abs((beta-true_beta)/rob) > 1.96:
			robustlist.append(1)
		else:
			robustlist.append(0)
		if np.abs((beta-true_beta)/clus) > 1.96:
			clustlist.append(1)
		else:
			clustlist.append(0)
		if np.abs((beta-true_beta)/wild) > 1.96:
			wildlist.append(1)
		else:
			wildlist.append(0)

	return np.mean(np.array(stdlist)), np.mean(np.array(robustlist)), \
	np.mean(np.array(clustlist)), np.mean(np.array(wildlist))


rholist = [0, .5, 1]
# this procedure takes a while, feel free to decrease number of runs
# in the second argument
for rho in rholist:
	print("rho is "+str(rho))
	print('Rejection for N=20: ', get_errors(20, 200, rho))
	print('Rejection for N=50: ', get_errors(50, 200, rho))
	print('Rejection for N=200: ', get_errors(200, 200, rho))

