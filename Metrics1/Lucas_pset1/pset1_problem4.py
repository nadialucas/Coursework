# Nadia Lucas
# ECON 31720
# Problem Set 1, Question 4
# Monte Carlo simulations to estimate E[Y|X=x]
# October 12, 2020
# 
# I would like to thank Yixin Sun and George Vojta for helpful comments
# 
# running on Python version 3.8.6
################################################################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import math

# for purposes of debugging (comment this out if you'd like)
np.random.seed(1234)
# First set up the DGP
N = 1000
X_min, X_max = -2, 2
U_mean, U_sd = 0, 0.3

X = np.random.uniform(X_min, X_max, N)
U = np.random.normal(U_mean, U_sd, N)
Y = np.sin(2*X) + 2*np.exp(-16*(np.power(X, 2))) + U

# From the data, construct a dataframe 
d = {'Y': Y, 'X': X}
df = pd.DataFrame(data = d)


plot_x = np.arange(-2, 2, .01)
# create a function for calculating the kernel regression
# arguments: df: dataframe with our generated data
#         h: bandwidth (tuning) parameter
# returns: non-parametric regression to plot
def kernel(df, h, plot_x):
	# arguments: df: pandas dataframe, h: bandwidth
	# returns: list of predicted data to plot
	df_copy = copy.copy(df)
	plot_y = np.zeros(len(plot_x))
	for i in range(len(plot_x)):
		# create a sliding window and calculate the mean
		window_parent = df_copy.loc[df_copy['X']>=plot_x[i]-h]
		window = window_parent.loc[window_parent['X']<=plot_x[i]+h]
		plot_y[i] = window.mean(axis=0).Y
	return plot_y

def local_linear(df, h, plot_x):
	# arguments: df: pandas dataframe, h: bandwidth
	# returns: list of predicted data to plot
	df_copy = copy.copy(df)
	plot_y = np.zeros(len(plot_x))
	xvars = ['X']
	yvars = ['Y']
	for i in range(len(plot_x)):
		# create a sliding window to regress on
		window_parent = df_copy.loc[df_copy['X']>=(plot_x[i]-h)]
		window = window_parent.loc[window_parent['X']<=(plot_x[i]+h)]
		# now run a regression within that window
		theta = regress(yvars, xvars, window)
		plot_y[i] = theta[[1]] + theta[[0]]*plot_x[i]
	return plot_y

def sieve(df, K, plot_x):
	# arguments: df: pandas dataframe, K: maximum power
	# returns: list of predicted data to plot
	plot_y = np.zeros(len(plot_x))
	df_copy = copy.copy(df)
	xvars = ['X']
	yvars = ['Y']
	for i in range(K-1):
		# construct each new X variable
		Xvar = 'X' + str(i+2)
		df_copy[Xvar] = np.power(df_copy['X'], i+2)
		xvars.append(Xvar)
	theta = regress(yvars, xvars, df_copy)
	hi = np.power(plot_x, 3) * 8
	for j in range(K-1):
		plot_y += theta[[j]][0]*np.power(plot_x, j+1)
	plot_y += theta[[K]][0]
	return plot_y

def nearest_neighbors(df, k, plot_x):
	# arguments: df: pandas dataframe, k: number of neighbors
	# returns: list of predicted data to plot
	df_copy = copy.copy(df)
	plot_y = np.zeros(len(plot_x))
	for i in range(len(plot_x)):
		# find the nearest neighbors and create the mean
		distances = np.absolute(df_copy.X - plot_x[i])
		idx = np.argpartition(distances, k)
		neighbs = idx[:k-1]
		window = df_copy.iloc[neighbs, :]
		plot_y[i] = window.mean(axis=0).Y
	return plot_y

def sieve_bernstein(df, K, plot_x):
	# arguments: df: pandas dataframe, K: max power for polynomial
	# returns: list of predicted data to plot
	plot_y = np.zeros(len(plot_x))
	# first generate the appropriate basis
	df_copy = copy.copy(df)
	bvars = []
	yvars = ['Y']
	# translate into a [0,1] basis
	df_copy['z'] = np.true_divide(np.add(df_copy['X'], 2), 4)
	# then construct each Bernstein polynomial covariate
	for i in range(K+1):
		coef = math.comb(K,i)
		bvar = 'b' + str(i)
		bvars.append(bvar)
		df_copy[bvar] = coef*np.power(df_copy['z'], i)*\
		np.power((1-df_copy['z']), K-i)
	# perform the regression
	theta = regress(yvars, bvars, df_copy, constants = False)
	# transform back into the correct basis to calculate predictions
	plot_x_transformed = (plot_x+2)/4
	for j in range(K+1):
		coef = math.comb(K,j)
		plot_y += theta[[j]][0]*coef*np.power(plot_x_transformed, j)*\
		np.power(1-plot_x_transformed, K-j)
	return plot_y

def sieve_splines(df, K, plot_x):
	# arguments: df: pandas dataframe, K: number of knots
	# returns: list of predicted data to plot
	df_copy = copy.copy(df)
	plot_y = np.zeros(len(plot_x))
	yvars = ['Y']
	xvars = ['X']
	# first generate the knots
	thetas = []
	knots = []
	# construct the data (essentially filtered by what knot each variabel is in)
	for i in range(K):
		i_quantile = df_copy['X'].quantile((i+1)/(K+1))
		knots.append(i_quantile)
		Xvar = "X"+str(i+1)
		xvars.append(Xvar)
		df_copy[Xvar] = df_copy['X']>=i_quantile
		df_copy[Xvar] = np.multiply(df_copy[Xvar], (df_copy['X'] - i_quantile))
	theta = regress(yvars, xvars, df_copy)
	# transform the basis for predicting data based on results
	for j in range(len(plot_x)):
		binary = plot_x[j]>=knots
		xs = np.array((plot_x[j] - knots)*binary)
		plot_y[j] = np.dot(theta[1:-1].T, xs.T) + \
		theta[-1] + theta[0] * plot_x[j]
	return plot_y

# this is a helper function to perform linear regression
def regress(yvars, xvars, df, constants = True):
	# arguments: yvars: list of outcome variable
	#			 xvars: list of covariates to regress on
	#			 constants: default is true: include intercept term
	# returns: theta: list of coefficients with the last one being 
	#			      a constant if applicable
	# NOTE: the constant is always returned as the last argument
	df_copy = copy.copy(df)
	df_copy = df_copy.dropna()
	xvars_copy = copy.copy(xvars)
	if constants == True:
		df_copy['C'] = 1
		xvars_copy.append('C')
	X = df_copy[xvars_copy]
	Y = df_copy[yvars]
	# classic OLS estimator
	return np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))

def plot_mc(plot_x, all_ys, k, title, k_title, df):
	# arguments: plot_x: list of x's to predict,
	#            all_ys: list of all the simulated predictions (monte_carlo),
	#            k: tuning param, title: descriptor, 
	#            k_title: what is tuning parameter, df: real data
	# results: automatically saves .png plots to current directory
	means_y = np.mean(all_ys, axis = 1)
	sd_y = np.std(all_ys, axis = 1)
	sd_0 = means_y - sd_y
	sd_1 = means_y + sd_y
	fig, ax = plt.subplots()
	# style of data points
	ax.plot(df.X, df.Y, '.', markersize = .8, color='dodgerblue')
	# predicted means
	ax.plot(plot_x, means_y, color = 'dodgerblue', alpha = 0.7)
	# shade in the standard deviation around the means
	ax.plot(plot_x, sd_0, alpha = 0)
	ax.plot(plot_x, sd_1, alpha = 0)
	ax.fill_between(plot_x, sd_0, sd_1, color = 'dodgerblue', alpha = 0.2)
	ax.set(xlabel='X', ylabel='Y',
	       title=title + ", " + k_title + " = "+str(k))
	if title == "Sieve":
		# the sieve will go way off the screen if we don't force the window
		ax.set_ylim([-3,3])
	fig.savefig(title+"_"+str(k)+"_mc.png")
	#plt.show()


def monte_carlo(df, k, N, M, plot_x, title, k_title):
	# arguments: df: real data, k: tuning param, M: number of iterations
	#            plot_x: list of x's to predict, title: descriptor, 
	#            k_title: what is tuning parameter
	# results: calls plot_mc to save figures
	all_ys = np.zeros((len(plot_x), M))
	if title == "Kernel":
		# cycle through iterations
		for i in range(M):
			# sample/resample the size of the dataset
			samples = df.sample(N, replace = True)
			predicted_y = kernel(samples, k, plot_x)
			# append the results to this matrix, all_ys
			all_ys[:,i] = predicted_y
	elif title == "Local linear":
		for i in range(M):
			samples = df.sample(N, replace = True)
			predicted_y = local_linear(samples, k, plot_x)
			all_ys[:,i] = predicted_y
	elif title == "Nearest neighbors":
		for i in range(M):
			samples = df.sample(N, replace = True)
			predicted_y = nearest_neighbors(samples, k, plot_x)
			all_ys[:,i] = predicted_y
	elif title == "Sieve":
		for i in range(M):
			samples = df.sample(N, replace = True)
			predicted_y = sieve(samples, k, plot_x)
			all_ys[:,i] = predicted_y
	elif title == "Bernstein sieve":
		for i in range(M):
			samples = df.sample(N, replace = True)
			predicted_y = sieve_bernstein(samples, k, plot_x)
			all_ys[:,i] = predicted_y
	elif title == "Spline sieve":
		for i in range(M):
			samples = df.sample(N, replace = True)
			predicted_y = sieve_splines(samples, k, plot_x)
			all_ys[:,i] = predicted_y
	plot_mc(plot_x, all_ys, k, title, k_title, df)

# the specifications I report in the writeup 
monte_carlo(df, .05, N, 75, plot_x, "Kernel", "h")
monte_carlo(df, .3, N, 75, plot_x, "Kernel", "h")
monte_carlo(df, .05, N, 75, plot_x, "Local linear", "h")
monte_carlo(df, .3, N, 75, plot_x, "Local linear", "h")
monte_carlo(df, 2, N, 75, plot_x, "Nearest neighbors", "k")
monte_carlo(df,20, N, 75, plot_x, "Nearest neighbors", "k")
monte_carlo(df, 3, N, 75, plot_x, "Sieve", "K")
monte_carlo(df, 11, N, 75, plot_x, "Sieve", "K")
monte_carlo(df, 2, N, 75, plot_x, "Bernstein sieve", "K")
monte_carlo(df, 12, N, 75, plot_x, "Bernstein sieve", "K")
monte_carlo(df, 3, N, 75, plot_x, "Spline sieve", "K")
monte_carlo(df, 12, N, 75, plot_x, "Spline sieve", "K")

