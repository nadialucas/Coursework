# Nadia Lucas

################################################################################


import numpy as np 
import pandas as pd 
import math
from scipy.stats import norm #for a normal cdf function
import matplotlib.pyplot as plt

datapath = "/Users/nadialucas/Dropbox/Second year/ECON 31720/ps3_files/"

df = pd.read_csv(datapath+"angrist_evans_clean.csv")


print(df.info(verbose=True))

Yvar = ['worked']
Dvar = ['more2kids']
Zvar = ['samesex']
Xvars = ['age', 'ageat1st', 'agekid1', 'agekid2', 'boy1st', 'boy2nd', 'black', 'hispanic', 'otherrace']
Zvars = Zvar+Xvars

# this will return u
def probit(D, Z):
	est = estimate(Z, D, Z)
	beta = est.regress()
	predicted = np.dot(beta.T, Z.T)
	return norm.cdf(predicted).T

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



class mtr_estimates:
	def __init__(self, df, Dvar, Zvars, Xvars, Yvar):
		self.D = df[Dvar].to_numpy()
		self.Z = df[Zvars].to_numpy()
		self.U = probit(self.D,self.Z)
		df['U'] = self.U
		self.X = df[Xvars].to_numpy()
		self.Y = df[Yvar].to_numpy()
		self.X1 = df[df[Dvar[0]] == 1][Xvars].to_numpy()
		self.X0 = df[df[Dvar[0]] == 0][Xvars].to_numpy()
		self.U1 = df[df[Dvar[0]] == 1]['U'].to_numpy()
		self.U0 = df[df[Dvar[0]] == 0]['U'].to_numpy()
		self.Y1 = df[df[Dvar[0]] == 1][Yvar].to_numpy()
		self.Y0 = df[df[Dvar[0]] == 0][Yvar].to_numpy()
		self.C1 = np.ones((len(self.U1), 1))
		self.C0 = np.ones((len(self.U0), 1))
		self.U1 = self.U1.reshape(len(self.U1), 1)
		self.U0 = self.U0.reshape(len(self.U0), 1)

	def mtr_helper(self, X1, X0):
		coefs1 = estimate(X1, self.Y1, self.U1).regress()
		coefs0 = estimate(X0, self.Y0, self.U0).regress()

		# use the estimates to impute the potential outcomes Y(1) and Y(0)
		Y0_imputed = np.dot(coefs0.T, X1.T).T
		Y1_imputed = np.dot(coefs1.T, X0.T).T

		# now construct an originally sized dataframe with potential outcomes (D==0 on top)
		Y0 = np.concatenate([self.Y0, Y0_imputed])
		Y1 = np.concatenate([Y1_imputed, self.Y1])
		MTE = Y1 - Y0
		C = np.ones((len(MTE), 1))
		U = np.concatenate([self.U0, self.U1])
		D = np.concatenate([np.zeros(len(self.U0)), np.ones(len(self.U1))])


		# WHY are these all virtually identical???
		ATE = np.mean(Y1) - np.mean(Y0)
		ATT = np.mean(self.Y1) - np.mean(Y0_imputed)
		ATU = np.mean(Y1_imputed) - np.mean(self.Y0)
		print(ATE)
		print(ATT)
		print(ATU)

		# what are we supposed to be plotting???
		# Evaluate X at the mean


		# fig, ax = plt.subplots()
		# ax.plot(Uplot, MTE_impute, '.', markersize =.8, color='dodgerblue')
		# plt.show()

		return(coefs0, coefs1)

	def mtr1(self):
		X1 = np.hstack((np.hstack((self.C1, self.U1)), self.X1))
		X0 = np.hstack((np.hstack((self.C0, self.U0)), self.X0))
		Xbar = np.mean(self.X, axis = 0)
		print(Xbar)
		coefs0, coefs1 = self.mtr_helper(X1, X0)

		Uplot = np.arange(0, 1, .001)

		Xsplot = np.repeat(Xbar, len(Uplot), axis = 1)
		print(Xsplot.shape)

		#Y1plot = 
		





hi = mtr_estimates(df, Dvar, Zvars, Xvars, Yvar)
hi.mtr1()






# mtr_estimate()




