# Nadia Lucas

################################################################################


import numpy as np 
import pandas as pd 
import math
from scipy.stats import norm #for a normal cdf function
import matplotlib.pyplot as plt
import statsmodels.api as sm

datapath = "/Users/nadialucas/Dropbox/Second year/ECON 31720/ps3_files/"

df = pd.read_csv(datapath+"angrist_evans_clean.csv")
df['C'] = 1

Yvar = ['worked']
Dvar = ['more2kids']
Xvars = ['age', 'ageat1st', 'agekid1', 'agekid2', 'boy1st', 'boy2nd', 'black', 'hispanic', 'otherrace']


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

# get the propensity scores
def logit(df, Zvar, Dvar, Xvars):
	data = df
	model = sm.GLM(data[Dvar], data[Zvar+Xvars], family = sm.families.Binomial())
	results = model.fit()
	theta = results.params.to_numpy()
	pred = np.dot(theta.T, df[Zvar+Xvars].T)
	pscores = np.divide(np.exp(pred), (1+np.exp(pred)))
	return pscores.reshape(len(pscores.flatten()),1)



class mtr_estimates:
	def __init__(self, df, Dvar, Zvar, Xvars, Yvar, name):
		self.D = df[Dvar].to_numpy()
		self.Z = df[Zvar+Xvars].to_numpy()
		self.justZ = df[Zvar].to_numpy()
		self.X = df[Xvars].to_numpy()
		self.U = logit(df, Zvar, Dvar, Xvars+['C'])
		self.C = np.ones((len(self.U), 1))
		df['U'] = self.U
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
		self.name = name

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


		# this is def not the right way to solve for this
		ATE = np.mean(Y1) - np.mean(Y0)
		ATT = np.mean(self.Y1) - np.mean(Y0_imputed)
		ATU = np.mean(Y1_imputed) - np.mean(self.Y0)
		LATE = np.sum(np.divide(np.multiply(U, np.subtract(Y1, Y0)), np.sum(U)))
		print("ATE: ",round(ATE, 5))
		print("ATT: ",round(ATT, 5))
		print("ATU: ",round(ATU, 5))
		print("LATE: ",round(LATE, 5))

		return(coefs0, coefs1)

	def mtr1(self):
		print("Spec 1")
		X1 = np.hstack((np.hstack((self.C1, self.U1)), self.X1))
		X0 = np.hstack((np.hstack((self.C0, self.U0)), self.X0))
		Xbar = np.mean(self.X, axis = 0)
		coefs0, coefs1 = self.mtr_helper(X1, X0)

		Uplot = np.arange(0, 1, .001)
		Xsplot = np.repeat([Xbar], len(Uplot), axis = 0)
		Xsplot = np.insert(Xsplot, 0, Uplot, axis = 1)
		Xsplot = np.insert(Xsplot, 0, np.ones(len(Uplot)), axis = 1)

		M0 = np.dot(coefs0.T, Xsplot.T)
		M1 = np.dot(coefs1.T, Xsplot.T)
		MTEplot = (M1 - M0).flatten()

		fig, ax = plt.subplots()
		ax.plot(Uplot, MTEplot, '-', linewidth = 1, color = 'dodgerblue')
		ax.set(title = 'Specification 1 MTE')
		fig.savefig(self.name+"_mte1.png")

	def mtr2(self):
		print("Spec 2")
		DU = np.multiply(self.U, self.D)
		X = np.hstack((np.hstack((np.hstack((np.hstack((self.C, self.D)), self.U)), DU)), self.X))
		
		Xbar = np.mean(self.X, axis = 0)
		coefs = estimate(X, self.Y, self.U).regress().flatten()
		alpha_0 = coefs[0]
		alpha_1 = coefs[0] + coefs[1]
		beta_0 = coefs[2]
		beta_1 = coefs[2]+coefs[3]
		gamma = coefs[4:]

		coefs0 = np.insert(gamma, 0, beta_0)
		coefs0 = np.expand_dims(np.insert(coefs0, 0, alpha_0), axis = 1)
		coefs1 = np.insert(gamma, 0, beta_1)
		coefs1 = np.expand_dims(np.insert(coefs1, 0, alpha_1), axis = 1)

		X1 = np.hstack((np.hstack((self.C1, self.U1)), self.X1))
		X0 = np.hstack((np.hstack((self.C0, self.U0)), self.X0))

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


		# this is def not the right way to solve for this
		ATE = np.mean(Y1) - np.mean(Y0)
		ATT = np.mean(self.Y1) - np.mean(Y0_imputed)
		ATU = np.mean(Y1_imputed) - np.mean(self.Y0)
		LATE = np.sum(np.divide(np.multiply(U, np.subtract(Y1, Y0)), np.sum(U)))
		print("ATE: ",round(ATE, 5))
		print("ATT: ",round(ATT, 5))
		print("ATU: ",round(ATU, 5))
		print("LATE: ",round(LATE, 5))

		Uplot = np.arange(0, 1, .001)
		Xsplot = np.repeat([Xbar], len(Uplot), axis = 0)
		Xsplot = np.insert(Xsplot, 0, Uplot, axis = 1)
		Xsplot = np.insert(Xsplot, 0, np.ones(len(Uplot)), axis = 1)

		M0 = np.dot(coefs0.T, Xsplot.T)
		M1 = np.dot(coefs1.T, Xsplot.T)
		MTEplot = (M1 - M0).flatten()

		fig, ax = plt.subplots()
		ax.plot(Uplot, MTEplot, '-', linewidth = 1, color = 'dodgerblue')
		ax.set(title = 'Specification 2 MTE')
		fig.savefig(self.name+"_mte2.png")


	def mtr3(self):
		print("Spec 3")
		XU1 = np.multiply(self.U1, self.X1)
		XU0 = np.multiply(self.U0, self.X0)
		X1 = np.hstack((np.hstack((np.hstack((self.C1, self.U1)), self.X1)), XU1))
		X0 = np.hstack((np.hstack((np.hstack((self.C0, self.U0)), self.X0)), XU0))
		Xbar = np.mean(self.X, axis = 0)
		coefs0, coefs1 = self.mtr_helper(X1, X0)

		Uplot = np.arange(0, 1, .001)
		XUplot = np.dot(np.expand_dims(Xbar, axis=1), np.expand_dims(Uplot, axis=1).T).T
		Xsplot1 = np.repeat([Xbar], len(Uplot), axis = 0)
		Xsplot = np.hstack((Xsplot1, XUplot))
		Xsplot = np.insert(Xsplot, 0, Uplot, axis = 1)
		Xsplot = np.insert(Xsplot, 0, np.ones(len(Uplot)), axis = 1)

		M0 = np.dot(coefs0.T, Xsplot.T)
		M1 = np.dot(coefs1.T, Xsplot.T)
		MTEplot = (M1 - M0).flatten()

		fig, ax = plt.subplots()
		ax.plot(Uplot, MTEplot, '-', linewidth = 1, color = 'dodgerblue')
		ax.set(title = 'Specification 3 MTE')
		fig.savefig(self.name+"_mte3.png")

	def mtr4(self):
		print("Spec 4")
		U12 = np.power(self.U1, 2)
		U02 = np.power(self.U0, 2)
		X1 = np.hstack((np.hstack((np.hstack((self.C1, self.U1)), U12)), self.X1))
		X0 = np.hstack((np.hstack((np.hstack((self.C0, self.U0)), U02)), self.X0))
		Xbar = np.mean(self.X, axis = 0)
		coefs0, coefs1 = self.mtr_helper(X1, X0)

		Uplot = np.arange(0, 1, .001)
		Uplot2 = np.power(Uplot, 2)
		Xsplot = np.repeat([Xbar], len(Uplot), axis = 0)
		Xsplot = np.insert(Xsplot, 0, Uplot2, axis = 1)
		Xsplot = np.insert(Xsplot, 0, Uplot, axis = 1)
		Xsplot = np.insert(Xsplot, 0, np.ones(len(Uplot)), axis = 1)

		M0 = np.dot(coefs0.T, Xsplot.T)
		M1 = np.dot(coefs1.T, Xsplot.T)
		MTEplot = (M1 - M0).flatten()

		fig, ax = plt.subplots()
		ax.plot(Uplot, MTEplot, '-', linewidth = 1, color = 'dodgerblue')
		ax.set(title = 'Specification 4 MTE')
		fig.savefig(self.name+"_mte4.png")

	def mtr5(self):
		print("Spec 5")
		U12 = np.power(self.U1, 2)
		U02 = np.power(self.U0, 2)
		U13 = np.power(self.U1, 3)
		U03 = np.power(self.U0, 3)
		X1 = np.hstack((np.hstack((np.hstack((np.hstack((self.C1, self.U1)), U12)), U13)), self.X1))
		X0 = np.hstack((np.hstack((np.hstack((np.hstack((self.C0, self.U0)), U02)), U03)), self.X0))
		Xbar = np.mean(self.X, axis = 0)
		coefs0, coefs1 = self.mtr_helper(X1, X0)

		Uplot = np.arange(0, 1, .001)
		Uplot2 = np.power(Uplot, 2)
		Uplot3 = np.power(Uplot, 3)
		Xsplot = np.repeat([Xbar], len(Uplot), axis = 0)
		Xsplot = np.insert(Xsplot, 0, Uplot2, axis = 1)
		Xsplot = np.insert(Xsplot, 0, Uplot3, axis = 1)
		Xsplot = np.insert(Xsplot, 0, Uplot, axis = 1)
		Xsplot = np.insert(Xsplot, 0, np.ones(len(Uplot)), axis = 1)

		M0 = np.dot(coefs0.T, Xsplot.T)
		M1 = np.dot(coefs1.T, Xsplot.T)
		MTEplot = (M1 - M0).flatten()

		fig, ax = plt.subplots()
		ax.plot(Uplot, MTEplot, '-', linewidth = 1, color = 'dodgerblue')
		ax.set(title = 'Specification 5 MTE')
		fig.savefig(self.name+"_mte5.png")
		
		
Yvar = ['worked']
Dvar = ['more2kids']
Xvars = ['age', 'ageat1st', 'agekid1', 'agekid2', 'boy1st', 'boy2nd', 'black', 'hispanic', 'otherrace']
X = df[['C']+Dvar+Xvars].to_numpy()
Y = df[Yvar].to_numpy()

# instrument 1
print("Same sex instrument:")
Zvar = ['samesex']
Z = df[Zvar+['C']+Xvars].to_numpy()
print("TSLS: ", round(estimate(X, Y, Z).tsls()[1][0], 5))
inst1 = mtr_estimates(df, Dvar, Zvar, Xvars, Yvar, "samesex")
inst1.mtr1()
inst1.mtr2()
inst1.mtr3()
inst1.mtr4()
inst1.mtr5()



# instrument 2
print("Twins instrument:")
Zvar = ['twins']
Z = df[Zvar+['C']+Xvars].to_numpy()
print("TSLS: ", round(estimate(X, Y, Z).tsls()[1][0], 5))
inst2 = mtr_estimates(df, Dvar, Zvar, Xvars, Yvar, "twins")
inst2.mtr1()
inst2.mtr2()
inst2.mtr3()
inst2.mtr4()
inst2.mtr5()


# instrument 3
print("Both instruments:")
Zvar = ['samesex', 'twins']
Z = df[Zvar+['C']+Xvars].to_numpy()
print("TSLS: ", round(estimate(X, Y, Z).tsls()[1][0], 5))
inst3 = mtr_estimates(df, Dvar, Zvar, Xvars, Yvar, "2 instruments")
inst3.mtr1()
inst3.mtr2()
inst3.mtr3()
inst3.mtr4()
inst3.mtr5()





# mtr_estimate()




