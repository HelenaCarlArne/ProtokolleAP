import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
from uncertainties import ufloat
from uncertainties.unumpy import (nominal_values as nom,
                                  std_devs as std)
#
##	LEERLAUF
###
#

N_0=ufloat(166,np.sqrt(166))
n_0=N_0/900
print("LEERLAUF:")
print(n_0)

#
##	INDIUM
###
#

N_ind=np.genfromtxt("../Werte/Indium.txt").T 
N_ind_err=np.sqrt(N_ind)
n_ind=unp.uarray(N_ind/250,N_ind_err/250)

n_ind=unp.log(n_ind)

def f(t, a, b):
	return a*t+b

params, covariance = curve_fit(f, np.arange(250,(1+len(N_ind))*250,250), nom(n_ind),sigma=std(n_ind))
errors = np.sqrt(np.diag(covariance))
print("INDIUM:")
print('a =', params[0], 'pm', errors[0])
print('b =', params[1], 'pm', errors[1])

X = np.linspace(0, 4000)
plt.plot(X, f(X, *params), 'b-', label='Ausgleichsgerade')

plt.errorbar(np.arange(250,(1+len(N_ind))*250,250),nom(n_ind),yerr=std(n_ind),fmt="x",label="Indium")
plt.xlabel(r'Zeit $t$ in s')
plt.xticks(np.arange(250,(1+len(N_ind))*250,250),np.arange(250,(1+len(N_ind))*250,250))
plt.grid()
plt.ylabel(r'Logarithmierte Zerfallrate im Zeitintervall $\Delta t$')
plt.legend(loc="best")
plt.tight_layout()
#plt.yscale("log")
plt.show()

#
##	RHODIUM
###
#

N_rho=np.genfromtxt("../Werte/Rhodium.txt").T 
N_rho_err=np.sqrt(N_rho)
n_rho=unp.uarray(N_rho/20,N_rho_err/20)
n_rho=unp.log(n_rho)

AA=np.arange(20,360,20)
BB=nom(n_rho)[0:17]
CC=std(n_rho)[0:17]

AAA=np.arange(360,(1+len(N_rho))*20,20)
BBB=nom(n_rho)[17:41]
CCC=std(n_rho)[17:41]

#print(AA)
#print(BB)
#print(len(AA))
#print(len(BB))

params, covariance = curve_fit(f,AA,BB,sigma=CC)
errors = np.sqrt(np.diag(covariance))
print("RHODIUM-Kurzlebig:")
print('a =', params[0], 'pm', errors[0])
print('b =', params[1], 'pm', errors[1])
X = np.linspace(0, 480)
plt.plot(X, f(X, *params), 'b-', label='1. Ausgleichsgerade')

params, covariance = curve_fit(f,AAA,BBB,sigma=CCC)
errors = np.sqrt(np.diag(covariance))
print("RHODIUM-Langlebig:")
print('a =', params[0], 'pm', errors[0])
print('b =', params[1], 'pm', errors[1])
X = np.linspace(0, 800)
plt.plot(X, f(X, *params), 'g-', label='2. Ausgleichsgerade')

plt.errorbar(AA,BB,yerr=CC,fmt="bx",label="1.Rhodium")
plt.errorbar(AAA,BBB,yerr=CCC,fmt="gx",label="2.Rhodium")
plt.ylim(-1,4)
plt.xlabel('Zeit $t$ in s')
plt.grid()
plt.xticks(np.arange(20,(1+len(N_rho))*20,60),np.arange(20,(1+len(N_rho))*20,60))
plt.ylabel(r'Logarithmierte Zerfallrate im Zeitintervall $\Delta t$')
plt.legend(loc="best")
plt.tight_layout()
plt.show()