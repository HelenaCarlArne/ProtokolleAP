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

N0=ufloat(166,np.sqrt(166))
n0=N0/900
print("LEERLAUF:")
print(n0)
print("")

#
##	INDIUM
###
#

N_ind=np.genfromtxt("../Werte/Indium.txt").T 
N_ind_err=np.sqrt(N_ind)
N_ind=unp.uarray(N_ind-nom(n0)*250,N_ind_err-std(n0)*250)
print("INDIUM:")
print("Berichtigung:")
print("")
#print(N_ind)
N_ind=unp.log(N_ind)

def f(t, a, b):
	return a*t+b

params, covariance = curve_fit(f, np.arange(250,(1+len(N_ind))*250,250), nom(N_ind),sigma=std(N_ind))
errors = np.sqrt(np.diag(covariance))
print("Parameter:")
a=ufloat(params[0],errors[0])
print("a={}".format(a))
print("Da={}".format((a-0.0002128)))##################################################
print("da={}".format(((a-0.0002128)/0.0002128)))##################################################
b=ufloat(params[1],errors[1])
print("b={}".format(b))
print("")
print('T ={}'.format(np.log(2)/-a))
print('DT ={}'.format((np.log(2)/-a)-3257))
print('N0 ={}'.format(unp.exp(b)/(1-unp.exp(a*250))))
print("")
X = np.linspace(0, 4000)
plt.plot(X, f(X, *params), 'b-', label='Ausgleichsgerade')

plt.errorbar(np.arange(250,(1+len(N_ind))*250,250),nom(N_ind),yerr=std(N_ind),fmt="x",label="Indium")
plt.xlabel(r'Zeit $t$ in s')
plt.xticks(np.arange(250,(1+len(N_ind))*250,250),np.arange(250,(1+len(N_ind))*250,250))
plt.grid()
plt.ylabel(r'Logarithmierte Zerfallrate $\Delta N$ im Zeitintervall $\Delta t$')
plt.legend(loc="best")
plt.tight_layout()
#plt.yscale("log")
plt.savefig("../Bilder/indium.pdf")

plt.close()

#
##	RHODIUM
###
#

N_rho=np.genfromtxt("../Werte/Rhodium.txt").T 
N_rho_err=np.sqrt(N_rho)
n_rho=unp.uarray(N_rho-nom(n0)*20,N_rho_err-std(n0)*20)
print("RHODIUM:")
print("Berichtigung:")
#print(n_rho)
print("")
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



params, covariance = curve_fit(f,AAA,BBB,sigma=CCC)
errors = np.sqrt(np.diag(covariance))
print("RHODIUM-Langlebige Parameter:")
a=ufloat(params[0],errors[0])
print("a={}".format(a))
print("Da={}".format(a-0.002666))###############################################################################
print("da={}".format((a-0.002666)/0.002666))###############################################################################
print(0.0045/0.002666)
a_star=a
b=ufloat(params[1],errors[1])
print("b={}".format(b))
b_star=b
print("")
print('T ={}'.format(np.log(2)/-a))
print('DT ={}'.format((np.log(2)/-a)-260))#############################################################
print('dT ={}'.format(((np.log(2)/-a)-260)/260))#############################################################
print('N0 ={}'.format(unp.exp(b)/(1-unp.exp(a*250))))
print("")
X = np.linspace(0, 800)
plt.plot(X, f(X, *params), 'g-', label='2. Ausgleichsgerade')

params, covariance = curve_fit(f,AA,BB,sigma=CC)
errors = np.sqrt(np.diag(covariance))
print("RHODIUM-Kurzlebige Parameter:")
a=ufloat(params[0],errors[0])
print("a={}".format(a))
print("Da={}".format(a-0.01639))################################################################
print("da={}".format((a-0.01639)/0.01639))################################################################
b=ufloat(params[1],errors[1])
print("b={}".format(b))
print("")
print('T ={}'.format(np.log(2)/-a))
print('DT ={}'.format((np.log(2)/-a)-90))#########################################################
print('DT ={}'.format(((np.log(2)/-a)-90)/90))#########################################################
print('N0 ={}'.format(unp.exp(b)/(1-unp.exp(a*250))))
print("")
X = np.linspace(0, 500)
plt.plot(X, f(X, *params), 'b-', label='1. Ausgleichsgerade')
plt.errorbar(AA,BB,yerr=CC,fmt="bx",label="1.Rhodium")
plt.errorbar(AAA,BBB,yerr=CCC,fmt="gx",label="2.Rhodium")
plt.ylim(1.4,6.5)
plt.xlabel('Zeit $t$ in s')
plt.grid()
plt.xticks(np.arange(20,(1+len(N_rho))*20,60),np.arange(20,(1+len(N_rho))*20,60))
plt.ylabel(r'Logarithmierte Zerfallrate $\Delta N$ im Zeitintervall $\Delta t$')
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("../Bilder/rhodium.pdf")
plt.close()

plt.errorbar(AA,BB,yerr=CC,fmt="kx",label="Rhodium")
plt.errorbar(AAA,BBB,yerr=CCC,fmt="kx")
plt.ylim(1.4,6.5)
plt.xlabel('Zeit $t$ in s')
plt.grid()
plt.xticks(np.arange(20,(1+len(N_rho))*20,60),np.arange(20,(1+len(N_rho))*20,60))
plt.ylabel(r'Logarithmierte Zerfallrate $\Delta N$ im Zeitintervall $\Delta t$')
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("../Bilder/rhodium_show.pdf")
plt.close()

#for i in range(0,40):
#	print(20*i+20)