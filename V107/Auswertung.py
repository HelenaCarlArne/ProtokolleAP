
import numpy as np 		
import uncertainties.unumpy as unp										#### Header
from scipy import stats
from uncertainties import ufloat
from uncertainties import umath
from uncertainties.unumpy import nominal_values as noms, std_devs as stds
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math


gross = np.genfromtxt('Messung1_Raumtemperatur_gross.txt').T 	#### Einladen der Daten
klein = np.genfromtxt('Messung1_Raumtemperatur_klein.txt').T
ta25,ta30,ta35,ta40,ta45,ta50,ta55,ta60,ta65,ta70 = np.genfromtxt('Messung2_Temperaturabhaengigkeit.txt')


mG=0.00461			# in kg											#### Konstanten
mK=0.00444			# in kg
rG=0.0158/2 		# in m
rK=0.01565/2 	# in m
rhoW = 997.54 # Dichte des Wassers bei 23 C in kg/m^3
apparatK = 7.64*10**(-8) #Pa m³/kg

g = ufloat(np.mean(gross),stats.sem(gross)) 					#### Berechnung der Werte
k = ufloat(np.mean(klein),stats.sem(klein))
t25 = ufloat(np.mean(ta25),stats.sem(ta25))
t30 = ufloat(np.mean(ta30),stats.sem(ta30))
t35 = ufloat(np.mean(ta35),stats.sem(ta35))
t40 = ufloat(np.mean(ta40),stats.sem(ta40))
t45 = ufloat(np.mean(ta45),stats.sem(ta45))
t50 = ufloat(np.mean(ta50),stats.sem(ta50))
t55 = ufloat(np.mean(ta55),stats.sem(ta55))
t60 = ufloat(np.mean(ta60),stats.sem(ta60))
t65 = ufloat(np.mean(ta65),stats.sem(ta65))
t70 = ufloat(np.mean(ta70),stats.sem(ta70))
fallzeit = np.array([t25,t30,t35,t40,t45,t50,t55,t60,t65,t70])

rhoG = mG/((4/3)*np.pi*(rG)**3)
rhoK = mK/((4/3)*np.pi*((rK)**3))

etaK= apparatK*(rhoK-rhoW)*k
apparatG=etaK/((rhoG-rhoW)*g)

#print("Fuer 25c:",t25)											#### Ausgabe
#print("Fuer 30c:",t30)
#print("")
#print("Fuer 35c:",t35)
#print("Fuer 40c:",t40)
#print("")
#print("Fuer 45c:",t45)
#print("Fuer 50c:",t50)
#print("")
#print("Fuer 50c:",t55)
#print("Fuer 60c:",t60)
#print("")
#print("Fuer 65c:",t65)	
#print("Fuer 70c:",t70)
#print("")
print("Zeit fuer die grosse Kugel:",g,"sec")
print("Zeit fue die kleine Kugel:",k,"sec")
print("")
print("Die Dichte der grossen Kugel:", rhoG, "kg/m^3")
print("Die Dichte der kleinen Kugel:", rhoK, "kg/m^3")
print("")
print("Eta, bestimmt mit der kleinen Kugel, ist:",etaK)
print("Mit diesem Eta ist die Apparaturkonstante:", apparatG)
print("")


T = np.arange(25,75,5)+273.15										#### Regressionen
T_func = np.linspace(0.0030,0.0034)

def f(x, m, b):
    return m*x+b
def linregress(x, y):
    N = len(y) 
    Delta = N*np.sum(x**2)-(np.sum(x))**2
    m = (N*np.sum(x*y)-np.sum(x)*np.sum(y))/Delta
    b = (np.sum(x**2) * np.sum(y) - np.sum(x) * np.sum(x * y)) / Delta
    sigma_y = np.sqrt(np.sum((y - m * x - b)**2) / (N - 2))
    m_error = sigma_y * np.sqrt(N / Delta)
    b_error = sigma_y * np.sqrt(np.sum(x**2) / Delta)
    print ("Die Regressionssteigung/B ist:", m,"+/-", m_error)
    print ("Der Regressions-Y-Achsenabschnitt ist:", b,"+/-", b_error)
    print("A ist:",np.exp(b),"+/-",np.sqrt(np.exp(2*b)*b_error**2))
    return(m,m_error,b,b_error) 

																#### Plots
etaG =unp.log(apparatG*(rhoG-rhoW)*fallzeit)
linregress(1/T,noms(etaG))
etaG_2= (noms(apparatG))*(rhoG-rhoW)*(noms(fallzeit))
#etaG_neu= unp.log(apparatG)+unp.log(rhoG-rhoW)+unp.log(fallzeit)


reg, cov = curve_fit(f, 1/T, noms(etaG))
plt.plot(T_func, f(T_func, *reg), 'r-', label='Fit')
plt.text(0.0031,-7.8, r"$x_{lin}= \; T^{-1} \; \lbrack a.u.\rbrack$", horizontalalignment='center',fontsize=12)
plt.text(0.00285,-7.1, r"$y = 29.61x+7.43$", horizontalalignment='center',color="r",fontsize=12)
plt.ylabel(r'$ln(\eta)\;\lbrack a.u.\rbrack$')
plt.errorbar(1/T, noms(etaG) ,yerr=stds(etaG),fmt="bx", label="Werte")
plt.legend(loc="best")
plt.tight_layout
plt.show()
print(etaG_2*1000)

