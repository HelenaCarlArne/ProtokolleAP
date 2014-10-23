
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


gross = np.genfromtxt('Messung1_Raumtemperatur_gross.txt').T
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

etaG_2= (apparatG)*(rhoG-rhoW)*(fallzeit)

Re = (rhoW*(0.1/fallzeit)*rG*2)/etaG_2

print("Viskositaet in Pa*sec:")
print(etaG_2) 
print("")
print("Viskositaet in mPa*sec, vgl. Literatur:")
print(etaG_2*1000) 
print("")
print("Geschwindigkeiten in m/s:")
print((0.1/fallzeit))
print("")
print("Reynoldzahlen:")
print(Re)

np.savetxt("Auswertung/Viskositaeten.txt", np.array([noms(etaG_2),stds(etaG_2)]).T)
np.savetxt("Auswertung/Reynoldzahlen.txt", np.array([noms(Re),stds(Re)]).T)
np.savetxt("Auswertung/Geschwindigkeiten.txt", np.array([noms((0.1/fallzeit)),stds((0.1/fallzeit))]).T)
np.savetxt("Auswertung/Zeiten.txt", np.array([noms((fallzeit)),stds((fallzeit))]).T)









