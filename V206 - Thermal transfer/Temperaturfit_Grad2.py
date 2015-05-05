import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy.constants as const

plt.rcParams['figure.figsize']= (10, 8)
plt.rcParams['font.size'] = 16

#Dateneinlesen 1
t, T_1 = np.genfromtxt('Temperatur_1.txt').T 	
T_1 = const.C2K(T_1)
#t = np.arange(0,1200)							### Hier wird die Zeit definiert. Wir gehen aber von 0 bis 20!

#Funktion 1 definieren
def f(t, a, b, c):
	return (a*(t**2)) + (b*t) + c
params, covariance = curve_fit(f, t, T_1)

##################Temperatur1
#Parameter der Funktion 1
errors = np.sqrt(np.diag(covariance))
print('a =',params[0],'+/-',errors[0],'K/s^3')
print('b =',params[1],'+/-',errors[1],'K/s^2')
print('c =',params[2],'+/-',errors[2],'K/s')

print('')
t_plot = np.linspace(0, 1200) 				### wahlweise 19 ;)

#Temperatur1 plotten
plt.plot(t, T_1, 'r.', label='Temperatur 1')
plt.plot(t_plot, f(t_plot, *params), 'r-', label='nicht-linearer Fit')


##################Temperatur2
#Dateneinlesen 2
T_2 = np.genfromtxt('Temperatur_2.txt').T
T_2 = const.C2K(T_2)

#Funktion 2 definieren
def i(t, e, f, g):
	return (e*t**2)+f*t+g
params, covariance = curve_fit(f, t, T_2)

#Parameter der Funktion 2
errors = np.sqrt(np.diag(covariance))
print('e =',params[0],'+/-',errors[0],'K/m^3')
print('f =',params[1],'+/-',errors[1],'K/m^2')
print('g =',params[2],'+/-',errors[2],'K/m')
	
print('')
#Temperatur2 plotten
plt.plot(t, T_2, 'b.', label='Temperatur 2')
plt.plot(t_plot, i(t_plot, *params), 'b-', label='nicht-linearer Fit')
plt.xlabel('Zeit $t$ /s')
plt.ylabel('Temperatur $T$ /K')
#Plotten
plt.legend(loc='best')
plt.show()
