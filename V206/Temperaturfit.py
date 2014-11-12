import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy.constants as const

plt.rcParams['figure.figsize']= (10, 8)
plt.rcParams['font.size'] = 16

#Dateneinlesen 1
T_1 = np.genfromtxt('Temperatur_1.txt').T 	### Hier gab es eine Spalte zuviel! Original: t, T_1 = np.gen…
T_1 = const.C2K(T_1)
t = np.arange(1,21)							### Hier wird die Zeit definiert.

#Funktion 1 definieren
def f(t, a, b, c, d ):
	return (a*(t**3)) + (b*t**2) + c*t + d
params, covariance = curve_fit(f, t, T_1)

##################Temperatur1
#Parameter der Funktion 1
errors = np.sqrt(np.diag(covariance))
print('a =',params[0],'+/-',errors[0],'K/s^3')
print('b =',params[1],'+/-',errors[1],'K/s^2')
print('c =',params[2],'+/-',errors[2],'K/s')
print('d =',params[3],'+/-',errors[3],'K')
print('')
t_plot = np.linspace(0, 19) 				### wahlweise 19 ;)

#Temperatur1 plotten
plt.plot(t, T_1, 'r.', label='Temperatur 1')
plt.plot(t_plot, f(t_plot, *params), 'r-', label='nicht-linearer Fit')


##################Temperatur2
#Dateneinlesen 2
T_2 = np.genfromtxt('Temperatur_2.txt').T
T_2 = const.C2K(T_2)

#Funktion 2 definieren
def i(t, e, f, g, h):
	return (e*t**3)+f*t**2+g*t+h
params, covariance = curve_fit(f, t, T_2)

#Parameter der Funktion 2
errors = np.sqrt(np.diag(covariance))
print('e =',params[0],'+/-',errors[0],'K/s^3')
print('f =',params[1],'+/-',errors[1],'K/s^2')
print('g =',params[2],'+/-',errors[2],'K/s')
print('h =',params[3],'+/-',errors[3],'K')
#t_plot = np.linspace(0, 20)				### Das hattest Du ja schon :)
print('')
#Temperatur2 plotten
plt.plot(t, T_2, 'b.', label='Temperatur 2')
plt.plot(t_plot, i(t_plot, *params), 'b-', label='nicht-linearer Fit')

#Plotten
plt.legend(loc='best')
plt.show()


###Berechent dT/dt für 4 verschiedene zeiten

a = -0.00370884780176
b = 0.101652264815
c = 0.974085469429
d = 294.111603614 

e = 0.00731279683143 
f = -0.190660095953 
g = -0.195335574077 
h = 294.97761152 


#wähle t1=120 t=480 t2=600, t3=840, t4=1080

dT1_dt_1= 3*a*(120**2)+2*b*120+c
dT1_dt_2= 3*a*(480**2)+2*b*480+c
dT1_dt_3= 3*a*(840**2)+2*b*840+c
dT1_dt_4= 3*a*(1080**2)+2*b*1080+c

dT2_dt_1= 3*e*(120**2)+2*f*120+g
dT2_dt_2= 3*e*(480**2)+2*f*480+g
dT2_dt_3= 3*e*(840**2)+2*f*840+g
dT2_dt_4= 3*e*(1080**2)+2*f*1080+g

print('')
print('dT1_dt_1=',dT1_dt_1,'K/s')
print('dT1_dt_2=',dT1_dt_2,'K/s')
print('dT1_dt_3=',dT1_dt_3,'K/s')
print('dT1_dt_4=',dT1_dt_4,'K/s')

print('')
print('dT2_dt_1=',dT1_dt_1,'K/s')
print('dT2_dt_2=',dT1_dt_2,'K/s')
print('dT2_dt_3=',dT1_dt_3,'K/s')
print('dT2_dt_4=',dT1_dt_4,'K/s')


