import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy.constants as const
import uncertainties
from uncertainties import ufloat

plt.rcParams['figure.figsize']= (10, 8)
plt.rcParams['font.size'] = 16

##################Temperatur1

#Dateneinlesen 1
t, T_1 = np.genfromtxt('Temperatur_1.txt').T 	
T_1 = const.C2K(T_1)

#Funktion 1 definieren
def f(t, a, b, c, d ):
	return (a*(t**3)) + (b*t**2) + c*t + d
params1, covariance1 = curve_fit(f, t, T_1)

#Parameter der Funktion 1
errors1 = np.sqrt(np.diag(covariance1))

t_plot = np.linspace(0, 1140) 				

#Temperatur1 plotten
plt.plot(t, T_1, 'r.', label='Temperatur 1')
plt.plot(t_plot, f(t_plot, *params1), 'r-', label='nicht-linearer Fit')


##################Temperatur2

#Dateneinlesen 2
T_2 = np.genfromtxt('Temperatur_2.txt').T
T_2 = const.C2K(T_2)

#Funktion 2 definieren
def i(t, e, f, g, h):
	return (e*t**3)+f*t**2+g*t+h
params2, covariance2 = curve_fit(i, t, T_2)

#Parameter der Funktion 2
errors2 = np.sqrt(np.diag(covariance2))


#Temperatur2 plotten
plt.plot(t, T_2, 'b.', label='Temperatur 2')
plt.plot(t_plot, i(t_plot, *params2), 'b-', label='nicht-linearer Fit')
plt.xlabel('Zeit t /s')
plt.ylabel('Temperatur T /K')

#Plotten
plt.legend(loc='best')
#plt.show()


###Berechne dT/dt für 4 verschiedene zeiten
a = ufloat(params1[0],errors1[0]) 
b = ufloat(params1[1],errors1[1])
c = ufloat(params1[2],errors1[2])
d = ufloat(params1[3],errors1[3])

e = ufloat(params2[0],errors2[0])
f = ufloat(params2[1],errors2[1])
g = ufloat(params2[2],errors2[2])
h = ufloat(params2[3],errors2[3])

print('a =',a,'K/s^3')
print('b =',b,'K/s^2')
print('c =',c,'K/s')
print('d =',d,'K')
print('')
print('e =',e,'K/m^3')
print('f =',f,'K/m^2')
print('g =',g,'K/m')
print('h =',h,'K')				
print('')


#wähle t1=120 t=480 t2=600, t3=840, t4=1080

dT1_dt_1= 3*a*(120**2)+2*b*120+c
dT1_dt_2= 3*a*(480**2)+2*b*480+c
dT1_dt_3= 3*a*(840**2)+2*b*840+c
dT1_dt_4= 3*a*(1080**2)+2*b*1080+c

dT2_dt_1= 3*e*(120**2)+2*f*120+g
dT2_dt_2= 3*e*(480**2)+2*f*480+g
dT2_dt_3= 3*e*(840**2)+2*f*840+g
dT2_dt_4= 3*e*(1080**2)+2*f*1080+g

##Fehler delta(dT/dt)


print('')
print('dT1_dt_1=',dT1_dt_1)
print('dT1_dt_2=',dT1_dt_2)
print('dT1_dt_3=',dT1_dt_3)
print('dT1_dt_4=',dT1_dt_4)

print('')
print('dT2_dt_1=',dT2_dt_1)
print('dT2_dt_2=',dT2_dt_2)
print('dT2_dt_3=',dT2_dt_3)
print('dT2_dt_4=',dT2_dt_4)


