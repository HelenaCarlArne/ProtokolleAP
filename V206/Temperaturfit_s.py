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
t_plot = np.linspace(0, 1140) 				### wahlweise 19 ;)

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
print('e =',params[0],'+/-',errors[0],'K/m^3')
print('f =',params[1],'+/-',errors[1],'K/m^2')
print('g =',params[2],'+/-',errors[2],'K/m')
print('h =',params[3],'+/-',errors[3],'K')				
print('')
#Temperatur2 plotten
plt.plot(t, T_2, 'b.', label='Temperatur 2')
plt.plot(t_plot, i(t_plot, *params), 'b-', label='nicht-linearer Fit')
plt.xlabel('Zeit t /s')
plt.ylabel('Temperatur T /K')
#Plotten
plt.legend(loc='best')
#plt.show()


###Berechent dT/dt für 4 verschiedene zeiten
a = -1.71705895195e-08 
b = 2.82367366738e-05 
c = 0.0162347592968 
d = 294.111603517 

e = 3.3855559517e-08 
f = -5.29611716421e-05 
g = -0.00325557626607 
h = 294.977609839 



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
error_dT1_dt_1= np.sqrt((2*120*0.19259208147)**2+0.00150160292315)
error_dT1_dt_2= np.sqrt((2*480*0.19259208147)**2+0.00150160292315)
error_dT1_dt_3= np.sqrt((2*840*0.19259208147)**2+0.00150160292315)
error_dT1_dt_4= np.sqrt((2*1080*0.19259208147)**2+0.00150160292315)

error_dT2_dt_1= np.sqrt((2*120* 0.265307963098)**2+ 0.00206855306032)
error_dT2_dt_2= np.sqrt((2*480* 0.265307963098)**2+ 0.00206855306032)
error_dT2_dt_3= np.sqrt((2*840* 0.265307963098)**2+ 0.00206855306032)
error_dT2_dt_4= np.sqrt((2*1080* 0.265307963098)**2+ 0.00206855306032)

print('')
print('dT1_dt_1=',dT1_dt_1,'+/-',error_dT1_dt_1,'K/s')
print('dT1_dt_2=',dT1_dt_2,'+/-',error_dT1_dt_2,'K/s')
print('dT1_dt_3=',dT1_dt_3,'+/-',error_dT1_dt_3,'K/s')
print('dT1_dt_4=',dT1_dt_4,'+/-',error_dT1_dt_4,'K/s')

print('')
print('dT2_dt_1=',dT1_dt_1,'+/-',error_dT2_dt_1,'K/s')
print('dT2_dt_2=',dT1_dt_2,'+/-',error_dT2_dt_2,'K/s')
print('dT2_dt_3=',dT1_dt_3,'+/-',error_dT2_dt_3,'K/s')
print('dT2_dt_4=',dT1_dt_4,'+/-',error_dT2_dt_4,'K/s')


