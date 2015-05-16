import os.path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.stats import sem
from uncertainties import ufloat
from uncertainties import unumpy as unp

alpha, M1, M2, M3, M4, y =np.genfromtxt('../Werte/werte.txt').T

U1=np.genfromtxt('../Werte/1.txt').T
U2=np.genfromtxt('../Werte/2.txt').T
U3=np.genfromtxt('../Werte/3.txt').T
U4=np.genfromtxt('../Werte/4.txt').T
U5=np.genfromtxt('../Werte/5.txt').T
U6=np.genfromtxt('../Werte/6.txt').T
U7=np.genfromtxt('../Werte/7.txt').T
U8=np.genfromtxt('../Werte/8.txt').T
U9=np.genfromtxt('../Werte/9.txt').T

alpha=np.radians(alpha)

U_1=ufloat(np.mean(U1),sem(U1))
U_2=ufloat(np.mean(U2),sem(U2))
U_3=ufloat(np.mean(U3),sem(U3))
U_4=ufloat(np.mean(U4),sem(U4))
U_5=ufloat(np.mean(U5),sem(U5))
U_6=ufloat(np.mean(U6),sem(U6))
U_7=ufloat(np.mean(U7),sem(U7))
U_8=ufloat(np.mean(U8),sem(U8))
U_9=ufloat(np.mean(U9),sem(U9))

#
##
### Mittelwerte und Fehler 
####

print('Mittelwerte und Fehler der Spannungen pro Winkel')
print(U_1)
print(U_2)
print(U_3)
print(U_4)
print(U_5)
print(U_6)
print(U_7)
print(U_8)
print(U_9)



#
##
### Berechnung der Inklination
####


#Maxima liegen bei U4 und U8, Minima bei U2 und U6
#Berechnung des Inklinationswinkels
##HIER FEHLT NOCH DIE FEHLERRECHNUNG
#IW1_=U_2/U_4
#IW2_=U_6/U_8
#print("")
#print(unp.arccos(IW1_)*180/np.pi)
#print(unp.arccos(IW2_)*180/np.pi)
#IW1=38.11760355
#IW2=24.07947132
#print('')
#print('Inklinationswinkel nach Arnes Formel fuer geteilte Messung')
#print(IW1)
#print(IW2)
#print(IW1,'pm',IW1_err)
#print(IW2,'pm',IW2_err)



#
##
### Plots 
####


#1. Einfach alle Messpunkte aufgetragen gegen den Winkel
plt.xlim(-0.5,7)
plt.plot(alpha,M1,'rx',label='Messung 1')
plt.plot(alpha,M2,'yx',label='Messung 2')
plt.plot(alpha,M3,'gx',label='Messung 3')
plt.plot(alpha,M4,'kx',label='Messung 4')
plt.xlabel('Winkel /rad')
plt.ylabel('Spannung U')
plt.legend(loc="best")
plt.savefig("../Bilder/M1-4.pdf")
plt.show()
plt.close()


#2. Plot der Mittelwerte + Unsicherheiten aufgetragen gegen den Winkel
plt.xlim(-0.5,7)
plt.errorbar(np.radians(0),unp.nominal_values(U_1),unp.std_devs(U_1),fmt='rx')
plt.errorbar(np.radians(45),unp.nominal_values(U_2),unp.std_devs(U_2),fmt='rx')
plt.errorbar(np.radians(90),unp.nominal_values(U_3),unp.std_devs(U_3),fmt='rx')
plt.errorbar(np.radians(135),unp.nominal_values(U_4),unp.std_devs(U_4),fmt='rx') 
plt.errorbar(np.radians(180),unp.nominal_values(U_5),unp.std_devs(U_5),fmt='rx')
plt.errorbar(np.radians(225),unp.nominal_values(U_6),unp.std_devs(U_6),fmt='rx')
plt.errorbar(np.radians(270),unp.nominal_values(U_7),unp.std_devs(U_7),fmt='rx')
plt.errorbar(np.radians(315),unp.nominal_values(U_8),unp.std_devs(U_8),fmt='rx')
plt.errorbar(np.radians(360),unp.nominal_values(U_9),unp.std_devs(U_9),fmt='rx')
plt.xlabel('Winkel /rad')
plt.ylabel('Spannung U')
#plt.legend(loc="best")
plt.savefig("../Bilder/MW.pdf")
plt.show()
plt.close()


#3. Plot der Regression; 1. Teil

def f(alpha, a, b, c, d):
	return a*np.cos(d*alpha+b)+c
params, cov = curve_fit(f,alpha[0:5], y[0:5] , maxfev=8000000, p0=[1.92172884017,3.16178588224,32,1])
errors = np.sqrt(np.diag(cov))
print('')

max1=np.max(f(np.linspace(0,7,10000),*params))
print('Maximum1')
print(max1)
min1=np.min(f(np.linspace(0,7,10000),*params))
print('Minimum1')
print(min1)
print('')

print('a =', params[0], 'pm', errors[0])
print('b =', params[1], 'pm', errors[1])
print('c =', params[2], 'pm', errors[2])
print('d =', params[3], 'pm', errors[3])

alpha_plot = np.linspace(-1, 7)
plt.xlim(-0.5,7)
plt.errorbar(np.radians(0),unp.nominal_values(U_1),unp.std_devs(U_1),fmt='kx')
plt.errorbar(np.radians(45),unp.nominal_values(U_2),unp.std_devs(U_2),fmt='kx')
plt.errorbar(np.radians(90),unp.nominal_values(U_3),unp.std_devs(U_3),fmt='kx')
plt.errorbar(np.radians(135),unp.nominal_values(U_4),unp.std_devs(U_4),fmt='kx') 
plt.errorbar(np.radians(180),unp.nominal_values(U_5),unp.std_devs(U_5),fmt='rx',label="Ausgewerte Daten")
plt.errorbar(np.radians(225),unp.nominal_values(U_6),unp.std_devs(U_6),fmt='rx',label="Nichtausgewerte Daten")
plt.errorbar(np.radians(270),unp.nominal_values(U_7),unp.std_devs(U_7),fmt='rx')
plt.errorbar(np.radians(315),unp.nominal_values(U_8),unp.std_devs(U_8),fmt='rx')
plt.errorbar(np.radians(360),unp.nominal_values(U_9),unp.std_devs(U_9),fmt='rx')
plt.plot(alpha_plot, f(alpha_plot, *params), 'b-', label='linearer Fit')
plt.legend(loc="best")
plt.show()
plt.savefig('../Bilder/plot1.pdf')
plt.close()


#3. Plot der Regression; 2. Teil

params, cov = curve_fit(f,alpha[4:9], y[4:9] , maxfev=8000000, p0=[1.92172884017,3.16178588224,32,1])
errors = np.sqrt(np.diag(cov))

max2=np.max(f(np.linspace(0,7,10000),*params))
print('')
print('Maximum2')
print(max2)
min2=np.min(f(np.linspace(0,7,10000),*params))
print('Minimum2')
print(min2)
print('')
print('a =', params[0], 'pm', errors[0])
print('b =', params[1], 'pm', errors[1])
print('c =', params[2], 'pm', errors[2])
print('d =', params[3], 'pm', errors[3])


plt.errorbar(np.radians(0),unp.nominal_values(U_1),unp.std_devs(U_1),fmt='rx')
plt.errorbar(np.radians(45),unp.nominal_values(U_2),unp.std_devs(U_2),fmt='rx')
plt.errorbar(np.radians(90),unp.nominal_values(U_3),unp.std_devs(U_3),fmt='rx')
plt.errorbar(np.radians(135),unp.nominal_values(U_4),unp.std_devs(U_4),fmt='rx',label="Nichtausgewerte Daten") 
plt.errorbar(np.radians(180),unp.nominal_values(U_5),unp.std_devs(U_5),fmt='kx')
plt.errorbar(np.radians(225),unp.nominal_values(U_6),unp.std_devs(U_6),fmt='kx',label="Ausgewerte Daten")
plt.errorbar(np.radians(270),unp.nominal_values(U_7),unp.std_devs(U_7),fmt='kx')
plt.errorbar(np.radians(315),unp.nominal_values(U_8),unp.std_devs(U_8),fmt='kx')
plt.errorbar(np.radians(360),unp.nominal_values(U_9),unp.std_devs(U_9),fmt='kx')
plt.plot(alpha_plot, f(alpha_plot, *params), 'b-', label='linearer Fit')
plt.legend(loc="best")
plt.show()
plt.savefig('../Bilder/plot2.pdf')
plt.close()

IW1=unp.arccos(min1/max1)*180/np.pi
IW2=unp.arccos(min2/max2)*180/np.pi

print('')
print('IW1=',IW1)
print('IW2=',IW2)

