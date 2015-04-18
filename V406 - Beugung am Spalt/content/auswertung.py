import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy.constants as const
import uncertainties
from uncertainties import ufloat
import peakdetect

#################################
#	ALLGEMEINE DATEN	#
#################################

d_0 =25e-3    						#Skalenmitte
L=1.203							#Abstand Blende - Schirm
I_0=1.2e-9						#Dunkelstrom
b_m1=2e-3/21e-3 					#Einfachspalt (1) 
b_h1=0.075e-3 
b_m2=3e-3/7e-3						#Einfachspalt (2)		
b_h2=0.4e-3  				
b_m=1e-3/7e-3						#Doppelspalt (3)
g_m =11e-3/21e-3
b_h=0.1e-3 
g_h=0.4e-3 	
l=633e-9 						#Wellenlänge
							#Mikroskopeichung: 21 Skalenstriche entsprechen 1 mm

#################################
#	DATEN EINLESEN		#
#################################
#d, I_1, I_2, I_3 = np.genfromtxt('Werte.txt').T
d = np.genfromtxt('Werte_d.txt').T
I_1 = np.genfromtxt('Werte_I1.txt').T
I_2 = np.genfromtxt('Werte_I2.txt').T
I_3 = np.genfromtxt('Werte_I3.txt').T

#################################
#	BEUGUNGSWINKEL PHI	#
#################################

phi = (d-d_0)/L

#################################
#	BERECHNUNG		#
#################################

#Dunkelstrom von Intensität subtrahieren
I1=I_1-I_0
I2=I_2-I_0
I3=I_3-I_0

#################################
#	FUNKTIONEN EINZELSPALT	#
#################################
def f(phi, A, b, c):
	return (A*l*(1/(np.pi*(phi+c)))*(np.sin((np.pi*b*(phi+c))/l)))**2


#################################
#	FUNKTION DOPPELSPALT	#
#################################
def F(phi, A, b, c, s):
	return (np.cos(np.pi*s*(phi+c)/l)*A*l*(1/(np.pi*(phi+c)))*(np.sin((np.pi*b*(phi+c))/l)))**2

#################################
#	PLOT MESSUNG 1		#
#################################
params, covariance = curve_fit(f, phi, I1, maxfev=8000000, p0=[14.0039970815,7.61269381475e-05,-0.000185424535465])
errors = np.sqrt(np.diag(covariance))

print('A1 =', params[0], '±', errors[0])
print('b1 =', params[1], '±', errors[1])
print('c1 =', params[2], '±', errors[2])
phi_plot = np.linspace(-0.050, 0.050, 1000)

plt.plot(phi, I1, 'rx', label="Messdaten")
plt.plot(phi_plot, f(phi_plot, *params), 'b-', label='nichtlinearer Fit')
plt.xlim(-0.022,0.022)	
plt.ylim(-100e-9,1300e-9)
plt.xlabel('Beugungswinkel $\phi$ / rad')
plt.ylabel('Strom $I$ /nA')
plt.yticks([0, 0.0000002, 0.0000004, 0.0000006, 0.0000008, 0.0000010, 0.0000012],[0, 200, 400, 600, 800, 1000, 1200])
plt.legend(loc="best")
plt.tight_layout
plt.savefig('../Bilder/Messung1.pdf')
plt.show()
#################################
#	PLOT MESSUNG 2		#
#################################
params, covariance = curve_fit(f, phi, I2, maxfev=8000000, p0=[15.2114879734,0.000371268553435,1.64888855298e-05])
errors = np.sqrt(np.diag(covariance))

print('A2 =', params[0], '±', errors[0])
print('b2 =', params[1], '±', errors[1])
print('c2 =', params[2], '±', errors[2])

phi_plot = np.linspace(-0.050, 0.050, 1000)

plt.plot(phi, I2, 'rx', label="Messdaten")
plt.plot(phi_plot, f(phi_plot, *params), 'b-', label='nichtlinearer Fit')
plt.xlim(-0.022,0.022)	
plt.ylim(-0.000004,0.000035)
plt.yticks([0, 0.00001, 0.00002, 0.00003],[0, 10, 20, 30])
plt.xlabel('Beugungswinkel $\phi$ / rad')
plt.ylabel('Strom $I$ /kA')
plt.legend(loc="best")
plt.tight_layout
plt.savefig('../Bilder/Messung2.pdf')
plt.show()

#################################
#	PLOT MESSUNG 3		#
#################################
params, covariance = curve_fit(F, phi, I3, maxfev=8000000, p0=[17.6542781097,9.88662064934e-05,-0.00024542086241,0.000490180094003])
errors = np.sqrt(np.diag(covariance))

print('A3 =', params[0], '±', errors[0])
print('b3 =', params[1], '±', errors[1])
print('c3 =', params[2], '±', errors[2])
print('s3 =', params[3], '±', errors[3])

phi_plot = np.linspace(-0.05, 0.05, 1000)



#################################
#    EINHUELLENDE DOPPELSPALT   #
#################################

x_array=np.array([-112,-100,-88,-75,-48,-35,-22,-10,2.518,15,28,41,53,81,93,105])
x_array=x_array*(10**(-4))
parms,cov = curve_fit(f,x_array,F(x_array, *params),p0=[0.8,0.0001,-0.000000001])
print(parms)

#maxds,minds = peakdetect.peakdetect(F(phi,*params),phi,lookahead=5,delta=0)
#maxds=np.array(maxds).T
#minds=np.array(minds).T

#print(maxds)
#print(minds)
##########################################################################################################################
'''
def f(phi, A, b, c):
	return (A*l*(1/(np.pi*(phi+c)))*(np.sin((np.pi*b*(phi+c))/l)))**2
(A, b, c), covariance = curve_fit(f, phi, I3, maxfev=8000000, p0=[17.6542827897,9.88662064934e-05,-0.00024542086241])
phi_plot = np.linspace(-0.050, 0.050, 1000)

plt.plot(phi, I3, 'rx', label="Messdaten")
plt.plot(phi_plot, f(phi_plot, A, b, c), 'b-', label='nichtlinearer Fit')
plt.xlim(-0.022,0.022)	
plt.ylim(-100e-9,1300e-9)
plt.xlabel('Beugungswinkel $\phi$ / rad')
plt.ylabel('Strom $I$ /nA')
plt.yticks([0, 0.0000002, 0.0000004, 0.0000006, 0.0000008, 0.0000010, 0.0000012],[0, 200, 400, 600, 800, 1000, 1200])
plt.legend(loc="best")
#plt.savefig('../Bilder/Messung1.pdf')

d=np.linspace(-0.03, 0.03,10000)
b=0.000072
A=20
def f(a):
    return A*b**2*((l)/(np.pi*b*np.sin(d)))**2*(np.sin((np.pi*b*np.sin(d))/(l)))**2


plt.plot(d, f(d), 'b-', label='Einzelspalt')

'''
##############################################################################################################################
"""plt.plot(maxds[0],maxds[1],"k+")
plt.plot(minds[0],minds[1],"k+")"""
plt.plot(phi, I3, 'rx', label="Messdaten")
plt.plot(phi_plot, F(phi_plot, *params), 'b-', label='nichtlinearer Fit')
plt.ylim(-0.0000001,0.0000035)
plt.xlim(-0.022,0.022)
plt.yticks([0, 0.0000005, 0.0000010, 0.0000015, 0.0000020, 0.0000025, 0.0000030, 0.0000035],[0, 50, 100, 150, 200, 250, 300, 350])
plt.xlabel('Beugungswinkel $\phi$ / rad')
plt.ylabel('Strom $I$ /nA')
plt.plot(x_array,F(x_array, *params),"g+")
plt.plot(phi_plot,f(phi_plot, *parms),"g-", label="Einhüllende")
plt.legend(loc="best")
plt.tight_layout
plt.savefig('../Bilder/Messung3.pdf')
plt.show()



