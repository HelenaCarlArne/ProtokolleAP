#
##
###
#### Header
###
##
#

import numpy as np 										
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
from uncertainties import ufloat
from scipy.stats import sem
from uncertainties.unumpy import (nominal_values as noms,std_devs as stds)
plt.rc('text', usetex=True)

def linregress(x, y):
    N = len(y) # Annahme: len(x) == len(y), sonst kommt während der Rechnung eine Fehlermeldung
    Delta = N * np.sum(x**2) - (np.sum(x))**2

    A = (N * np.sum(x * y) - np.sum(x) * np.sum(y)) / Delta
    B = (np.sum(x**2) * np.sum(y) - np.sum(x) * np.sum(x * y)) / Delta

    sigma_y = np.sqrt(np.sum((y - A * x - B)**2) / (N - 2))

    A_error = sigma_y * np.sqrt(N / Delta)
    B_error = sigma_y * np.sqrt(np.sum(x**2) / Delta)

    return A, A_error, B, B_error

def lin(x,m_lin,b_lin):
	return m_lin*x+b_lin


# plt.rc('font', family='serif')
#
##
###
#### Laden der Daten
###
##
#

length1,length2,diameter=np.genfromtxt("./Drahtgeo.txt").T
zeit_schubmodul=np.genfromtxt("./Schubmodul.txt").T
strom_dipolmoment,zeit_dipolmoment=np.genfromtxt("./Dipolmoment.txt").T
zeit_erdmagnet=np.genfromtxt("./Erdmagnet.txt").T
zeit_dipolmoment_01,zeit_dipolmoment_02,zeit_dipolmoment_04,zeit_dipolmoment_06,zeit_dipolmoment_08=np.split(zeit_dipolmoment,5)
# print(zeit_dipolmoment_01,zeit_dipolmoment_02,zeit_dipolmoment_04,zeit_dipolmoment_06,zeit_dipolmoment_08)
# print(length1)
# print(length2)
# print(diameter)
# print(zeit_erdmagnet)
# print(zeit_dipolmoment)
# print(strom_dipolmoment)
# print(zeit_schubmod)	


# Kugelmasse m_k = 512.2 g \pm 0.04 %
m_K = ufloat(0.5122,0.0004*0.5122)
# Kugeldurchmesser 2R_k = 50.76 \pm 0.007 %
r_K = ufloat(0.05076/2,0.00007*0.05076/2)
# Trägheitsmoment der Kugelhalterung theta_KH = 22.5 gcm²
Theta_KH= 22.5*10**(-7)
# Windungszahl der Helmholtzspule N = 390
N_HS= 390
# Radius der Helmholtzspule r_HS = 78 mm
r_HS = 0.078
# Kugelträgheit
Theta_K= (2/5)*(m_K*r_K**2)

"""
Das Feld einer Helmholtzspule ist mit der Formel gegeben:


B = (µ_0*8)/(125)^(1/2) N_HS/(r_HS) * I = B_VF * I

B_VF:	Vorfaktor vor I
"""
B_VF = (8*4*np.pi*10**(-7))*N_HS/(((125)**(1/2))*r_HS)

#
##
###
#### Berechnung der Daten
###
##
#

length1=ufloat(np.mean(length1),sem(length1))
length2=ufloat(np.mean(length2),sem(length2))
length=length1+length2
diameter=ufloat(np.mean(diameter),sem(diameter))
r_D=diameter/2
zeit_dipolmoment_01=ufloat(np.mean(zeit_dipolmoment_01),sem(zeit_dipolmoment_01))
zeit_dipolmoment_02=ufloat(np.mean(zeit_dipolmoment_02),sem(zeit_dipolmoment_02))
zeit_dipolmoment_04=ufloat(np.mean(zeit_dipolmoment_04),sem(zeit_dipolmoment_04))
zeit_dipolmoment_06=ufloat(np.mean(zeit_dipolmoment_06),sem(zeit_dipolmoment_06))
zeit_dipolmoment_08=ufloat(np.mean(zeit_dipolmoment_08),sem(zeit_dipolmoment_08))
zeit_schubmodul=	ufloat(np.mean(zeit_schubmodul),sem(zeit_schubmodul))
zeit_erdmagnet=		ufloat(np.mean(zeit_erdmagnet),sem(zeit_erdmagnet))
# print(zeit_dipolmoment_01,zeit_dipolmoment_02,zeit_dipolmoment_04,zeit_dipolmoment_06,zeit_dipolmoment_08)
# print(zeit_schubmodul)
# print(zeit_erdmagnet)
# print(length)
# print(r_D)


#
##
###
#### Beginn der Auswertung
###
##
#


"""
Teil 1: Der Schubmodul
Die Formel für das Schubmodul ist im Skript Nr. 14:


G = 16/5 (pi*[m_K]*[r_K]^2*L) / ([T]^2*[r_D]^4) = Theta_K* 8 (pi*L) / ([T]^2*[r_D]^4)


G: 	der Schubmodul, wobei die Trägheit der Halterung berücksichtigt wurde
D:	die Richtgröße
r_D: Radius des Drahtes
r_K: Radius der Kugel
Theta_K: Trägheit der Kugel
	= 2/5 m_K*r_K


"""
# length = 0.604				# Werte von Team Unna zur Kontrolle
# zeit_schubmodul= 18.557
# r_D= 98.1*10**(-6)
G =(Theta_K+Theta_KH)*8*(np.pi*length)/(zeit_schubmodul**2*(r_D)**4)
D =np.pi*G*(r_D)**4/(2*length)
# print(G)
# print(D)

"""
Teil 2: Das Dipolmoment
Die Formel für das Dipolmoment ist im Skript nach Nr. 19:


B = (4*pi^2 [Theta_K]/[m]) (1/[zeit_dipolmoment]^2) - ([D]/[m])
		=m 					=x 						=b

B: 		Das Feld der Helmholtzspule
D: 		Richtgröße
m:		Dipolmoment
Theta_K: 	Trägheit der Kugel
	= 2/5 m_K*r_K

Aus der Linearisierung ergibt sich das Dipolmoment.
In dem Plot wurde einiges angepasst.

"""

B1=B_VF*0.1
B2=B_VF*0.2
B4=B_VF*0.4
B6=B_VF*0.6
B8=B_VF*0.8
B=B_VF*np.array([0.1,0.2,0.4,0.6,0.8])
Zeit_dipolmoment=np.array([zeit_dipolmoment_01,zeit_dipolmoment_02,zeit_dipolmoment_04,zeit_dipolmoment_06,zeit_dipolmoment_08])
# print(B1)
# print(B2)
# print(B4)
# print(B6)
# print(B8)
# print(B)
# print(Zeit_dipolmoment)
xplot=np.arange(0,5)
params,helena=curve_fit(lin,noms(1/(Zeit_dipolmoment))**2,B)
plt.plot(xplot,lin(noms(xplot)**2,*params), label=r"Fit")
plt.xlabel(r'$\displaystyle \frac{1}{\,T_s} \left[\frac{1}{\,\mathrm{s^2}}\right]$',fontsize=12)
plt.ylabel(r'$\displaystyle B \,\mathrm{[mT]}$',fontsize=12)
plt.xlim(0,0.05)
plt.xticks([0,0.01,0.02,0.03,0.04,0.05],[0,0.01,0.02,0.03,0.04,0.05])
plt.ylim(-0.001,0.005,0.001)
plt.yticks([-0.001,0,0.001,0.002,0.003,0.004,0.005],[-1,0,1,2,3,4,5])
plt.errorbar(noms(1/(Zeit_dipolmoment))**2,B,yerr=stds(1/(Zeit_dipolmoment))**2,fmt="rx",label=r"Messdaten")
plt.legend(loc="best")
# plt.savefig("../Bilder/Magnetfeld.pdf")
# plt.show()
a,aerr,b,berr=linregress(noms(1/(Zeit_dipolmoment))**2,B)
a_uni = ufloat(a,aerr)

m = 4*np.pi**2*(Theta_K+Theta_KH)/a_uni

"""
Teil 3: Das Erdmagnetfeld
Die Formel für das Erdmagnetfeld ist im Skript nach Nr. 19:


B = (4*pi^2 [Theta_K]/[m]) (1/[zeit_dipolmoment]^2) - ([D]/[m]),
sofern m,D und Theta_K bekannt sind.


B: 		Erdmagnetfeld
D: 		Richtgröße
m:		Dipolmoment
Theta_K: 	Trägheit der Kugel
	= 2/5 m_K*r_K

"""
B = (4*np.pi**2*(Theta_K+Theta_KH))/m*(1/zeit_erdmagnet**2)-(D/m)

"""
Teil 4: Der Rest der Formeln

E=2G(\mu+1) &\quad\text{und} &&\quad E=3(1-2\mu)Q
"""
E= 190300000000
mu=E/(2*G)-1
Q=E/(3-6*mu)

#
##
###
#### Ausgabe
###
##
#

print(
"""
############################################################################################################
Abmessungen

Laenge des Seils:
L=	{}

Durchmesser des Seils:
r_D={}

Gesamttraegheit:
Theta= {}

Masse der Kugel:
m_K={}

Radius der Kugel:
r_K={}
############################################################################################################
Zeiten

Zeit fuer das Schubmodul:
T= 	{}

Zeit fuer das Magnetfeld der Erde:
T=	{}

Zeit fuer das Dipolmoment (aufsteigende Stromstaerke):
T=	{}

############################################################################################################
Auswertung

Der Schubmodul betraegt:
G=	{}

Die Winkelrichtgroesse D betraegt:
D = {}

Die Regressionsparameter m, b sind:
a=	{}+/-{}
b=	{}+/-{}

Daraus folgt fuer das magnetische Moment:
m= {}

Das Erdmagnetfeld ist:
E=	{}

############################################################################################################
Modulen 
Der Schubmodul betraegt:
G=	{}

Der Elastizitaetsmodul betraegt:
E=	{}

Der Kompressionsmodul betraegt:
Q=	{}

Die Poissonsche Querkontraktionszahl betraegt:
mu=	{}


""".format(length,2*r_D,(Theta_K+Theta_KH),m_K,r_K,zeit_schubmodul,zeit_erdmagnet,Zeit_dipolmoment,G,D,a,aerr,b,berr,m,B,G,E,Q,mu))
