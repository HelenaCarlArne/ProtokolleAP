#### Header
import numpy as np 										
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
from uncertainties import ufloat
from scipy.stats import sem

#### Laden der Daten
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
r_K = ufloat(0.005076/2,0.00007*0.005076/2)
# Trägheitsmoment der Kugelhalterung theta_KH = 22.5 gcm²
T_KH= 2.25
# Windungszahl der Helmholtzspule N = 390
N_HS= 390
# Radius der Helmholtzspule r_HS = 78 mm
r_HS = 0.078

"""
Das Feld einer Helmholtzspule ist mit der Formel gegeben:

B = µ_0*N_HS/((5/4)^(3/2)*r_HS) * I = B_VF * I

B_VF:	Vorfaktor vor I
"""
B_VF = (4*np.pi*10**(-7))*N_HS/(((5/4)**(3/2))*r_HS)
#### Berechnung der Daten
length1=ufloat(np.mean(length1),sem(length1))
length2=ufloat(np.mean(length2),sem(length2))
length=length1+length2
diameter=ufloat(np.mean(diameter),sem(diameter))
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



#### Beginn der Auswertung
"""
Die Formel für das Schubmodul ist im Skript Nr. 14:

G = 16/5 (pi*[m_K]*[r_K]^2*L) / ([T]^2*[r_D]^4) = T_K* 8 (pi*L) / ([T]^2*[r_D]^4)

r_D: Radius des Drahtes
r_K: Radius der Kugel
T_K: Trägheit der Kugel
	= 2/5 m_K*r_K

G 		ist das rohe Schubmodul
G_KH 	ist das Schubmodul, wobei die Trägheit der Halterung berücksichtigt wurde
D 		ist die Richtgröße
D_KH	ist die Richtgröße mit Berücksichtigung

"""
# G =16/5*(np.pi*m_K*(r_K)**2*length)/(zeit_schubmodul**2*(diameter/2)**4)
G_KH =(T_KH+(2/5)*m_K*(r_K)**2)*8*(np.pi*length)/(zeit_schubmodul**2*(diameter/2)**4)
# D=np.pi*G*(diameter/2)**4/(2*length)
D_KH=np.pi*G_KH*(diameter/2)**4/(2*length)
# print(G)
print(G_KH)

"""
Die Formel für das Dipolmoment ist im Skript nach Nr. 19:

m = 4*pi^2 [T_K]/[B][zeit_dipolmoment]^2 - [D]/[B]

B: Das Feld der Helmholtzspule
D: Richtgröße
T_K: Trägheit der Kugel
	= 2/5 m_K*r_K

m 		ist das Dipolmoment
m_KH	ist das Dipolmoment mit Brücksichtigung
B 		ist die Feldstärke der Helmholtzspule

"""
B1=B_VF*0.1
B2=B_VF*0.2
B4=B_VF*0.4
B6=B_VF*0.6
B8=B_VF*0.8

# m_1 = 4*np.pi**2*((2/5)*m_K*(r_K)**2)/(B1*(zeit_dipolmoment_01)**2) - D/B1
m_KH_1 = 4*np.pi**2*(T_KH+(2/5)*m_K*(r_K)**2)/(B1*(zeit_dipolmoment_01)**2) - D_KH/B1
# m_2 = 4*np.pi**2*((2/5)*m_K*(r_K)**2)/(B2*(zeit_dipolmoment_02)**2) - D/B2
m_KH_2 = 4*np.pi**2*(T_KH+(2/5)*m_K*(r_K)**2)/(B2*(zeit_dipolmoment_02)**2) - D_KH/B2
# m_4 = 4*np.pi**2*((2/5)*m_K*(r_K)**2)/(B4*(zeit_dipolmoment_04)**2) - D/B4
m_KH_4 = 4*np.pi**2*(T_KH+(2/5)*m_K*(r_K)**2)/(B4*(zeit_dipolmoment_04)**2) - D_KH/B4
# m_6 = 4*np.pi**2*((2/5)*m_K*(r_K)**2)/(B6*(zeit_dipolmoment_06)**2) - D/B6
m_KH_6 = 4*np.pi**2*(T_KH+(2/5)*m_K*(r_K)**2)/(B6*(zeit_dipolmoment_06)**2) - D_KH/B6
# m_8 = 4*np.pi**2*((2/5)*m_K*(r_K)**2)/(B8*(zeit_dipolmoment_08)**2) - D/B8
m_KH_8 = 4*np.pi**2*(T_KH+(2/5)*m_K*(r_K)**2)/(B8*(zeit_dipolmoment_08)**2) - D_KH/B8
