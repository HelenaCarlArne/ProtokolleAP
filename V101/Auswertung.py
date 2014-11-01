
import numpy as np 									                                                    #### Header
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
#import uncertainties as unp
from uncertainties import ufloat
from scipy.stats import sem



# Bestimmung der Winkelrichtgroesse D gemessen im Abstand r=9.965 cm fuer alle Auslenkungen             #### Berechnung und Eingabe

F = np.genfromtxt('Werte/Werte_F_M1.txt').T                                                  
Phi = np.genfromtxt('Werte/Werte_Phi_M1.txt').T
Phi_Bogenmass = np.genfromtxt('Werte/Werte_Phi_Bogenmass_M1.txt').T
r = 0.09965 #m; r = const.
D = (F*r)/Phi_Bogenmass
D_neu = ufloat(np.mean(D),sem(D))    
print('Die Winkelrichtgroesse D betraegt:',D_neu,'Nm')


# Lineare Regression 

def linregress(x, y):
    N = len(y) # Annahme: len(x) == len(y), sonst kommt waehrend der Rechnung eine Fehlermeldung
    Delta = N*np.sum(x**2)-(np.sum(x))**2

    A = (N*np.sum(x*y)-np.sum(x)*np.sum(y))/Delta
    B = (np.sum(x**2) * np.sum(y) - np.sum(x) * np.sum(x * y)) / Delta

    sigma_y = np.sqrt(np.sum((y - A * x - B)**2) / (N - 2))

    A_error = sigma_y * np.sqrt(N / Delta)
    B_error = sigma_y * np.sqrt(np.sum(x**2) / Delta)
    b=ufloat(B,B_error)
    I=D_neu*b/(4*np.pi*np.pi)
    plt.plot(x,y,"rx")
    #x_plot=np.linspace(0,10)
    #plt.plot(x_plot**2,x_plot**2*A+B)
    plt.show()
    print (A, A_error, B, B_error, I)


###Bestimmung des Eigentraegheitsmoments I_D der Drillachse

A = np.genfromtxt('Werte/Werte_A_M2.txt').T							 
T_D = np.genfromtxt('Werte/Werte_T_M2.txt').T
m_1 = np.genfromtxt('Werte/Werte_Masse1_M2.txt').T
m_2 = np.genfromtxt('Werte/Werte_Masse2_M2.txt').T


m_1 = ufloat(np.mean(m_1),sem(m_1)) 						
m_2 = ufloat(np.mean(m_2),sem(m_2))						

I = (T_D)**2*D/(4*(np.pi)**2)							
I_D = ufloat(np.mean(I),sem(I))						        

#print('Zeit Eigentraegheit',T_D,'s')
print("")
print('Das Gewicht der Masse m_1 ist',m_1,'kg')
print('Das Gewicht der Masse m_2 ist',m_2,'kg')
print('Das Eigentraegheitsmoment I der Drillachse ist',I_D,'kgm**2')


#Bestimmung des Eigentraegheitsmoments I_Z des Styroporzylinders
		
T_Z = np.genfromtxt('Werte/Werte_Zylinderzeit_M3_.txt').T				#Einlesen der Werte
d_Z = np.genfromtxt('Werte/Werte_Zylinderdurchmesser_M3_.txt').T
h_Z = np.genfromtxt('Werte/Werte_Zylinderhohe_M3_.txt').T
m_Z = np.genfromtxt('Werte/Werte_Zylindermasse_M3_.txt').T


T_Z_neu = ufloat(np.mean(T_Z),sem(T_Z)) 						#Mittelwert und Fehler der Periodendauer
d_Z_neu = ufloat(np.mean(d_Z),sem(d_Z)) 						#Mittelwert und Fehler des Durchmessers
m_Z_neu = ufloat(np.mean(m_Z),sem(m_Z)) 						#Mittelwert und Fehler der Masse
h_Z_neu = ufloat(np.mean(h_Z),sem(h_Z))							##Mittelwert und Fehler der Hoehe


I_Z = T_Z**2*D/(4*(np.pi)**2)
I_Z_Messung = ufloat(np.mean(I_Z),sem(I_Z))
V_Z= np.pi*(d_Z_neu/2)**2*h_Z_neu
m_Z=V_Z*1050
print("")
print('Die Schwingungsdauer des Zylinders T_Z_neu ist',T_Z_neu,'s')
print('Der Durchmesser des Zylinders d_Z_neu ist',d_Z_neu,'m')
print('Die Hoehe des Zylinders h_Z_neu ist',h_Z_neu,'m')
print('Die gemessene Masse des Zylinders m_Z_neu ist',m_Z_neu,'kg')
print('Das Traegheitsmoment des Zylinders ist ',I_Z_Messung,'kgm**2')
print("Das aussere Volumen des Zylinders ist ",V_Z,"m^3")
print("Die errechnete Masse des Voll-Zylinders ist ",m_Z,"kg")

#Bestimmung des Eigentraegheitsmoments I_K der Kugel
		
T_K = np.genfromtxt('Werte/Werte_Kugelzeit_M4_.txt').T				#Einlesen der Werte
d_K = np.genfromtxt('Werte/Werte_Kugeldurchmesser_M4_.txt').T
m_K = np.genfromtxt('Werte/Werte_Kugelmasse_M4_.txt').T


T_K_neu = ufloat(np.mean(T_K),sem(T_K)) 						#Mittelwert und Fehler der Periodendauer
d_K_neu = ufloat(np.mean(d_K),sem(d_K)) 						#Mittelwert und Fehler des Durchmessers
m_K_neu = ufloat(np.mean(m_K),sem(m_K)) 						#Mittelwert und Fehler der Masse

I_K = T_K**2*D/(4*(np.pi)**2)
I_K_Messung = ufloat(np.mean(I_K),sem(I_K))	
print("")						
print('Die Schwingungsdauer der Kugel T_K_neu ist',T_K_neu,'s')
print('Der Durchmesser derKugel d_K_neu ist',d_K_neu,'m')
print('Die Masse der Kugel m_K_neu ist',m_K_neu,'kg')
print('Das Traegheitsmoment der Kugel ist ',I_K_Messung,'kgm**2')


#Berechnung der theoretischen Werte vom Traegheitsmoment der Kugel und des Zylinders

I_K_Theoretisch = (2/5)*m_K_neu*(d_K_neu/2)**2
I_Z_Theoretisch = (1/2)*m_Z_neu*(d_Z_neu/2)**2
print("")
print('Der theoretisch errechnete Wert fuer das Traegheitsmoment der Kugel ist',I_K_Theoretisch,'kgm**2')
print('Der theoretisch errechnete Wert fuer das Traegheitsmoment des Zylinders ist',I_Z_Theoretisch,'kgm**2')

#Bestimmung des Eigentraegheitsmoment I_P der Puppe

A_d_l = np.genfromtxt('Werte/Werte_Armdurchmesser_l_M5.txt').T		#Abmessungen der Puppe
A_d_r = np.genfromtxt('Werte/Werte_Armdurchmesser_r_M5.txt').T
A_h_l = np.genfromtxt('Werte/Werte_Armlange_l_M5.txt').T
A_h_r = np.genfromtxt('Werte/Werte_Armlange_r_M5.txt').T
B_d_l = np.genfromtxt('Werte/Werte_Beindurchmesser_l_M5.txt').T
B_d_r = np.genfromtxt('Werte/Werte_Beindurchmesser_r_M5.txt').T
B_h_l = np.genfromtxt('Werte/Werte_Beinlange_l_M5.txt').T
B_h_r = np.genfromtxt('Werte/Werte_Beinlange_r_M5.txt').T
R_d = np.genfromtxt('Werte/Werte_Rumpfdurchmesser_M5.txt').T
R_h = np.genfromtxt('Werte/Werte_Rumpflange_M5.txt').T
K_d = np.genfromtxt('Werte/Werte_Kopfdurchmesser_M5.txt').T
m_P = np.genfromtxt('Werte/Werte_Masse_M5.txt').T

T_1 = np.genfromtxt('Werte/Werte_Puppenzeit1_M5.txt').T			#VORSICHT: T_1 und T_2 sind fuer 5 Schwingungen!
T_2 = np.genfromtxt('Werte/Werte_Puppenzeit2_M5.txt').T


A_d_l =  ufloat(np.mean(A_d_l),sem(A_d_l))
A_d_r =  ufloat(np.mean(A_d_r),sem(A_d_r))
A_h_l =  ufloat(np.mean(A_h_l),sem(A_h_l))
A_h_r =  ufloat(np.mean(A_h_r),sem(A_h_r))
B_d_l =  ufloat(np.mean(B_d_l),sem(B_d_l))
B_d_r =  ufloat(np.mean(B_d_r),sem(B_d_r))
B_h_l =  ufloat(np.mean(B_h_l),sem(B_h_l))
B_h_r =  ufloat(np.mean(B_h_r),sem(B_h_r))
R_d =  ufloat(np.mean(R_d),sem(R_d))
R_h =  ufloat(np.mean(R_h),sem(R_h))
K_d =  ufloat(np.mean(K_d),sem(K_d))
m_P =  ufloat(np.mean(m_P),sem(m_P))
T_1_neu =  ufloat(np.mean(T_1/5),sem(T_1/5))
T_2_neu =  ufloat(np.mean(T_2/5),sem(T_2/5))

print("")
print('Der Durchmesser des linken Arms ist',A_d_l,'m') 				#Armabmessungen
print('Der Durchmesser des rechten Arms ist',A_d_r,'m')
print('Die Laenge des linken Arms ist',A_h_l,'m')
print('Die Laenge des rechten Arms ist',A_h_r,'m')
print("")
print('Der Durchmesser des linken Beins ist',B_d_l,'m')				#Beinabmessungen
print('Der Durchmesser des rechten Beins ist',B_d_r,'m')
print('Die Laenge des linken Beins ist',B_h_l,'m')
print('Die Laenge des rechten Beins ist',B_h_r,'m')
print("")
print('Der Durchmesser des Rumpfes ist',R_d,'m')				#Rumpf und Kopf
print('Die Laenge des Rumpfes ist',R_h,'m')
print('Der Durchmesser des Kopfes ist',K_d,'m')
print("")
print('Die Masse der Puppe ist',m_P,'kg')					#Masse; zeit fuer Position 1 und 2
print('Die Zeit fuer Position 1 ist',T_1_neu,'s')
print('Die Zeit fuer Position 2 ist',T_2_neu,'s')



I_1= (T_1/5)**2*D/(4*(np.pi)**2)						#Tabelle fuer I_1 und I_2
I_2= (T_2/5)**2*D/(4*(np.pi)**2)
I_1_neu=ufloat(np.mean(I_1),sem(I_1))						#Mittelwert und Fehler von I_1 und I_2
I_2_neu=ufloat(np.mean(I_2),sem(I_2))
print("")
print('Das Traegheitsmoment in Position 1 ist',I_1_neu,'kgm**2')
print('Das Traegheitsmoment in Position 2 ist',I_2_neu,'kgm**2')
print("")
linregress(A**2,(T_D)**2)

