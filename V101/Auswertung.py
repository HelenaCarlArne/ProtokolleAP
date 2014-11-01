#
# Header
#

import numpy as np 									                                                    
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
from uncertainties import ufloat
from scipy.stats import sem


#
# Eingabe der Werte
#

#Messung 1, Richtwinkel
    r = 0.09965
    F = np.genfromtxt('Werte/Werte_F_M1.txt').T                                                  
    Phi = np.genfromtxt('Werte/Werte_Phi_M1.txt').T
    Phi_Bogenmass = np.genfromtxt('Werte/Werte_Phi_Bogenmass_M1.txt').T
#Messung 2, Eigentraegheit
    A_array = np.genfromtxt('Werte/Werte_A_M2.txt').T                          
    T_D_array = np.genfromtxt('Werte/Werte_T_M2.txt').T
    m_1_array = np.genfromtxt('Werte/Werte_Masse1_M2.txt').T
    m_2_array = np.genfromtxt('Werte/Werte_Masse2_M2.txt').T
#Messung 3, Zylinder
    T_Z_array = np.genfromtxt('Werte/Werte_Zylinderzeit_M3_.txt').T               
    d_Z_array = np.genfromtxt('Werte/Werte_Zylinderdurchmesser_M3_.txt').T
    h_Z_array = np.genfromtxt('Werte/Werte_Zylinderhohe_M3_.txt').T
    m_Z_array = np.genfromtxt('Werte/Werte_Zylindermasse_M3_.txt').T
#Messung 4, Kugel
    T_K_array = np.genfromtxt('Werte/Werte_Kugelzeit_M4_.txt').T              
    d_K_array = np.genfromtxt('Werte/Werte_Kugeldurchmesser_M4_.txt').T
    m_K_array = np.genfromtxt('Werte/Werte_Kugelmasse_M4_.txt').T
#Messung 5, Puppe
    A_d_l_array = np.genfromtxt('Werte/Werte_Armdurchmesser_l_M5.txt').T      
    A_d_r_array = np.genfromtxt('Werte/Werte_Armdurchmesser_r_M5.txt').T
    A_h_l_array = np.genfromtxt('Werte/Werte_Armlange_l_M5.txt').T
    A_h_r_array = np.genfromtxt('Werte/Werte_Armlange_r_M5.txt').T
    B_d_l_array = np.genfromtxt('Werte/Werte_Beindurchmesser_l_M5.txt').T
    B_d_r_array = np.genfromtxt('Werte/Werte_Beindurchmesser_r_M5.txt').T
    B_h_l_array = np.genfromtxt('Werte/Werte_Beinlange_l_M5.txt').T
    B_h_r_array = np.genfromtxt('Werte/Werte_Beinlange_r_M5.txt').T
    R_d_array = np.genfromtxt('Werte/Werte_Rumpfdurchmesser_M5.txt').T
    R_h_array = np.genfromtxt('Werte/Werte_Rumpflange_M5.txt').T
    K_d_array = np.genfromtxt('Werte/Werte_Kopfdurchmesser_M5.txt').T
    m_P_array = np.genfromtxt('Werte/Werte_Masse_M5.txt').T
    T_5Schwingungen_1_array = np.genfromtxt('Werte/Werte_Puppenzeit1_M5.txt').T              #VORSICHT: T_1 und T_2 sind fuer 1 Schwingungen!
    T_5Schwingungen_2_array = np.genfromtxt('Werte/Werte_Puppenzeit2_M5.txt').T
    T_1_array = T_5Schwingungen_1_array/5
    T_2_array = T_5Schwingungen_2_array/5


#
# Berechnungen
#
#Detached
    I_1 = I_A_l_1+I_A_r_1+I_B_l+I_B_r+I_K+I_R
    I_2 = I_A_l_2+I_A_r_2+I_B_l+I_B_r+I_K+I_R

#Messung 1, Richtwinkel
   
    D_array = (F*r)/Phi_Bogenmass
    D = ufloat(np.mean(D_array),sem(D_array))   
    m_1 = ufloat(np.mean(m_1_array),sem(m_1_array))                         
    m_2 = ufloat(np.mean(m_2_array),sem(m_2_array))                     
#Messung 2, Eigenträgheit
    I = (T_D_array)**2*D/(4*(np.pi)**2)                           
    I_D = ufloat(np.mean(I),sem(I))   

#Messung 3, Zylinder  
    T_Z = ufloat(np.mean(T_Z_array),sem(T_Z_array))                         
    d_Z = ufloat(np.mean(d_Z_array),sem(d_Z_array))                         
    m_Z = ufloat(np.mean(m_Z_array),sem(m_Z_array))     #Experiementelle Masse                     
    h_Z = ufloat(np.mean(h_Z_array),sem(h_Z_array)) 
    V_Z= np.pi*(d_Z/2)**2*h_Z   
    m_Z_Theorie=V_Z*1050                                #Masse, falls Styropor-Vollzylinder               
    I_Z_array = T_Z_array**2*D/(4*(np.pi)**2)           #Traegheit via Schwingdauer
    I_Z_Messung = ufloat(np.mean(I_Z_array),sem(I_Z_array))
    I_Z_Theoretisch = (1/2)*m_2*(d_Z_neu/2)**2          #Traegheit via Geometrie

#Messung 4, Kugel
    T_K = ufloat(np.mean(T_K_array),sem(T_K_array))                         
    d_K = ufloat(np.mean(d_K_array),sem(d_K_array))                         
    m_K = ufloat(np.mean(m_K_array),sem(m_K_array))                         

    I_K_array = T_K_array**2*D/(4*(np.pi)**2)
    I_K_Messung = ufloat(np.mean(I_K_array),sem(I_K_array))     #Traegheit via Schwingdauer
    I_K_Theoretisch = (2/5)*m_K_neu*(d_K_neu/2)**2      #Traegheit via Geometrie
    
#Messung 5, Puppe
A_d_l =  ufloat(np.mean(A_d_l_array),sem(A_d_l_array))
A_d_r =  ufloat(np.mean(A_d_r_array),sem(A_d_r_array))
A_h_l =  ufloat(np.mean(A_h_l_array),sem(A_h_l_array))
A_h_r =  ufloat(np.mean(A_h_r_array),sem(A_h_r_array))
B_d_l =  ufloat(np.mean(B_d_l_array),sem(B_d_l_array))
B_d_r =  ufloat(np.mean(B_d_r_array),sem(B_d_r_array))
B_h_l =  ufloat(np.mean(B_h_l_array),sem(B_h_l_array))
B_h_r =  ufloat(np.mean(B_h_r_array),sem(B_h_r_array))
R_d =  ufloat(np.mean(R_d_array),sem(R_d_array))
R_h =  ufloat(np.mean(R_h_array),sem(R_h_array))
K_d =  ufloat(np.mean(K_d_array),sem(K_d_array))
m_P =  ufloat(np.mean(m_P_array),sem(m_P_array))
T_1 =  ufloat(np.mean(T_1_array),sem(T_1_array))
T_2 =  ufloat(np.mean(T_2_array),sem(T_2_array))


I_1_array = (T_1_array)**2*D/(4*(np.pi)**2)                        
I_2_array = (T_2_array)**2*D/(4*(np.pi)**2)
I_1_ =ufloat(np.mean(I_1_array),sem(I_1_array))                      
I_2 =ufloat(np.mean(I_2_array),sem(I_2_array))

V_A_l = np.pi*(A_d_l/2)**2*A_h_l
V_A_r = np.pi*(A_d_r/2)**2*A_h_r
V_B_l = np.pi*(B_d_l/2)**2*B_h_l
V_B_r = np.pi*(B_d_r/2)**2*B_h_r
V_K = (4/3)*np.pi*(K_d/2)**3
V_R = np.pi*(R_d/2)**2*R_h
V_Gesamt = V_A_l+V_A_r+V_B_l+V_B_r+V_R+V_K
Dichte = m_P/V_Gesamt

m_A_l = Dichte*V_A_l
m_A_r = Dichte*V_A_r
m_B_l = Dichte*V_B_l
m_B_r = Dichte*V_B_r
m_K = Dichte*V_K
m_R = Dichte*V_R

I_A_l_1 = m_A_l*(A_d_l/2)**2*0.5+(R_d/2+A_d_l/2)**2
I_A_r_1 = m_A_r*(A_d_r/2)**2*0.5+(R_d/2+A_d_r/2)**2
I_B_l = m_B_l*((B_d_l/2)**2/4+B_h_l**2/12)+(B_h_l/2)**2
I_B_r = m_B_r*((B_d_r/2)**2/4+B_h_r**2/12)+(B_h_r/2)**2
I_K = 2/5*m_K*(K_d/2)**2
I_R = m_R*(R_d/2)**2*0.5


I_A_l_2 = m_A_l*((A_d_l/2)**2/4+A_h_l**2/12)+(R_d/2+A_h_l/2)**2
I_A_r_2 = m_A_r*((A_d_r/2)**2/4+A_h_r**2/12)+(R_d/2+A_h_r/2)**2


 
print('Die Winkelrichtgroesse D betraegt:',D,'Nm')


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
    I=D*b/(4*np.pi*np.pi)
    plt.plot(x,y,"rx")
    #x_plot=np.linspace(0,10)
    #plt.plot(x_plot**2,x_plot**2*A+B)
    plt.show()
    print (A, A_error, B, B_error, I)


###Bestimmung des Eigentraegheitsmoments I_D der Drillachse



				        

#print('Zeit Eigentraegheit',T_D_array,'s')
print("")
print('Das Gewicht der Masse m_1 ist',m_1,'kg')
print('Das Gewicht der Masse m_2 ist',m_2,'kg')
print('Das Eigentraegheitsmoment I der Drillachse ist',I_D,'kgm**2')


#Bestimmung des Eigentraegheitsmoments I_Z des Styroporzylinders
		




print("")
print('Die Schwingungsdauer des Zylinders T_Z_neu ist',T_Z_neu,'s')
print('Der Durchmesser des Zylinders m_2 ist',d_Z_neu,'m')
print('Die Hoehe des Zylinders h_Z_neu ist',h_Z_neu,'m')
print('Die gemessene Masse des Zylinders m_2 ist',m_2,'kg')
print('Das Traegheitsmoment des Zylinders ist ',I_Z_Messung,'kgm**2')
print("Das aussere Volumen des Zylinders ist ",V_Z,"m^3")
print("Die errechnete Masse des Voll-Zylinders ist ",m_2,"kg")

#Bestimmung des Eigentraegheitsmoments I_K der Kugel
		


print("")						
print('Die Schwingungsdauer der Kugel T_K_neu ist',T_K_neu,'s')
print('Der Durchmesser derKugel d_K_neu ist',d_K_neu,'m')
print('Die Masse der Kugel m_K_neu ist',m_K_neu,'kg')
print('Das Traegheitsmoment der Kugel ist ',I_K_Messung,'kgm**2')


#Berechnung der theoretischen Werte vom Traegheitsmoment der Kugel und des Zylinders

print("")
print('Der theoretisch errechnete Wert fuer das Traegheitsmoment der Kugel ist',I_K_Theoretisch,'kgm**2')
print('Der theoretisch errechnete Wert fuer das Traegheitsmoment des Zylinders ist',I_Z_Theoretisch,'kgm**2')

#Bestimmung des Eigentraegheitsmoment I_P der Puppe


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



print("")
print('Das gemessene Traegheitsmoment in Position 1 ist',I_1_neu,'kgm**2')
print('Das gemessene Traegheitsmoment in Position 2 ist',I_2_neu,'kgm**2')
print("")
#linregress(A_array**2,(T_D_array)**2)



#print('Das Volumen des linken Armes ist',V_A_l,'m³')
#print('Das Volumen des rechten Armes ist',V_A_r,'m³')
#print('Das Volumen des linken Beines ist',V_B_l,'m³')
#print('Das Volumen des rechten Beines ist',V_B_r,'m³')
#print('Das Volumen des Rumpfes ist',V_R,'m³')
#print('Das Volumen des Kopfes ist',V_K,'m³')
print('Das Gesamtvolumen',V_Gesamt,'m^3')
print('Die Dichte ist',Dichte,'kg/m^3')



#print('Masse des linken Armes:',m_A_l,'kg')
#print('Masse des rechten Armes:',m_A_r,'kg')
#print('Masse des linken Beines:',m_B_l,'kg')
#print('Masse des rechten Beines:',m_B_r,'kg')
#print('Masse des Kopfes:',m_K,'kg')
#print('Masse des Rumpfes:',m_R,'kg')


print("Theorie:")
print('Das Traegheitsmoment des linken Beines ist',I_B_l,'kgm^2')
print('Das Traegheitsmoment des rechten Beines ist',I_B_r,'kgm^2')
print('Das Traegheitsmoment des Kopfes ist',I_K,'kgm^2')
print('Das Traegheitsmoment des Rumpfes ist',I_R,'kgm^2')
print('')
print('Das Traegheitsmoment des linken Armes in Position 1 ist',I_A_l_1,'kgm^2')
print('Das Traegheitsmoment des rechten Armes in Position 1 ist',I_A_r_1,'kgm^2')

print('Das Traegheitsmoment des linken Armes in Position 2 ist',I_A_l_2,'kgm^2')
print('Das Traegheitsmoment des rechten Armes in Position 2 ist',I_A_r_2,'kgm^2')


print('Gesamttraegheit Pos. 1',I_1,'kgm^2')
print('Gesamttraegheit Pos. 2',I_2,'kgm^2')
