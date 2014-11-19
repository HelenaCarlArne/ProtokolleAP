#
# Header
#
import numpy as np 									                                                    
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
import uncertainties.unumpy as unp
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as stds
from uncertainties import ufloat
from scipy.stats import sem
import peakdetect

#
# Amplitudenbestimmung mit Ausgabe der Quotienten
#

# Aluminium M2
T5x_M2_x,T5x_M2_y = np.genfromtxt('../Tabellen/M2_Alu_T5_max.txt').T
T6x_M2_x,T6x_M2_y = np.genfromtxt('../Tabellen/M2_Alu_T6_max.txt').T
T5n_M2_x,T5n_M2_y = np.genfromtxt('../Tabellen/M2_Alu_T5_min.txt').T
T6n_M2_x,T6n_M2_y = np.genfromtxt('../Tabellen/M2_Alu_T6_min.txt').T

T5x_M2=ufloat(np.mean(T5x_M2_y),sem(T5x_M2_y))
T5n_M2=ufloat(np.mean(T5n_M2_y),sem(T5n_M2_y))
T6x_M2=ufloat(np.mean(T6x_M2_y),sem(T6x_M2_y))
T6n_M2=ufloat(np.mean(T6n_M2_y),sem(T6n_M2_y))

print("")
print("")
print("Aluminium M2, T5 und T6")
print(T5x_M2)
print(T5n_M2)
print("ergibt",(T5x_M2-T5n_M2)/2)
print("")
print(T6x_M2)
print(T6n_M2)
print("ergibt",(T6x_M2-T6n_M2)/2)
print("")
print("Quotient ist:",(T6x_M2-T6n_M2)/(T5x_M2-T5n_M2))
# Aluminium M3
#T5x_M3_x,T5x_M3_y,T6x_M3_x,T6x_M3_y = np.genfromtxt('../Tabellen/M3_Alu_max.txt').T
#T5n_M3_x,T5n_M3_y,T6n_M3_x,T6n_M3_y = np.genfromtxt('../Tabellen/M3_Alu_min.txt').T
# plt.plot(T5x_M3_x,T5x_M3_y,"rx")
# plt.plot(T5n_M3_x,T5n_M3_y,"rx")
# plt.show()
# print("")
# print("")
# print("Aluminium M3, T5 und T6")
# print(np.mean(T5x_M3_y))
# print(np.mean(T5n_M3_y))
# print("ergibt",(np.mean(T5x_M3_y)-np.mean(T5n_M3_y))/2)
# print("")
# print(np.mean(T6x_M3_y))
# print(np.mean(T6n_M3_y))
# print("ergibt",(np.mean(T6x_M3_y)-np.mean(T6n_M3_y))/2)
# print("")
# print("Quotient ist:",(np.mean(T6x_M3_y)-np.mean(T6n_M3_y))/(np.mean(T5x_M3_y)-np.mean(T5n_M3_y)))

# Messing M2
T1x_M2_x,T1x_M2_y = np.genfromtxt('../Tabellen/M2_Mess_T1_max.txt').T
T2x_M2_x,T2x_M2_y = np.genfromtxt('../Tabellen/M2_Mess_T2_max.txt').T
T1n_M2_x,T1n_M2_y = np.genfromtxt('../Tabellen/M2_Mess_T1_min.txt').T
T2n_M2_x,T2n_M2_y = np.genfromtxt('../Tabellen/M2_Mess_T2_min.txt').T
T1x_M2=ufloat(np.mean(T1x_M2_y),sem(T1x_M2_y))
T1n_M2=ufloat(np.mean(T1n_M2_y),sem(T1n_M2_y))
T2x_M2=ufloat(np.mean(T2x_M2_y),sem(T2x_M2_y))
T2n_M2=ufloat(np.mean(T2n_M2_y),sem(T2n_M2_y))
print("")
print("")
print("Messing M2, T1 und T2")
print(T1x_M2)
print(T1n_M2)
print("ergibt",(T1x_M2-T1n_M2)/2)
print("")
print(np.mean(T2x_M2_y))
print(np.mean(T2n_M2_y))
print("ergibt",(T2x_M2-T2n_M2)/2)
print("")
print("Quotient ist:",(T2x_M2-T2n_M2)/(T1x_M2-T1n_M2))
# Messing M3
#T1x_M3_x,T1x_M3_y = np.genfromtxt('../Tabellen/M3_Messing_T1_max.txt').T
#T2x_M3_x,T2x_M3_y = np.genfromtxt('../Tabellen/M3_Messing_T2_max.txt').T
#T1n_M3_x,T1n_M3_y = np.genfromtxt('../Tabellen/M3_Messing_T1_min.txt').T
#T2n_M3_x,T2n_M3_y = np.genfromtxt('../Tabellen/M3_Messing_T2_min.txt').T
# plt.plot(T1x_M3_x,T1x_M3_y,"rx")
# plt.plot(T1n_M3_x,T1n_M3_y,"rx")
# plt.show()
# print("")
# print("")
# print("Messing M3, T1 und T2")
# print(np.mean(T1x_M3_y))
# print(np.mean(T1n_M3_y))
# print("ergibt",(np.mean(T1x_M3_y)-np.mean(T1n_M3_y))/2)
# print("")
# print(np.mean(T2x_M3_y))
# print(np.mean(T2n_M3_y))
# print("ergibt",(np.mean(T2x_M3_y)-np.mean(T2n_M3_y))/2)
# print("")
# print("Quotient ist:",(np.mean(T2x_M3_y)-np.mean(T2n_M3_y))/(np.mean(T1x_M3_y)-np.mean(T1n_M3_y)))

# Edelstahl M3
T7x_M3_x,T7x_M3_y = np.genfromtxt('../Tabellen/M3_Edelstahl_T7_max.txt').T
T8x_M3_x,T8x_M3_y = np.genfromtxt('../Tabellen/M3_Edelstahl_T8_max.txt').T
T7n_M3_x,T7n_M3_y = np.genfromtxt('../Tabellen/M3_Edelstahl_T7_min.txt').T
T8n_M3_x,T8n_M3_y = np.genfromtxt('../Tabellen/M3_Edelstahl_T8_min.txt').T

T8x_M3=ufloat(np.mean(T8x_M3_y),sem(T8x_M3_y))
T8n_M3=ufloat(np.mean(T8n_M3_y),sem(T8n_M3_y))
T7x_M3=ufloat(np.mean(T7x_M3_y),sem(T7x_M3_y))
T7n_M3=ufloat(np.mean(T7n_M3_y),sem(T7n_M3_y))

print("")
print("")
print("Edelstahl M3, T8 und T7")
print(T8x_M3)
print(T8n_M3)
print("ergibt",(T8x_M3-T8n_M3)/2)
print("")
print(T7x_M3)
print(T7n_M3)
print("ergibt",(T7x_M3-T7n_M3)/2)
print("")
print("Quotient ist:",(T7x_M3-T7n_M3)/(T8x_M3-T8n_M3))
#
# Phasenversatz
#
T1x_M2=ufloat(np.mean(T1x_M2_x-T2x_M2_x),sem(T1x_M2_x-T2x_M2_x))
T5x_M2=ufloat(np.mean(T5x_M2_x-T6x_M2_x),sem(T5x_M2_x-T6x_M2_x))
T8x_M3=ufloat(np.mean(T8x_M3_x-T7x_M3_x),sem(T8x_M3_x-T7x_M3_x))
# Aluminium M2
print("Alu M2")
#warmth=(c*rho*0.009)/(2*T8x_M3*ln(quot))
print(T5x_M2)
warmthi=((830*2800*0.0009)/(2*T5x_M2*unp.log(ufloat(1.893,0.021))))
print(warmthi)
# Aluminium M3
#print("Alu M3")
#print(np.mean(T5x_M3_x-T6x_M3_x))
# Messing M2
print("Messing M2")
print(T1x_M2)
warmthi=((385*8520*0.0009)/(2*T1x_M2*unp.log(ufloat(3.47,0.06))))
print(warmthi)
# Messing M3
#print("Messing M3")
#print(np.mean(T1x_M3_x-T2x_M3_x))
# Edelstahl M3
print("Edelstahl M3")
print(T8x_M3)
warmthi=((400*8000*0.0009)/(2*T8x_M3*unp.log(ufloat(5.477,0.024))))
print(warmthi)
