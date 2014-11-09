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
T5x_M2_x,T5x_M2_y,T6x_M2_x,T6x_M2_y = np.genfromtxt('../Tabellen/M2_Alu_max.txt').T
T5n_M2_x,T5n_M2_y,T6n_M2_x,T6n_M2_y = np.genfromtxt('../Tabellen/M2_Alu_min.txt').T
# plt.plot(T5x_M2_x,T5x_M2_y,"rx")
# plt.plot(T5n_M2_x,T5n_M2_y,"rx")
# plt.show()
print("")
print("")
print("Aluminium M2, T5 und T6")
print(np.mean(T5x_M2_y))
print(np.mean(T5n_M2_y))
print("ergibt",(np.mean(T5x_M2_y)-np.mean(T5n_M2_y))/2)
print("")
print(np.mean(T6x_M2_y))
print(np.mean(T6n_M2_y))
print("ergibt",(np.mean(T6x_M2_y)-np.mean(T6n_M2_y))/2)
print("")
print("Quotient ist:",(np.mean(T6x_M2_y)-np.mean(T6n_M2_y))/(np.mean(T5x_M2_y)-np.mean(T5n_M2_y)))
# Aluminium M3
T5x_M3_x,T5x_M3_y,T6x_M3_x,T6x_M3_y = np.genfromtxt('../Tabellen/M3_Alu_max.txt').T
T5n_M3_x,T5n_M3_y,T6n_M3_x,T6n_M3_y = np.genfromtxt('../Tabellen/M3_Alu_min.txt').T
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
# plt.plot(T1x_M2_x,T1x_M2_y,"rx")
# plt.plot(T1n_M2_x,T1n_M2_y,"rx")
# plt.show()
print("")
print("")
print("Messing M2, T1 und T2")
print(np.mean(T1x_M2_y))
print(np.mean(T1n_M2_y))
print("ergibt",(np.mean(T1x_M2_y)-np.mean(T1n_M2_y))/2)
print("")
print(np.mean(T2x_M2_y))
print(np.mean(T2n_M2_y))
print("ergibt",(np.mean(T2x_M2_y)-np.mean(T2n_M2_y))/2)
print("")
print("Quotient ist:",(np.mean(T2x_M2_y)-np.mean(T2n_M2_y))/(np.mean(T1x_M2_y)-np.mean(T1n_M2_y)))
# Messing M3
T1x_M3_x,T1x_M3_y,T2x_M3_x,T2x_M3_y = np.genfromtxt('../Tabellen/M3_Messing_max.txt').T
T1n_M3_x,T1n_M3_y,T2n_M3_x,T2n_M3_y = np.genfromtxt('../Tabellen/M3_Messing_min.txt').T
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
T7x_M3_x,T7x_M3_y,T8x_M3_x,T8x_M3_y = np.genfromtxt('../Tabellen/M3_Edelstahl_max.txt').T
T7n_M3_x,T7n_M3_y,T8n_M3_x,T8n_M3_y = np.genfromtxt('../Tabellen/M3_Edelstahl_min.txt').T
# plt.plot(T7x_M3_x,T7x_M3_y,"rx")
# plt.plot(T7n_M3_x,T7n_M3_y,"rx")
# plt.show()
print("")
print("")
print("Edelstahl M3, T7 und T8")
print(np.mean(T7x_M3_y))
print(np.mean(T7n_M3_y))
print("ergibt",(np.mean(T7x_M3_y)-np.mean(T7n_M3_y))/2)
print("")
print(np.mean(T8x_M3_y))
print(np.mean(T8n_M3_y))
print("ergibt",(np.mean(T8x_M3_y)-np.mean(T8n_M3_y))/2)
print("")
print("Quotient ist:",(np.mean(T7x_M3_y)-np.mean(T7n_M3_y))/(np.mean(T8x_M3_y)-np.mean(T8n_M3_y)))
print("")
print("")
#
# Phasenversatz
#

# Aluminium M2
print("Alu M2")
print(np.mean(T5x_M2_x-T6x_M2_x))
# Aluminium M3
#print("Alu M3")
#print(np.mean(T5x_M3_x-T6x_M3_x))
# Messing M2
print("Messing M2")
print(np.mean(T1x_M2_x-T2x_M2_x))
# Messing M3
#print("Messing M3")
#print(np.mean(T1x_M3_x-T2x_M3_x))
# Edelstahl M3
print("Edelstahl M3")
print(np.mean(T8x_M3_x-T7x_M3_x))