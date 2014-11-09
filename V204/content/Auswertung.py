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
# Einladen der Daten
#
ID_M1, T1_M1, T2_M1, T3_M1, T4_M1, T5_M1, T6_M1, T7_M1, T8_M1 = np.genfromtxt("../Originalwerte/MessungStat").T
ID_M2, T1_M2, T5_M2, T2_M2, T6_M2, Junk_M2 = np.genfromtxt("../Originalwerte/Messung80").T 
ID_M3, T1_M3, T2_M3, T3_M3, T4_M3, T5_M3, T6_M3, T7_M3, T8_M3, Junk_M3= np.genfromtxt("../Originalwerte/Messung200",delimiter = ',').T
t_M1=np.arange(0,T1_M1.size*5,5)
t_M2=np.arange(0,T1_M2.size*5,5)
t_M3=np.arange(0,T1_M3.size*5,5)
def f(x,a,b,c):
	return -a*np.exp(-b*x)+c
def g(x,A,B,C,D):
	return A*np.sin(x)+B*np.cos(x)+C
#
# Extremalwerte für Messung 2ç
#
max_M2_T1,min_M2_T1 = peakdetect.peakdetect(T1_M2,t_M2,1,0)
max_M2_T2,min_M2_T2 = peakdetect.peakdetect(T2_M2,t_M2,1,0)
max_M2_T5,min_M2_T5 = peakdetect.peakdetect(T5_M2,t_M2,1,0)
max_M2_T6,min_M2_T6 = peakdetect.peakdetect(T6_M2,t_M2,1,0)

max_M2_T1_d = np.array(max_M2_T1).T
min_M2_T1_d = np.array(min_M2_T1).T
max_M2_T2_d = np.array(max_M2_T2).T
min_M2_T2_d = np.array(min_M2_T2).T
max_M2_T5_d = np.array(max_M2_T5).T
min_M2_T5_d = np.array(min_M2_T5).T
max_M2_T6_d = np.array(max_M2_T6).T
min_M2_T6_d = np.array(min_M2_T6).T

max_M2_T1[0] = np.delete(max_M2_T1_d[0], [9,10])
min_M2_T1[0] = np.delete(min_M2_T1_d[0], [8,9])
max_M2_T2[0] = np.delete(max_M2_T2_d[0], [9,10])
min_M2_T2[0] = np.delete(min_M2_T2_d[0], [8,9])
max_M2_T5[0] = np.delete(max_M2_T5_d[0], [9,10])
min_M2_T5[0] = np.delete(min_M2_T5_d[0], [8,9])
max_M2_T6[0] = np.delete(max_M2_T6_d[0], [9,10])
min_M2_T6[0] = np.delete(min_M2_T6_d[0], [8,9])
max_M2_T1[1] = np.delete(max_M2_T1_d[1], [9,10])
min_M2_T1[1] = np.delete(min_M2_T1_d[1], [8,9])
max_M2_T2[1] = np.delete(max_M2_T2_d[1], [9,10])
min_M2_T2[1] = np.delete(min_M2_T2_d[1], [8,9])
max_M2_T5[1] = np.delete(max_M2_T5_d[1], [9,10])
min_M2_T5[1] = np.delete(min_M2_T5_d[1], [8,9])
max_M2_T6[1] = np.delete(max_M2_T6_d[1], [9,10])
min_M2_T6[1] = np.delete(min_M2_T6_d[1], [8,9])

#
# Extremalwerte für Messung 3
#
max_M3_T1,min_M3_T1 = peakdetect.peakdetect(T1_M3,t_M3,1,0)
max_M3_T2,min_M3_T2 = peakdetect.peakdetect(T2_M3,t_M3,1,0)
max_M3_T5,min_M3_T5 = peakdetect.peakdetect(T5_M3,t_M3,1,0)
max_M3_T6,min_M3_T6 = peakdetect.peakdetect(T6_M3,t_M3,1,0)
max_M3_T7,min_M3_T7 = peakdetect.peakdetect(T7_M3,t_M3,1,0)
max_M3_T8,min_M3_T8 = peakdetect.peakdetect(T8_M3,t_M3,1,0.02)

max_M3_T1 = np.array(max_M3_T1).T
min_M3_T1 = np.array(min_M3_T1).T
max_M3_T2 = np.array(max_M3_T2).T
min_M3_T2 = np.array(min_M3_T2).T
max_M3_T5 = np.array(max_M3_T5).T
min_M3_T5 = np.array(min_M3_T5).T
max_M3_T6 = np.array(max_M3_T6).T
min_M3_T6 = np.array(min_M3_T6).T
max_M3_T7 = np.array(max_M3_T7).T
min_M3_T7 = np.array(min_M3_T7).T
max_M3_T8 = np.array(max_M3_T8).T
min_M3_T8 = np.array(min_M3_T8).T



#
# Plot 4 - Periodisch Messing
#
coeff, covar = curve_fit(f,max_M2_T1[0],max_M2_T1[1], p0=(1, 1e-6, 1))
plt.plot(max_M2_T1[0],f(max_M2_T1[0],*coeff),"k--", label="Amplitudenfunktion")
coeff, covar = curve_fit(f,min_M2_T1[0],min_M2_T1[1], p0=(1, 1e-6, 1))
plt.plot(min_M2_T1[0],f(min_M2_T1[0],*coeff),"k--")
plt.plot(t_M2,T1_M2, "b-",label="kühles Ende")
plt.plot(max_M2_T1_d[0],max_M2_T1_d[1],"kx",label="Extrema")
plt.plot(min_M2_T1_d[0],min_M2_T1_d[1],"kx")
plt.legend(loc="best")
plt.xlabel(r"Zeit $t [s]$")
plt.ylabel(r"Temperatur [$^{\circ}{\rm C}$]")
plt.savefig("../Bilder/M2_Messing_kuehl.pdf")
plt.show()
coeff, covar = curve_fit(f,max_M2_T6[0],max_M2_T6[1], p0=(1, 1e-6, 1))
plt.plot(max_M2_T6[0],f(max_M2_T6[0],*coeff),"k--", label="Amplitudenfunktion")
coeff, covar = curve_fit(f,min_M2_T6[0],min_M2_T6[1], p0=(1, 1e-6, 1))
plt.plot(min_M2_T6[0],f(min_M2_T6[0],*coeff),"k--")
plt.plot(t_M2,T6_M2, "r-",label="warmes Ende")
plt.plot(max_M2_T6_d[0],max_M2_T6_d[1],"kx",label="Extrema")
plt.plot(min_M2_T6_d[0],min_M2_T6_d[1],"kx")
plt.legend(loc="best")
plt.xlabel(r"Zeit $t [s]$")
plt.ylabel(r"Temperatur [$^{\circ}{\rm C}$]")
plt.savefig("../Bilder/M2_Messing_warm.pdf")
plt.show()
coeffx, covar = curve_fit(f,max_M2_T1[0],max_M2_T1[1], p0=(1, 1e-6, 1))
coeffn, covar = curve_fit(f,min_M2_T1[0],min_M2_T1[1], p0=(1, 1e-6, 1))
plt.plot(t_M2,T1_M2-f(t_M2,*coeffx),"g-",label="Ampl.fkt. oben")
plt.plot(t_M2,T1_M2-f(t_M2,*coeffn),"y--")
coeffx, covar = curve_fit(f,max_M2_T2[0],max_M2_T2[1], p0=(1, 1e-6, 1))
coeffn, covar = curve_fit(f,min_M2_T2[0],min_M2_T2[1], p0=(1, 1e-6, 1))
plt.plot(t_M2,T2_M2-f(t_M2,*coeffx),"g-")
plt.plot(t_M2,T2_M2-f(t_M2,*coeffn),"y--",label="Ampl.fkt. unten")
plt.legend(loc="lower right")
plt.xlabel(r"Zeit $t [s]$")
plt.ylim(-10,10)
plt.ylabel(r"Temperatur [$^{\circ}{\rm C}$]")
plt.savefig("../Bilder/Normierungsauswahl/M2_Messing_norm.pdf")
plt.show()

coeffx, covar = curve_fit(f,max_M2_T1[0],max_M2_T1[1], p0=(1, 1e-6, 1))
Junk1, t_M2_norm, Junk2 = np.split(t_M2,[50, 350])
Junk1, T1_M2_norm, Junk2= np.split(T1_M2-f(t_M2,*coeffx),[50, 350])
plt.plot(t_M2_norm,T1_M2_norm,"b-",label="Grundschwingung, fern")
coeffx, covar = curve_fit(f,max_M2_T2[0],max_M2_T2[1], p0=(1, 1e-6, 1))
Junk1, T2_M2_norm, Junk2= np.split(T2_M2-f(t_M2,*coeffx),[50, 350])
plt.plot(t_M2_norm,T2_M2_norm,"r-",label="Grundschwingung, nah")
plt.legend(loc="upper left")
plt.ylim(-7,2)
plt.xlabel(r"Zeit $t [s]$")
plt.ylabel(r"Temperatur [$^{\circ}{\rm C}$]")
max_M2_T1_furt,min_M2_T1_furt = peakdetect.peakdetect(T1_M2_norm,t_M2_norm,1,0)
max_M2_T2_furt,min_M2_T2_furt = peakdetect.peakdetect(T2_M2_norm,t_M2_norm,1,0)
max_M2_T2_furt = np.array(max_M2_T2_furt).T
min_M2_T2_furt = np.array(min_M2_T2_furt).T
max_M2_T1_furt = np.array(max_M2_T1_furt).T
min_M2_T1_furt = np.array(min_M2_T1_furt).T
plt.plot(max_M2_T2_furt[0],max_M2_T2_furt[1],"x")
plt.plot(max_M2_T1_furt[0],max_M2_T1_furt[1],"x")
plt.plot(min_M2_T1_furt[0],min_M2_T1_furt[1],"x")
plt.plot(min_M2_T2_furt[0],min_M2_T2_furt[1],"x")
plt.savefig("../Bilder/M2_Messing_norm.pdf")
plt.show()
max_M2_T1_furt_x = max_M2_T1_furt[0] 
max_M2_T1_furt_y = max_M2_T1_furt[1] 
min_M2_T1_furt_x = min_M2_T1_furt[0] 
min_M2_T1_furt_y = min_M2_T1_furt[1] 
min_M2_T2_furt_x = min_M2_T2_furt[0] 
min_M2_T2_furt_y = min_M2_T2_furt[1] 
max_M2_T2_furt_x = max_M2_T2_furt[0] 
max_M2_T2_furt_y = max_M2_T2_furt[1] 

np.savetxt('../Tabellen/M2_Mess_T1_max.txt', np.array([max_M2_T1_furt_x,max_M2_T1_furt_y]).T, header="Messung 3 - Maxima\nT1\nx\t y\t x\t y")
np.savetxt('../Tabellen/M2_Mess_T2_max.txt', np.array([max_M2_T2_furt_x,max_M2_T2_furt_y]).T, header="Messung 3 - Maxima\nT2\nx\t y\t x\t y")
np.savetxt('../Tabellen/M2_Mess_T1_min.txt', np.array([min_M2_T1_furt_x,min_M2_T1_furt_y]).T, header="Messung 3 - Minima\nT1\nx\t y\t x\t y")
np.savetxt('../Tabellen/M2_Mess_T2_min.txt', np.array([min_M2_T2_furt_x,min_M2_T2_furt_y]).T, header="Messung 3 - Minima\nT2\nx\t y\t x\t y")

coeff, covar = curve_fit(f,max_M3_T1[0],max_M3_T1[1], p0=(1, 1e-6, 1))
plt.plot(max_M3_T1[0],f(max_M3_T1[0],*coeff),"k--", label="Amplitudenfunktion")
coeff, covar = curve_fit(f,min_M3_T1[0],min_M3_T1[1], p0=(1, 1e-6, 1))
plt.plot(min_M3_T1[0],f(min_M3_T1[0],*coeff),"k--")
plt.plot(t_M3,T1_M3, "b-",label="kühles Ende")
plt.plot(max_M3_T1[0],max_M3_T1[1],"kx",label="Extrema")
plt.plot(min_M3_T1[0],min_M3_T1[1],"kx")
plt.legend(loc="lower right")
plt.xlabel(r"Zeit $t [s]$")
plt.ylabel(r"Temperatur [$^{\circ}{\rm C}$]")
plt.savefig("../Bilder/M3_Messing_kuehl.pdf")
plt.show()
coeff, covar = curve_fit(f,max_M3_T2[0],max_M3_T2[1], p0=(1, 1e-6, 1))
plt.plot(max_M3_T2[0],f(max_M3_T2[0],*coeff),"k--", label="Amplitudenfunktion")
coeff, covar = curve_fit(f,min_M3_T2[0],min_M3_T2[1], p0=(1, 1e-6, 1))
plt.plot(min_M3_T2[0],f(min_M3_T2[0],*coeff),"k--")
plt.plot(t_M3,T2_M3, "r-",label="warmes Ende")
plt.plot(max_M3_T2[0],max_M3_T2[1],"kx",label="Extrema")
plt.plot(min_M3_T2[0],min_M3_T2[1],"kx")
plt.legend(loc="lower right")
plt.xlabel(r"Zeit $t [s]$")
plt.ylabel(r"Temperatur [$^{\circ}{\rm C}$]")
plt.savefig("../Bilder/M3_Messing_warm.pdf")
plt.show()
coeffx, covar = curve_fit(f,max_M3_T1[0],max_M3_T1[1], p0=(1, 1e-6, 1))
coeffn, covar = curve_fit(f,min_M3_T1[0],min_M3_T1[1], p0=(1, 1e-6, 1))
plt.plot(t_M3,T1_M3-f(t_M3,*coeffx),"g-",label="Ampl.fkt. oben")
plt.plot(t_M3,T1_M3-f(t_M3,*coeffn),"y--")
coeffx, covar = curve_fit(f,max_M3_T2[0],max_M3_T2[1], p0=(1, 1e-6, 1))
coeffn, covar = curve_fit(f,min_M3_T2[0],min_M3_T2[1], p0=(1, 1e-6, 1))
plt.plot(t_M3,T2_M3-f(t_M3,*coeffx),"g-")
plt.plot(t_M3,T2_M3-f(t_M3,*coeffn),"y--",label="Ampl.fkt. unten")
plt.legend(loc="best")
plt.xlabel(r"Zeit $t [s]$")
plt.ylabel(r"Temperatur [$^{\circ}{\rm C}$]")
plt.savefig("../Bilder/Normierungsauswahl/M3_Messing_norm.pdf")
plt.show()


coeffx, covar = curve_fit(f,max_M3_T1[0],max_M3_T1[1], p0=(1, 1e-6, 1))
Junk1, t_M3_norm, Junk2 = np.split(t_M3,[120, 900])
Junk1, T1_M3_norm, Junk2= np.split(T1_M3-f(t_M3,*coeffx),[120, 900])
plt.plot(t_M3_norm,T1_M3_norm,"b-",label="Grundschwingung, fern")
coeffx, covar = curve_fit(f,max_M3_T2[0],max_M3_T2[1], p0=(1, 1e-6, 1))
Junk1, T2_M3_norm, Junk2= np.split(T2_M3-f(t_M3,*coeffx),[120, 900])
plt.plot(t_M3_norm,T2_M3_norm,"r-",label="Grundschwingung, nah")
plt.legend(loc="upper left")
plt.ylim(-14,4)
plt.xlabel(r"Zeit $t [s]$")
plt.ylabel(r"Temperatur [$^{\circ}{\rm C}$]")
max_M3_T1_furt,min_M3_T1_furt = peakdetect.peakdetect(T1_M3_norm,t_M3_norm,1,0.01)
max_M3_T2_furt,min_M3_T2_furt = peakdetect.peakdetect(T2_M3_norm,t_M3_norm,1,0)
max_M3_T2_furt = np.array(max_M3_T2_furt).T
min_M3_T2_furt = np.array(min_M3_T2_furt).T
max_M3_T1_furt = np.array(max_M3_T1_furt).T
min_M3_T1_furt = np.array(min_M3_T1_furt).T
plt.plot(max_M3_T2_furt[0],max_M3_T2_furt[1],"x")
plt.plot(max_M3_T1_furt[0],max_M3_T1_furt[1],"x")
plt.plot(min_M3_T1_furt[0],min_M3_T1_furt[1],"x")
plt.plot(min_M3_T2_furt[0],min_M3_T2_furt[1],"x")

plt.savefig("../Bilder/M3_Messing_norm.pdf")
plt.show()
max_M3_T1_furt_x = max_M3_T1_furt[0] 
max_M3_T1_furt_y = max_M3_T1_furt[1] 
min_M3_T1_furt_x = min_M3_T1_furt[0] 
min_M3_T1_furt_y = min_M3_T1_furt[1] 
min_M3_T2_furt_x = min_M3_T2_furt[0] 
min_M3_T2_furt_y = min_M3_T2_furt[1] 
max_M3_T2_furt_x = max_M3_T2_furt[0] 
max_M3_T2_furt_y = max_M3_T2_furt[1] 
np.savetxt('../Tabellen/M3_Messing_max.txt', np.array([max_M3_T1_furt_x,max_M3_T1_furt_y,max_M3_T2_furt_x,max_M3_T2_furt_y]).T, header="Messung 3 - Maxima\nT1\tT2\nx\t y\t x\t y")
np.savetxt('../Tabellen/M3_Messing_min.txt', np.array([min_M3_T1_furt_x,min_M3_T1_furt_y,min_M3_T2_furt_x,min_M3_T2_furt_y]).T, header="Messung 3 - Minima\nT1\tT2\nx\t y\t x\t y")

#
# Plot 5 - Periodisch Aluminium
#
coeff, covar = curve_fit(f,max_M2_T5[0],max_M2_T5[1], p0=(1, 1e-6, 1))
plt.plot(max_M2_T5[0],f(max_M2_T5[0],*coeff),"k--", label="Amplitudenfunktion")
coeff, covar = curve_fit(f,min_M2_T5[0],min_M2_T5[1], p0=(1, 1e-6, 1))
plt.plot(min_M2_T5[0],f(min_M2_T5[0],*coeff),"k--")
plt.plot(t_M2,T5_M2, "b-",label="kühles Ende")
plt.plot(max_M2_T5_d[0],max_M2_T5_d[1],"kx",label="Extrema")
plt.plot(min_M2_T5_d[0],min_M2_T5_d[1],"kx")
plt.legend(loc="lower right")
plt.ylabel(r"Temperatur [$^{\circ}{\rm C}$]")
plt.xlabel(r"Zeit $t [s]$")
plt.savefig("../Bilder/M2_Alu_kuehl.pdf")
plt.show()
coeff, covar = curve_fit(f,max_M2_T6[0],max_M2_T6[1], p0=(1, 1e-6, 1))
plt.plot(max_M2_T6[0],f(max_M2_T6[0],*coeff),"k--", label="Amplitudenfunktion")
coeff, covar = curve_fit(f,min_M2_T6[0],min_M2_T6[1], p0=(1, 1e-6, 1))
plt.plot(min_M2_T5[0],f(min_M2_T5[0],*coeff),"k--")
plt.plot(t_M2,T6_M2, "r-",label="warmes Ende")
plt.plot(max_M2_T6_d[0],max_M2_T6_d[1],"kx",label="Extrema")
plt.plot(min_M2_T6_d[0],min_M2_T6_d[1],"kx")
plt.legend(loc="lower right")
plt.ylabel(r"Temperatur [$^{\circ}{\rm C}$]")
plt.xlabel(r"Zeit $t [s]$")
plt.savefig("../Bilder/M2_Alu_warm.pdf")
plt.show()
coeffx, covar = curve_fit(f,max_M2_T5[0],max_M2_T5[1], p0=(1, 1e-6, 1))
coeffn, covar = curve_fit(f,min_M2_T5[0],min_M2_T5[1], p0=(1, 1e-6, 1))
plt.plot(t_M2,T5_M2-f(t_M2,*coeffx),"g-",label="Ampl.fkt. oben")
plt.plot(t_M2,T5_M2-f(t_M2,*coeffn),"y--")
coeffx, covar = curve_fit(f,max_M2_T6[0],max_M2_T6[1], p0=(1, 1e-6, 1))
coeffn, covar = curve_fit(f,min_M2_T6[0],min_M2_T6[1], p0=(1, 1e-6, 1))
plt.plot(t_M2,T6_M2-f(t_M2,*coeffx),"g-")
plt.plot(t_M2,T6_M2-f(t_M2,*coeffn),"y--",label="Ampl.fkt. unten")
plt.legend(loc="best")
plt.ylim(-10,10)
plt.xlabel(r"Zeit $t [s]$")
plt.ylabel(r"Temperatur [$^{\circ}{\rm C}$]")
plt.savefig("../Bilder/Normierungsauswahl/M2_Alu_norm.pdf")
plt.show()

coeffx, covar = curve_fit(f,max_M2_T5[0],max_M2_T5[1], p0=(1, 1e-6, 1))
Junk1, t_M2_norm, Junk2 = np.split(t_M2,[50, 350])
Junk1, T5_M2_norm, Junk2= np.split(T5_M2-f(t_M2,*coeffx),[50, 350])
plt.plot(t_M2_norm,T5_M2_norm,"b-",label="Grundschwingung, fern")
coeffx, covar = curve_fit(f,max_M2_T6[0],max_M2_T6[1], p0=(1, 1e-6, 1))
Junk1, T6_M2_norm, Junk2= np.split(T6_M2-f(t_M2,*coeffx),[50, 350])
plt.plot(t_M2_norm,T6_M2_norm,"r-",label="Grundschwingung, nah")
plt.legend(loc="upper left")
plt.ylim(-7,2)
plt.xlabel(r"Zeit $t [s]$")
plt.ylabel(r"Temperatur [$^{\circ}{\rm C}$]")
max_M2_T6_furt,min_M2_T6_furt = peakdetect.peakdetect(T6_M2_norm,t_M2_norm,1,0.01)
max_M2_T5_furt,min_M2_T5_furt = peakdetect.peakdetect(T5_M2_norm,t_M2_norm,1,0)
max_M2_T5_furt = np.array(max_M2_T5_furt).T
min_M2_T5_furt = np.array(min_M2_T5_furt).T
max_M2_T6_furt = np.array(max_M2_T6_furt).T
min_M2_T6_furt = np.array(min_M2_T6_furt).T
plt.plot(max_M2_T5_furt[0],max_M2_T5_furt[1],"x")
plt.plot(max_M2_T6_furt[0],max_M2_T6_furt[1],"x")
plt.plot(min_M2_T6_furt[0],min_M2_T6_furt[1],"x")
plt.plot(min_M2_T5_furt[0],min_M2_T5_furt[1],"x")

plt.savefig("../Bilder/M2_Alu_norm.pdf")
plt.show()
max_M2_T5_furt_x = max_M2_T5_furt[0] 
max_M2_T5_furt_y = max_M2_T5_furt[1] 
min_M2_T5_furt_x = min_M2_T5_furt[0] 
min_M2_T5_furt_y = min_M2_T5_furt[1] 
min_M2_T6_furt_x = min_M2_T6_furt[0] 
min_M2_T6_furt_y = min_M2_T6_furt[1] 
max_M2_T6_furt_x = max_M2_T6_furt[0] 
max_M2_T6_furt_y = max_M2_T6_furt[1] 
np.savetxt('../Tabellen/M2_Alu_max.txt', np.array([max_M2_T5_furt_x,max_M2_T5_furt_y,max_M2_T6_furt_x,max_M2_T6_furt_y]).T, header="Messung 3 - Maxima\nT5\tT6\nx\t y\t x\t y")
np.savetxt('../Tabellen/M2_Alu_min.txt', np.array([min_M2_T5_furt_x,min_M2_T5_furt_y,min_M2_T6_furt_x,min_M2_T6_furt_y]).T, header="Messung 3 - Minima\nT5\tT6\nx\t y\t x\t y")


coeff, covar = curve_fit(f,max_M3_T5[0],max_M3_T5[1], p0=(1, 1e-6, 1))
plt.plot(max_M3_T5[0],f(max_M3_T5[0],*coeff),"k--", label="Amplitudenfunktion")
coeff, covar = curve_fit(f,min_M3_T5[0],min_M3_T5[1], p0=(1, 1e-6, 1))
plt.plot(min_M3_T5[0],f(min_M3_T5[0],*coeff),"k--")
plt.plot(t_M3,T5_M3, "b-",label="kühles Ende")
plt.plot(max_M3_T5[0],max_M3_T5[1],"kx",label="Extrema")
plt.plot(min_M3_T5[0],min_M3_T5[1],"kx")
plt.legend(loc="best")
plt.xlabel(r"Zeit $t [s]$")
plt.ylabel(r"Temperatur [$^{\circ}{\rm C}$]")
plt.savefig("../Bilder/M3_Alu_kuehl.pdf")
plt.show()
coeff, covar = curve_fit(f,max_M3_T6[0],max_M3_T6[1], p0=(1, 1e-6, 1))
plt.plot(max_M3_T6[0],f(max_M3_T6[0],*coeff),"k--", label="Amplitudenfunktion")
coeff, covar = curve_fit(f,min_M3_T6[0],min_M3_T6[1], p0=(1, 1e-6, 1))
plt.plot(min_M3_T6[0],f(min_M3_T6[0],*coeff),"k--")
plt.plot(t_M3,T6_M3, "r-",label="warmes Ende")
plt.plot(max_M3_T6[0],max_M3_T6[1],"kx",label="Extrema")
plt.plot(min_M3_T6[0],min_M3_T6[1],"kx")
plt.legend(loc="lower right")
plt.xlabel(r"Zeit $t [s]$")
plt.ylabel(r"Temperatur [$^{\circ}{\rm C}$]")
plt.savefig("../Bilder/M3_Alu_warm.pdf")
plt.show()
coeffx, covar = curve_fit(f,max_M3_T5[0],max_M3_T5[1], p0=(1, 1e-6, 1))
coeffn, covar = curve_fit(f,min_M3_T5[0],min_M3_T5[1], p0=(1, 1e-6, 1))
plt.plot(t_M3,T5_M3-f(t_M3,*coeffx),"g-",label="Ampl.fkt. oben")
plt.plot(t_M3,T5_M3-f(t_M3,*coeffn),"y--")
coeffx, covar = curve_fit(f,max_M3_T6[0],max_M3_T6[1], p0=(1, 1e-6, 1))
coeffn, covar = curve_fit(f,min_M3_T6[0],min_M3_T6[1], p0=(1, 1e-6, 1))
plt.plot(t_M3,T6_M3-f(t_M3,*coeffx),"g-")
plt.plot(t_M3,T6_M3-f(t_M3,*coeffn),"y--",label="Ampl.fkt. unten")
plt.legend(loc="lower center")
plt.xlabel(r"Zeit $t [s]$")
plt.ylabel(r"Temperatur [$^{\circ}{\rm C}$]")
plt.savefig("../Bilder/Normierungsauswahl/M3_Alu_norm.pdf")
plt.show()


coeffx, covar = curve_fit(f,max_M3_T5[0],max_M3_T5[1], p0=(1, 1e-6, 1))
Junk1, t_M3_norm, Junk2 = np.split(t_M3,[120, 900])
Junk1, T5_M3_norm, Junk2= np.split(T5_M3-f(t_M3,*coeffx),[120, 900])
plt.plot(t_M3_norm,T5_M3_norm,"b-",label="Grundschwingung, fern")
coeffx, covar = curve_fit(f,max_M3_T6[0],max_M3_T6[1], p0=(1, 1e-6, 1))
Junk1, T6_M3_norm, Junk2= np.split(T6_M3-f(t_M3,*coeffx),[120, 900])
plt.plot(t_M3_norm,T6_M3_norm,"r-",label="Grundschwingung, nah")
plt.legend(loc="best")
plt.ylim(-16,4)
plt.xlabel(r"Zeit $t [s]$")
plt.ylabel(r"Temperatur [$^{\circ}{\rm C}$]")
max_M3_T5_furt,min_M3_T5_furt = peakdetect.peakdetect(T5_M3_norm,t_M3_norm,1,0.01)
max_M3_T6_furt,min_M3_T6_furt = peakdetect.peakdetect(T6_M3_norm,t_M3_norm,1,0)
max_M3_T6_furt = np.array(max_M3_T6_furt).T
min_M3_T6_furt = np.array(min_M3_T6_furt).T
max_M3_T5_furt = np.array(max_M3_T5_furt).T
min_M3_T5_furt = np.array(min_M3_T5_furt).T
plt.plot(max_M3_T6_furt[0],max_M3_T6_furt[1],"x")
plt.plot(max_M3_T5_furt[0],max_M3_T5_furt[1],"x")
plt.plot(min_M3_T5_furt[0],min_M3_T5_furt[1],"x")
plt.plot(min_M3_T6_furt[0],min_M3_T6_furt[1],"x")

plt.savefig("../Bilder/M3_Alu_norm.pdf")
plt.show()
max_M3_T5_furt_x = max_M3_T5_furt[0] 
max_M3_T5_furt_y = max_M3_T5_furt[1] 
min_M3_T5_furt_x = min_M3_T5_furt[0] 
min_M3_T5_furt_y = min_M3_T5_furt[1] 
min_M3_T6_furt_x = min_M3_T6_furt[0] 
min_M3_T6_furt_y = min_M3_T6_furt[1] 
max_M3_T6_furt_x = max_M3_T6_furt[0] 
max_M3_T6_furt_y = max_M3_T6_furt[1] 
np.savetxt('../Tabellen/M3_Alu_max.txt', np.array([max_M3_T5_furt_x,max_M3_T5_furt_y,max_M3_T6_furt_x,max_M3_T6_furt_y]).T, header="Messung 3 - Maxima\nT5\tT6\nx\t y\t x\t y")
np.savetxt('../Tabellen/M3_Alu_min.txt', np.array([min_M3_T5_furt_x,min_M3_T5_furt_y,min_M3_T6_furt_x,min_M3_T6_furt_y]).T, header="Messung 3 - Minima\nT5\tT6\nx\t y\t x\t y")

#
# Plot 6 - Periodisch Edelstahl
#
coeff, covar = curve_fit(f,max_M3_T7[0],max_M3_T7[1], p0=(1, 1e-6, 1))
plt.plot(max_M3_T7[0],f(max_M3_T7[0],*coeff),"k--", label="Amplitudenfunktion")
coeff, covar = curve_fit(f,min_M3_T7[0],min_M3_T7[1], p0=(1, 1e-6, 1))
plt.plot(min_M3_T7[0],f(min_M3_T7[0],*coeff),"k--")
coeff, covar = curve_fit(f,max_M3_T8[0],max_M3_T8[1], p0=(1, 1e-6, 1))
plt.plot(max_M3_T8[0],f(max_M3_T8[0],*coeff),"k--")
coeff, covar = curve_fit(f,min_M3_T8[0],min_M3_T8[1], p0=(1, 1e-6, 1))
plt.plot(min_M3_T8[0],f(min_M3_T8[0],*coeff),"k--")
plt.plot(t_M3,T7_M3, "r-",label="warmes Ende")
plt.plot(t_M3,T8_M3, "b-",label="kühles Ende")
plt.plot(max_M3_T7[0],max_M3_T7[1],"kx",label="Extrema")
plt.plot(min_M3_T7[0],min_M3_T7[1],"kx")
plt.plot(max_M3_T8[0],max_M3_T8[1],"kx")
plt.plot(min_M3_T8[0],min_M3_T8[1],"kx")
plt.xlabel(r"Zeit $t [s]$")
plt.ylabel(r"Temperatur [$^{\circ}{\rm C}$]")
plt.legend(loc="best")
plt.savefig("../Bilder/M3_Edelstahl.pdf")
plt.show()
coeffx, covar = curve_fit(f,max_M3_T7[0],max_M3_T7[1], p0=(1, 1e-6, 1))
coeffn, covar = curve_fit(f,min_M3_T7[0],min_M3_T7[1], p0=(1, 1e-6, 1))
plt.plot(t_M3,T7_M3-f(t_M3,*coeffx),"g-",label="Ampl.fkt. oben")
plt.plot(t_M3,T7_M3-f(t_M3,*coeffn),"y--")
coeffx, covar = curve_fit(f,max_M3_T8[0],max_M3_T8[1], p0=(1, 1e-6, 1))
coeffn, covar = curve_fit(f,min_M3_T8[0],min_M3_T8[1], p0=(1, 1e-6, 1))
plt.plot(t_M3,T8_M3-f(t_M3,*coeffx),"g-")
plt.plot(t_M3,T8_M3-f(t_M3,*coeffn),"y--",label="Ampl.fkt. unten")
plt.legend(loc="upper left")
plt.ylim(-15,17)
plt.xlabel(r"Zeit $t [s]$")
plt.ylabel(r"Temperatur [$^{\circ}{\rm C}$]")
plt.savefig("../Bilder/Normierungsauswahl/M3_Edelstahl_norm.pdf")
plt.show()
coeffx, covar = curve_fit(f,max_M3_T8[0],max_M3_T8[1], p0=(1, 1e-6, 1))
Junk1, t_M3_norm, Junk2 = np.split(t_M3,[130, 900])
Junk1, T8_M3_norm, Junk2= np.split(T8_M3-f(t_M3,*coeffx),[130, 900])
plt.plot(t_M3_norm,T8_M3_norm,"b-",label="Grundschwingung, fern")
coeffx, covar = curve_fit(f,max_M3_T7[0],max_M3_T7[1], p0=(1, 1e-6, 1))
Junk1, T7_M3_norm, Junk2= np.split(T7_M3-f(t_M3,*coeffx),[130, 900])
plt.plot(t_M3_norm,T7_M3_norm,"r-",label="Grundschwingung, nah")
plt.legend(loc="upper left")
plt.ylim(-12,3)
plt.xlabel(r"Zeit $t [s]$")
plt.ylabel(r"Temperatur [$^{\circ}{\rm C}$]")
max_M3_T8_furt,min_M3_T8_furt = peakdetect.peakdetect(T8_M3_norm,t_M3_norm,1,0.01)
max_M3_T7_furt,min_M3_T7_furt = peakdetect.peakdetect(T7_M3_norm,t_M3_norm,1,0)
max_M3_T7_furt = np.array(max_M3_T7_furt).T
min_M3_T7_furt = np.array(min_M3_T7_furt).T
max_M3_T8_furt = np.array(max_M3_T8_furt).T
min_M3_T8_furt = np.array(min_M3_T8_furt).T
plt.plot(max_M3_T7_furt[0],max_M3_T7_furt[1],"x")
plt.plot(max_M3_T8_furt[0],max_M3_T8_furt[1],"x")
plt.plot(min_M3_T8_furt[0],min_M3_T8_furt[1],"x")
plt.plot(min_M3_T7_furt[0],min_M3_T7_furt[1],"x")
max_M3_T7_furt_x = max_M3_T7_furt[0] 
max_M3_T7_furt_y = max_M3_T7_furt[1] 
min_M3_T7_furt_x = min_M3_T7_furt[0] 
min_M3_T7_furt_y = min_M3_T7_furt[1] 
min_M3_T8_furt_x = min_M3_T8_furt[0] 
min_M3_T8_furt_y = min_M3_T8_furt[1] 
max_M3_T8_furt_x = max_M3_T8_furt[0] 
max_M3_T8_furt_y = max_M3_T8_furt[1] 
np.savetxt('../Tabellen/M3_Edelstahl_max.txt', np.array([max_M3_T7_furt_x,max_M3_T7_furt_y,max_M3_T8_furt_x,max_M3_T8_furt_y]).T, header="Messung 3 - Maxima\nT7\tT8\nx\t y\t x\t y")
np.savetxt('../Tabellen/M3_Edelstahl_min.txt', np.array([min_M3_T7_furt_x,min_M3_T7_furt_y,min_M3_T8_furt_x,min_M3_T8_furt_y]).T, header="Messung 3 - Minima\nT7\tT8\nx\t y\t x\t y")

plt.savefig("../Bilder/M3_Edelstahl_norm.pdf")
plt.show()

