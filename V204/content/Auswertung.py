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

#
# Extremalwerte für Messung 2
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
plt.title(r"Temperaturwellen in Messing (80s-Periode)")
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
plt.title(r"Temperaturwellen in Messing (80s-Periode)")
plt.xlabel(r"Zeit $t [s]$")
plt.ylabel(r"Temperatur [$^{\circ}{\rm C}$]")
plt.savefig("../Bilder/M2_Messing_warm.pdf")
plt.show()
coeffx, covar = curve_fit(f,max_M2_T1[0],max_M2_T1[1], p0=(1, 1e-6, 1))
coeffn, covar = curve_fit(f,min_M2_T1[0],min_M2_T1[1], p0=(1, 1e-6, 1))
plt.plot(t_M2,T1_M2/f(t_M2,*coeffx),"g-",label="Ampl.fkt. oben")
plt.plot(t_M2,T1_M2/f(t_M2,*coeffn),"y--")
coeffx, covar = curve_fit(f,max_M2_T6[0],max_M2_T6[1], p0=(1, 1e-6, 1))
coeffn, covar = curve_fit(f,min_M2_T6[0],min_M2_T6[1], p0=(1, 1e-6, 1))
plt.plot(t_M2,T6_M2/f(t_M2,*coeffx),"g-")
plt.plot(t_M2,T6_M2/f(t_M2,*coeffn),"y--",label="Ampl.fkt. unten")
plt.legend(loc="best")
plt.title(r"Genormte Temperaturwellen in Messing (80s-Periode)")
plt.xlabel(r"Zeit $t [s]$")
plt.ylabel(r"Temperatur [$^{\circ}{\rm C}$]")
plt.savefig("../Bilder/M2_Messing_norm.pdf")
plt.show()



coeff, covar = curve_fit(f,max_M3_T1[0],max_M3_T1[1], p0=(1, 1e-6, 1))
plt.plot(max_M3_T1[0],f(max_M3_T1[0],*coeff),"k--", label="Amplitudenfunktion")
coeff, covar = curve_fit(f,min_M3_T1[0],min_M3_T1[1], p0=(1, 1e-6, 1))
plt.plot(min_M3_T1[0],f(min_M3_T1[0],*coeff),"k--")
plt.plot(t_M3,T1_M3, "b-",label="kühles Ende")
plt.plot(max_M3_T1[0],max_M3_T1[1],"kx",label="Extrema")
plt.plot(min_M3_T1[0],min_M3_T1[1],"kx")
plt.legend(loc="best")
plt.title(r"Temperaturwellen in Messing (200s-Periode)")
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
plt.legend(loc="best")
plt.title(r"Temperaturwellen in Messing (200s-Periode)")
plt.xlabel(r"Zeit $t [s]$")
plt.ylabel(r"Temperatur [$^{\circ}{\rm C}$]")
plt.savefig("../Bilder/M3_Messing_warm.pdf")
plt.show()
coeffx, covar = curve_fit(f,max_M3_T1[0],max_M3_T1[1], p0=(1, 1e-6, 1))
coeffn, covar = curve_fit(f,min_M3_T1[0],min_M3_T1[1], p0=(1, 1e-6, 1))
plt.plot(t_M3,T1_M3/f(t_M3,*coeffx),"g-",label="Ampl.fkt. oben")
plt.plot(t_M3,T1_M3/f(t_M3,*coeffn),"y--")
coeffx, covar = curve_fit(f,max_M3_T2[0],max_M3_T2[1], p0=(1, 1e-6, 1))
coeffn, covar = curve_fit(f,min_M3_T2[0],min_M3_T2[1], p0=(1, 1e-6, 1))
plt.plot(t_M3,T2_M3/f(t_M3,*coeffx),"g-")
plt.plot(t_M3,T2_M3/f(t_M3,*coeffn),"y--",label="Ampl.fkt. unten")
plt.legend(loc="best")
plt.title(r"Genormte Temperaturwellen in Messing (200s-Periode)")
plt.xlabel(r"Zeit $t [s]$")
plt.ylabel(r"Temperatur [$^{\circ}{\rm C}$]")
plt.savefig("../Bilder/M3_Messing_norm.pdf")
plt.show()


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
plt.legend(loc="best")
plt.title(r"Temperaturwellen in Aluminium (80s-Periode)")
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
plt.legend(loc="best")
plt.title(r"Temperaturwellen in Aluminium (80s-Periode)")
plt.ylabel(r"Temperatur [$^{\circ}{\rm C}$]")
plt.xlabel(r"Zeit $t [s]$")
plt.savefig("../Bilder/M2_Alu_warm.pdf")
plt.show()
coeffx, covar = curve_fit(f,max_M2_T5[0],max_M2_T5[1], p0=(1, 1e-6, 1))
coeffn, covar = curve_fit(f,min_M2_T5[0],min_M2_T5[1], p0=(1, 1e-6, 1))
plt.plot(t_M2,T5_M2/f(t_M2,*coeffx),"g-",label="Ampl.fkt. oben")
plt.plot(t_M2,T5_M2/f(t_M2,*coeffn),"y--")
coeffx, covar = curve_fit(f,max_M2_T6[0],max_M2_T6[1], p0=(1, 1e-6, 1))
coeffn, covar = curve_fit(f,min_M2_T6[0],min_M2_T6[1], p0=(1, 1e-6, 1))
plt.plot(t_M2,T6_M2/f(t_M2,*coeffx),"g-")
plt.plot(t_M2,T6_M2/f(t_M2,*coeffn),"y--",label="Ampl.fkt. unten")
plt.legend(loc="best")
plt.title(r"Genormte Temperaturwellen in Aluminium (80s-Periode)")
plt.xlabel(r"Zeit $t [s]$")
plt.ylabel(r"Temperatur [$^{\circ}{\rm C}$]")
plt.savefig("../Bilder/M2_Alu_norm.pdf")
plt.show()



coeff, covar = curve_fit(f,max_M3_T5[0],max_M3_T5[1], p0=(1, 1e-6, 1))
plt.plot(max_M3_T5[0],f(max_M3_T5[0],*coeff),"k--", label="Amplitudenfunktion")
coeff, covar = curve_fit(f,min_M3_T5[0],min_M3_T5[1], p0=(1, 1e-6, 1))
plt.plot(min_M3_T5[0],f(min_M3_T5[0],*coeff),"k--")
plt.plot(t_M3,T5_M3, "b-",label="kühles Ende")
plt.plot(max_M3_T5[0],max_M3_T5[1],"kx",label="Extrema")
plt.plot(min_M3_T5[0],min_M3_T5[1],"kx")
plt.legend(loc="best")
plt.title(r"Temperaturwellen in Aluminium (200s-Periode)")
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
plt.legend(loc="best")
plt.title(r"Temperaturwellen in Aluminium (200s-Periode)")
plt.xlabel(r"Zeit $t [s]$")
plt.ylabel(r"Temperatur [$^{\circ}{\rm C}$]")
plt.savefig("../Bilder/M3_Alu_warm.pdf")
plt.show()
coeffx, covar = curve_fit(f,max_M3_T5[0],max_M3_T5[1], p0=(1, 1e-6, 1))
coeffn, covar = curve_fit(f,min_M3_T5[0],min_M3_T5[1], p0=(1, 1e-6, 1))
plt.plot(t_M3,T5_M3/f(t_M3,*coeffx),"g-",label="Ampl.fkt. oben")
plt.plot(t_M3,T5_M3/f(t_M3,*coeffn),"y--")
coeffx, covar = curve_fit(f,max_M3_T6[0],max_M3_T6[1], p0=(1, 1e-6, 1))
coeffn, covar = curve_fit(f,min_M3_T6[0],min_M3_T6[1], p0=(1, 1e-6, 1))
plt.plot(t_M3,T6_M3/f(t_M3,*coeffx),"g-")
plt.plot(t_M3,T6_M3/f(t_M3,*coeffn),"y--",label="Ampl.fkt. unten")
plt.legend(loc="best")
plt.title(r"Genormte Temperaturwellen in Aluminium (200s-Periode)")
plt.xlabel(r"Zeit $t [s]$")
plt.ylabel(r"Temperatur [$^{\circ}{\rm C}$]")
plt.savefig("../Bilder/M3_Alu_norm.pdf")
plt.show()


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
plt.title(r"Temperaturwellen in Edelstahl (200s-Periode)")
plt.savefig("../Bilder/M3_Edelstahl.pdf")
plt.show()
coeffx, covar = curve_fit(f,max_M3_T7[0],max_M3_T7[1], p0=(1, 1e-6, 1))
coeffn, covar = curve_fit(f,min_M3_T7[0],min_M3_T7[1], p0=(1, 1e-6, 1))
plt.plot(t_M3,T7_M3/f(t_M3,*coeffx),"g-",label="Ampl.fkt. oben")
plt.plot(t_M3,T7_M3/f(t_M3,*coeffn),"y--")
plt.plot(peakdetect.peakdetect(T1_M2,t_M2,1,0))
coeffx, covar = curve_fit(f,max_M3_T8[0],max_M3_T8[1], p0=(1, 1e-6, 1))
coeffn, covar = curve_fit(f,min_M3_T8[0],min_M3_T8[1], p0=(1, 1e-6, 1))
plt.plot(t_M3,T8_M3/f(t_M3,*coeffx),"g-")
plt.plot(t_M3,T8_M3/f(t_M3,*coeffn),"y--",label="Ampl.fkt. unten")
plt.legend(loc="best")
plt.title(r"Genormte Temperaturwellen in Edelstahl (200s-Periode)")
plt.xlabel(r"Zeit $t [s]$")
plt.ylabel(r"Temperatur [$^{\circ}{\rm C}$]")
#plt.savefig("../Bilder/M3_Edelstahl_norm.pdf")
plt.show()