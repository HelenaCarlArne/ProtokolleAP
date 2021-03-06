"""
 Hallo, werter Leser!

 Dies ist die erste Auswerungsdatei von V500:Photoeffekt


 Erforderlich:
 - Pakete: Numpy, matplotlib, Scipy und math
 - Dateien mit Werten Bremsspannung, Photostrom
"""
#
# Header
#
#
# Numpy und matplotlib aus offensichtlichen Gruenden;
# curve_fit aus scipy.optimize fuer das Fitting;
# ceil (aufrunden) aus math fuer das Anpassen der "plotting limits"
#

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from math import ceil

#
# Betriebsfunktionen
#
#
# linear() ist eine lineare Funktion (oh, really?);
# das Dictionary "farben" ist fuer die Schleifen weiter unten definiert;
# das array "zeros" enthaelt die Nullstellen der Fits und wird hier initialisiert,
# aehnliches fuer das array "intersept" und die y-Achsenabschnitte,
# auch fuer die Reihenfolge "order"
#

def linear(x,m,b):
	return m*x+b

farben={"rot":640,"gelb":578,"gruen":546,"violett":435.8,"uv":366}
zeros=np.array([])
zeros_error=np.array([])
intersept=np.array([])
intersept_error=np.array([])
grad=np.array([])
grad_error=np.array([])
order=np.array([])
#
# Daten einladen
# 
# 
# Aus den Wertedateien werden pro Farbe Spannung U und Strom I in 
# die Arrays "u_%farbe" und "i_%farbe" gespeichert.
# Die Strom-Arrays werden gemaess des Vorzeichens mit 10^-12 plutimiziert.
#
# Leider kann hier nicht die "farben"-Schleife laufen, 
# da string-Ersaetze nicht fuer Zuweisungen funktionieren.
#

u_uv,i_uv			=np.genfromtxt("../Werte/Werte_UV.txt").T
u_violett,i_violett	=np.genfromtxt("../Werte/Werte_violett.txt").T
u_gruen,i_gruen		=np.genfromtxt("../Werte/Werte_gruen.txt").T
u_gelb,i_gelb		=np.genfromtxt("../Werte/Werte_gelb.txt").T
u_rot,i_rot			=np.genfromtxt("../Werte/Werte_rot.txt").T

i_rot		=i_rot*10**(-12)
i_gelb		=i_gelb*10**(-12)
i_gruen		=i_gruen*10**(-12)
i_violett	=i_violett*10**(-12)
i_uv		=i_uv*10**(-12)

#
# Funktionen plotten
#
#
# Der erste Schleifenteil plottet die Messwerte,
# der zweite wertet die Fits aus und plottet sie
#
# Erste Schleife: enumerate zaehlt die Eintraege des dictionary "farben" durch (urspruenglich benoetigt!)
#
# %s sind string-Ersaetze unter Python 2 und 3; 
# eval() gibt aus einem string-Satz eine Variable zurueck (bsp. eval("test") -> var: test [kein String]) 
# farben[] benoetigt als Argument "rot","gelb",... fuer die Ausgabe der Wellenlaenge
#

#for quant in {"u","i"}:
for i,var in enumerate(farben):
	#Messwerte plotten
	plt.plot(np.sqrt(eval("i_%s"%(var))*10**12),eval("u_%s"%(var)),"k+",label=r"$\lambda=$ %s nm"%(farben["%s"%(var)]))
	plt.xlabel(r"$\sqrt{I}/ \sqrt{pA}$")
	plt.ylabel(r"$U /V$")
	#Fits berechnen, NST und Fehler berechnen
	parameter, unfug=curve_fit(linear,np.sqrt(eval("i_%s"%(var))*10**12),eval("u_%s"%(var)))
	zero= -parameter[1]/parameter[0]
	errors = np.sqrt(np.diag(unfug))
	#Fits plotten
	x_plot=np.arange(-1,ceil(-parameter[1]/parameter[0])+1)
	plt.xlim(0,ceil(zero))
	plt.ylim(0,ceil(10*parameter[1])/10)
	plt.plot(x_plot,linear(x_plot,*parameter),label="Fit")
	plt.grid()
	plt.legend(loc="best")
	plt.tight_layout()
	plt.savefig("../Bilder/Fit_%s.png"%(var))
	plt.show()
	#Textausgabe und Array-Befuellung
	print("%i.%s:"%(i,var.upper()))
	print(parameter)
	print("NST bei x_%s = \n\t %f+/-%f"%(var,zero,errors[0]))
	zeros=np.append(zero,zeros)
	zeros_error=np.append(errors[0],zeros_error)
	print("Abschnitt bei y_%s = \n\t %f+/-%f\n"%(var,parameter[1],errors[1]))
	grad=np.append(parameter[0],grad)
	grad_error=np.append(errors[0],grad_error)
	intersept=np.append(parameter[1],intersept)
	intersept_error=np.append(errors[1],intersept_error)
	order=np.append(farben["%s"%(var)],order)
print("Und nun alle zusammen:")
print(zip(order,zeros,zeros_error,intersept, intersept_error))
np.savetxt('../Werte/LinPar.txt', np.array([zeros,zeros_error,grad, grad_error,intersept, intersept_error]).T,header="%s\t Alle Werte"%(order))
np.savetxt('../Werte/intersept.txt', np.array([order,intersept, intersept_error]).T,header="%s\t Abschnitte"%(order))