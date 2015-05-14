# -*- coding: utf-8 -*-
# 
#
#
#
#
#
#
#

import numpy as np 
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from uncertainties import unumpy as unp
from uncertainties import ufloat
#
##	Kalibrierung
###

x_quant_Ekalt, x_length_Ekalt, x_quant_Ewarm, x_length_Ewarm, x_quant_Ion, x_length_Ion, x_quant_FH, x_length_FH =np.genfromtxt("../Werte/Umrechnung_x.txt").T
y_quant_Ekalt, y_length_Ekalt, y_quant_Ewarm , y_length_Ewarm =np.genfromtxt("../Werte/Umrechnung_y.txt").T

def linear(x,a,b):
	return a*x+b

name = np.array(["Ekalt","Ewarm","Ion","FH"])

for v in name:
	parameter, coeffmatrix =curve_fit(linear, eval("x_length_%s"%(v)),eval("x_quant_%s"%(v)))
	error=np.sqrt(np.diag(coeffmatrix))
	print(v)
	print("{0:1.4f}+/-{1:1.4f}\n{2:1.4f}+/-{3:1.4f}\n".format(parameter[0],error[0],parameter[1],error[1]))

print('*********************************************')
name = np.array(["Ekalt","Ewarm"])

for v in name:
	parameter, coeffmatrix =curve_fit(linear, eval("y_length_%s"%(v)),eval("y_quant_%s"%(v)))
	print(v)
	print("{0:1.4f}\n{1:1.4f}\n".format(parameter[0],parameter[1]))

print('*********************************************')
print('*********************************************')



#
##	Energieverteilung, kalt
###

"" "AUSGABE_ORDNER"
Ediff_y=np.array([])
u_vert_kalt,i_vert_kalt	=np.genfromtxt("../Werte/Vert_kalt.txt").T

plt.plot(u_vert_kalt,i_vert_kalt,"x",label="Messdaten")
plt.xlabel(r"$\mathrm{Diagramml\ddot{a}nge\,in}\,cm$")
plt.ylabel(r"$\mathrm{Diagramml\ddot{a}nge\,in}\,cm$")
plt.legend()
plt.savefig("../Bilder/Vert_kalt.pdf")
plt.show()


for i in range(0,np.size(u_vert_kalt)-1):
	#print((i_vert_kalt[i]-i_vert_kalt[i+1])/(u_vert_kalt[i+1]-u_vert_kalt[i]))
	Ediff_y=np.append(Ediff_y,(i_vert_kalt[i]-i_vert_kalt[i+1])/(u_vert_kalt[i+1]-u_vert_kalt[i]))

for i in range(0,np.size(u_vert_kalt)-1):
	u_vert_kalt[i]=(u_vert_kalt[i+1]+u_vert_kalt[i])/2

plt.plot(np.delete(u_vert_kalt,-1),Ediff_y,"+",label="Errechnete Steigung")
parameter,coeffmatrix=curve_fit(linear,u_vert_kalt[-9:-1],Ediff_y[-9:-1])

plt.plot(u_vert_kalt[-13:-1],linear(u_vert_kalt[-13:-1],*parameter),label="Fit der fallenden Flanke")
plt.xlim(0,27)
plt.ylim(0,1.6)
plt.xlabel(r"$\mathrm{Diagramml\ddot{a}nge\,in}\,cm$")
plt.ylabel(r"$\mathrm{\ddot{A}nderung\,der\,Diagramml\ddot{a}nge\,in}\,cm$")
plt.legend(loc="best")
plt.savefig("../Bilder/Vert_kalt_diff.pdf")
plt.show()

errors=np.sqrt(np.diag(coeffmatrix))
parameter=unp.uarray(parameter,errors)
print("Die Nullstelle der Tangente ist:")
print(-parameter[1]/parameter[0])
print(ufloat(0.4036,0.0044)*-parameter[1]/parameter[0]+ufloat(-0.1246,0.0579))

#ufloat{0.4036,0.0044}
#ufloat{-0.1246,0.0579}

"" "AUSGABE_ORDNER"


#
##	Energieverteilung, warm
###

Ediff_y=np.array([])
u_vert_warm,i_vert_warm	=np.genfromtxt("../Werte/Vert_warm.txt").T

plt.plot(u_vert_warm,i_vert_warm,"x",label="Messdaten")
plt.xlabel(r"$\mathrm{Diagramml\ddot{a}nge\,in}\,cm$")
plt.ylabel(r"$\mathrm{Diagramml\ddot{a}nge\,in}\,cm$")
plt.legend()
plt.savefig("../Bilder/Vert_warm.pdf")
plt.show()


for i in range(0,np.size(u_vert_warm)-1):
	#print((i_vert_warm[i]-i_vert_warm[i+1])/(u_vert_warm[i+1]-u_vert_warm[i]))
	Ediff_y=np.append(Ediff_y,(i_vert_warm[i]-i_vert_warm[i+1])/(u_vert_warm[i+1]-u_vert_warm[i]))

for i in range(0,np.size(u_vert_warm)-1):
	u_vert_warm[i]=(u_vert_warm[i+1]+u_vert_warm[i])/2

plt.plot(np.delete(u_vert_warm,-1),Ediff_y,"+",label="Errechnete Steigung")
plt.xlabel(r"$\mathrm{Diagramml\ddot{a}nge\,in}\,cm$")
plt.ylabel(r"$\mathrm{\ddot{A}nderung\,der\,Diagramml\ddot{a}nge\,in}\,cm$")

parameter,coeffmatrix=curve_fit(linear,np.delete(u_vert_warm,[23,24,25,26,27,28,29,30]),np.delete(Ediff_y,[23,24,25,26,27,28,29]))
plt.plot(np.delete(u_vert_warm,[24,25,26,27,28,29,30]),linear(np.delete(u_vert_warm,[24,25,26,27,28,29,30]),*parameter),label="Fit der fallenden Flanke")
plt.xlim(0,25)
plt.ylim(0,2)
plt.legend()
plt.savefig("../Bilder/Vert_warm_diff.pdf")
plt.show()

errors=np.sqrt(np.diag(coeffmatrix))
parameter=unp.uarray(parameter,errors)
print("Die Nullstelle der Tangente ist:")
print(-parameter[1]/parameter[0])
print(ufloat(0.4047,0.0040)*-parameter[1]/parameter[0]+ufloat(0.0154,0.0594))

#ufloat{0.4047,0.0040}
#ufloat{0.0154,0.0594}

#
##	Energieverteilung, warm
###

#print(u_fh,i_fh)
#u_fh,i_fh				=np.genfromtxt("../Werte/FHKurve.txt")
#u_ion,ion				=np.genfromtxt("../Werte/Ion.txt")
#print(u_ion,i_ion)