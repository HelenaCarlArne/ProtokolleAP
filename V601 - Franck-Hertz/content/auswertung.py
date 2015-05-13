import numpy as np 
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
#
##	Kalibrierung
###

x_quant_Ekalt, x_length_Ekalt, x_quant_Ewarm, x_length_Ewarm, x_quant_Ion, x_length_Ion, x_quant_FH, x_length_FH =np.genfromtxt("../Werte/Umrechnung_x.txt").T
y_quant_Ekalt, y_length_Ekalt, y_quant_Ewarm , y_length_Ewarm =np.genfromtxt("../Werte/Umrechnung_y.txt").T

def linear(x,a,b):
	return a*x+b

name = np.array(["Ekalt","Ewarm","Ion","FH"])
""" AUSGABE_ORDNER
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
"""

#
##	Energieverteilung
###

Ediff_y=np.array([])
u_vert_kalt,i_vert_kalt	=np.genfromtxt("../Werte/Vert_kalt.txt").T
u_vert_warm,i_vert_warm	=np.genfromtxt("../Werte/Vert_warm.txt").T

"" "AUSGABE_ORDNER"
for i in range(1,np.size(u_vert_kalt)-1):
	#print((i_vert_kalt[i]-i_vert_kalt[i+1])/(u_vert_kalt[i+1]-u_vert_kalt[i]))
	Ediff_y=np.append(Ediff_y,(i_vert_kalt[i]-i_vert_kalt[i+1])/(u_vert_kalt[i+1]-u_vert_kalt[i]))
plt.plot(Ediff_y,"+")
plt.show()
Ediff_y=np.array([])
for i in range(1,np.size(u_vert_warm)-1):
	#print((i_vert_warm[i]-i_vert_warm[i+1])/(u_vert_warm[i+1]-u_vert_warm[i]))
	Ediff_y=np.append(Ediff_y,(i_vert_warm[i]-i_vert_warm[i+1])/(u_vert_warm[i+1]-u_vert_warm[i]))
plt.plot(Ediff_y,"+")
plt.show()
#print(u_vert_kalt,i_vert_kalt)
#print(u_vert_warm,i_vert_warm)


#print(u_fh,i_fh)
#u_fh,i_fh				=np.genfromtxt("../Werte/FHKurve.txt")
#u_ion,ion				=np.genfromtxt("../Werte/Ion.txt")
#print(u_ion,i_ion)