import numpy as np
from uncertainties import unumpy as unp
from scipy.stats import sem
from matplotlib import pyplot as plt
from uncertainties.unumpy import (nominal_values as noms,std_devs as stds)
from scipy.optimize import curve_fit
from uncertainties import ufloat
#
## Abmessungen des Blocks
###
#

height,width,depth=np.genfromtxt("../Werte/Acrylblock.txt").T
height,width,depth=height*10**(-2),width*10**(-2),depth*10**(-2)
height=unp.uarray(np.mean(height),sem(height))
width=unp.uarray(np.mean(width),sem(width))
depth=unp.uarray(np.mean(depth),sem(depth))
print("####################################################")
print("####################################################")
print("Abmessungen des Blocks")
print("""
Hoehe:\t{}
Breite:\t{}
Tiefe:\t{}
	""".format(height,width,depth))
print("")
print("####################################################")
#
## Abmessungen der Zylinder
###
#

large,medium,small=np.genfromtxt("../Werte/Zylinder.txt").T
large=unp.uarray(np.mean(large),sem(large))
medium=unp.uarray(np.mean(medium),sem(medium))
small=unp.uarray(np.mean(small),sem(small))
print("Abmessungen der Zylinder")
print("""
Gross:\t{}
Mittel:\t{}
Klein:\t{}
	""".format(large,medium,small))
print("")
print("####################################################")
large,medium,small=large*10**(-2),medium*10**(-2),small*10**(-2)

#
## Abmessungen der Loecher
###
#

set_1,set_2,set_3,set_4,set_5,set_6,set_7,set_8,set_9,set_10,set_11=np.genfromtxt("../Werte/Loecherabmessungen.txt")


#for i in range(1,12):
#	print(unp.uarray(np.mean(eval("set_%s"%(i))),sem(eval("set_%s"%(i)))))

print("Abmessungen der Zylinderloecher")
print("###")
#for i in range(1,12):
#	print(eval("set_%s[0]"%i))
#	print(eval("set_%s[1]"%i))
#	print(eval("set_%s[2]"%i))
#	print(eval("set_%s[3]"%i))
#	print("")
print("")
print("####################################################")
#
## Bestimmung der Schallgeschwindigkeit, die Erste
###
#

print("Plots...")
print("und alle so 'Yeah!'")
print("")
parameter=np.array([])
x_plot=np.linspace(0,5*10**-5)
def linear(x,a,b):
	return a*x+b

rot,blau,rot_durchgehend=np.genfromtxt("../Werte/c_Zylinder.txt")
rot,blau,rot_durchgehend=rot*10**(-6),blau*10**(-6),rot_durchgehend*10**(-6)

parms,coeff=curve_fit(linear,rot/2,[noms(small),noms(medium),noms(large)])
plt.plot(x_plot,linear(x_plot,*parms),"r")
plt.plot(rot/2,[noms(small),noms(medium),noms(large)],"rx")
parameter=np.append(parameter,ufloat(parms[0],np.sqrt(np.diag(coeff)[0])))
parameter=np.append(parameter,ufloat(parms[1],np.sqrt(np.diag(coeff)[1])))

parms,coeff=curve_fit(linear,blau/2,[noms(small),noms(medium),noms(large)])
plt.plot(x_plot,linear(x_plot,*parms),"b")
plt.plot(blau/2,[noms(small),noms(medium),noms(large)],"bx")
parameter=np.append(parameter,ufloat(parms[0],np.sqrt(np.diag(coeff)[0])))
parameter=np.append(parameter,ufloat(parms[1],np.sqrt(np.diag(coeff)[1])))

parms,coeff=curve_fit(linear,rot_durchgehend,[noms(small),noms(medium),noms(large)])
plt.plot(x_plot,linear(x_plot,*parms),"k")
plt.plot(rot_durchgehend,[noms(small),noms(medium),noms(large)],"kx")
parameter=np.append(parameter,ufloat(parms[0],np.sqrt(np.diag(coeff)[0])))
parameter=np.append(parameter,ufloat(parms[1],np.sqrt(np.diag(coeff)[1])))
print("Parameter:")
print(parameter)
print("Nullstellen:")
print(-parameter[1]/parameter[0])
print(-parameter[3]/parameter[2])
print(-parameter[5]/parameter[4])
cacryl=(parameter[0]+parameter[2]+parameter[4])/3
print("Geschwindigkeit:")
print(cacryl)
print("")

plt.xticks([0,0.5*10**-5,1*10**-5,1.5*10**-5,2*10**-5,2.5*10**-5,3*10**-5,3.5*10**-5,4*10**-5,4.5*10**-5,5*10**-5],[0,10,15,20,25,30,35,40,45,50])
plt.ylim(0,0.13)
plt.ylabel("Laufzeitstrecke in m")
plt.xlabel("Laufzeit in Mikrosekunden")
plt.xlim(0,4.5*10**-5)
#plt.xticks([1.5*10**-5,2*10**-5,2.5*10**-5,3*10**-5,3.5*10**-5,4*10**-5,4.5*10**-5],[15,20,25,30,35,40,45])
plt.savefig("../Bilder/Schallgeschwindigkeit.pdf")
plt.close()
print("####################################################")
#rot,blau,rot_durchgehend=np.genfromtxt("../Werte/c_Zylinder.txt")
#rot,blau,rot_durchgehend=rot*10**(-3),blau*10**(-3),rot_durchgehend*10**(-3)
#print("Schallgeschwindigkeiten")
def c(k,s,t):
	return k*s/t
def s(k,c,t):
	return k*c*t

#print("Zunaechst Rot, danach Blau, dann die Differenz, darauf die Geschwindigkeit ohne Laufzeitfehler, abschliessend Durchgehend-Rot")
#print("")
#print("")
#print("Klein")
#print(c(2,small, rot[0]))
#print(c(2,small, blau[0]))
#print(c(1,small, rot_durchgehend[0]))
#print("")
#print("")
#print("Mittel")
#print(c(2,medium, rot[1]))
#print(c(2,medium, blau[1]))
#print(c(1,medium, rot_durchgehend[1]))
#print("")
#print("")
#print("Gross")
#print(c(2,large, rot[2]))
#print(c(2,large, blau[2]))
#print(c(1,large, rot_durchgehend[2]))
#print("")


#
## Tiefe der Stoerstellen
###
#

time_1,time_2,time_3,time_4,time_5,time_6,time_7,time_8,time_9,time_10,time_11=np.genfromtxt("../Werte/Acrylblock_Zeiten.txt")
time_1,time_2,time_3,time_4,time_5,time_6,time_7,time_8,time_9,time_10,time_11=time_1*10**-4,time_2*10**-4,time_3*10**-4,time_4*10**-4,time_5*10**-4,time_6*10**-4,time_7*10**-4,time_8*10**-4,time_9*10**-4,time_10*10**-4,time_11*10**-4

print("Tiefe der Stoerstellen in cm")
print("""Angabe:
	Oben: Messung, Realitaet, Abweichung in Prozent
	Unten: Messung, Realitaet, Abweichung in Prozent
	Durchmesser: Differenz aus Messung, Realitaet, Abweichung in Prozent""")
print("")
for i in range(1,12):
	print("Nummer %s"%i)
	print("Von oben:")
	print(0.5*cacryl*eval("time_%s[1]"%i))
	#print(eval("set_%s[2]"%i))
	print((0.5*cacryl*eval("time_%s[1]"%i)-eval("set_%s[2]"%i))/eval("set_%s[2]"%i)*100)
	#print("Von unten:")
	print(0.5*cacryl*eval("time_%s[2]"%i))
	#print(eval("set_%s[3]"%i))
	print((0.5*cacryl*eval("time_%s[2]"%i)-eval("set_%s[3]"%i))/eval("set_%s[3]"%i)*100)
	#print("Durchmesser:")
	print(abs(0.5*cacryl*eval("time_%s[2]"%i)-0.5*cacryl*eval("time_%s[1]"%i)))
	#print(eval("set_%s[1]"%i))
	print(abs((0.5*cacryl*eval("time_%s[2]"%i)-0.5*cacryl*eval("time_%s[1]"%i)-eval("set_%s[1]"%i)))/eval("set_%s[1]"%i)*100)
	print("")

print("######################################")
print(cacryl*5.5e-4)
print(2500*7.6e-4)
print(1410*45e-4)
print("")
print(cacryl*5.5e-4/3)
print(2500*7.6e-4/3)
print(1410*45e-4/3)