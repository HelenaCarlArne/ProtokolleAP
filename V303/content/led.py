import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

xled=np.array([
	0.02,
	0.05,0.08,0.11,
	0.14,0.17,0.20,0.23,0.26,0.29,0.39,0.49,0.59,0.69,0.79,0.99,1.19])

yled=np.array([
	-5,-1,-0.5, 
	-0.2,-0.1,-0.09, -0.006,-0.0045,-0.0035,-0.0025,-0.001,-0.0007,-0.0005,-0.0003,-0.00025,-0.00012,-0.0001])

phase=np.array([
	0,	
	45	,
	90	,
	120,
	135,
	180,
	225,
	270,
	315,
	360])

yvolt=np.array([
	-6.00	,
	-4.00	,
	0.20	,
	2.62	,
	4.25	,
	5.81	,
	3.95	,
	0.20	,
	-4.17	,
	-5.83])	

ynoise=np.array([
	6.00,
	4.00,
	-0.50,
	-3.00,
	-4.50,
	-6.00,
	-3.50,
	0.50,
	4.50,
	5.50])

def f(x, a):
	return a*np.cos(x/180*np.pi)
xplot = np.linspace(0,360)

#Ohne Störung
plt.xlabel(r"Phase [°]")
plt.ylabel(r"Spannung [V]")
plt.xlim(0,360)
plt.xticks([0,90,180,270,360])
params,covar = curve_fit(f,phase,yvolt, p0=[-3])
plt.plot(xplot, f(xplot, params),"k",label=r"$f(x)=A \mathrm{cos(x+\alpha)}$")
plt.plot(phase,yvolt,"bx",label="Messdaten")
plt.legend(loc="lower center")
plt.tight_layout
plt.savefig("../Bilder/AusgangSpannung.pdf")
plt.close()

#Mit Störung
plt.xlabel(r"Phase [°]")
plt.ylabel(r"Spannung [V]")
plt.xlim(0,360)
plt.xticks([0,90,180,270,360])
params,covar = curve_fit(f,phase,-ynoise, p0=[3])
plt.plot(xplot, f(xplot, params),"k",label=r"$f(x)=A \mathrm{cos(x+\alpha)}$")
plt.plot(phase,-ynoise,"bx",label="Messdaten")
plt.legend(loc="lower center")
plt.tight_layout
plt.savefig("../Bilder/AusgangStoerung.pdf")
plt.close()

#LED
plt.ylim(-1,5)
plt.xlabel(r"Abstand [m]")
plt.ylabel(r"Negative Spannung [V]")
plt.plot(xled,-yled,"bx",label="Messdaten")
plt.legend(loc="best")
plt.tight_layout
plt.savefig("../Bilder/LED.pdf")
plt.close()

plt.xlabel(r"Abstand [m]")
plt.ylabel(r"Negative Spannung [V]")
plt.yscale("log")
plt.xscale("log")
plt.plot(xled,-yled,"bx",label="Messdaten")
plt.legend(loc="best")
plt.tight_layout
plt.savefig("../Bilder/LED_log.pdf")
plt.close()
