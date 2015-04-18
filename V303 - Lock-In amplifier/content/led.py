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
	-0.2,-0.1,-0.09, -0.06,-0.045,-0.035,-0.025,-0.01,-0.007,-0.005,-0.003,-0.0025,-0.0012,-0.001])

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

def f(x, a,b):
	return a*np.cos(x/180*np.pi+b)

def exp(x, b, c):
	return b*x**(c)

def exp2(x, b, c,d):
	return b*(x+d)**(c)

def lin(x, d, e):
	return d*x+e

xplot = np.linspace(0.001,360)
# #Ohne Störung
# plt.xlabel(r"Phase [°]")
# plt.ylabel(r"Spannung [V]")
# plt.xlim(0,360)
# plt.xticks([0,90,180,270,360])
# params,covar = curve_fit(f,phase,yvolt, p0=[-3,np.pi])
# plt.plot(xplot, f(xplot, *params),"k",label=r"$f(x)=A \mathrm{cos(x+\alpha)}$")
# plt.plot(xplot, f(xplot, 2*4.63/np.pi,0),"g",label=r"Theorie")
# plt.plot(phase,yvolt,"bx",label="Messdaten")
# print(*params)
# plt.legend(loc="lower center")
# plt.tight_layout
# plt.text(10,5,r"$A=5.813$")
# plt.text(10,4.5,r"$\alpha=180.174$")
# plt.savefig("../Bilder/AusgangSpannung.pdf")
# plt.show()

# #Mit Störung
# plt.xlabel(r"Phase [°]")
# plt.ylabel(r"Spannung [V]")
# plt.xlim(0,360)
# plt.xticks([0,90,180,270,360])
# params,covar = curve_fit(f,phase,-ynoise, p0=[3, np.pi])
# plt.plot(xplot, f(xplot, *params),"k",label=r"$f(x)=A \mathrm{cos(x+\alpha)}$")
# plt.plot(phase,-ynoise,"bx",label="Messdaten")
# plt.plot(xplot, f(xplot, 2*4.63/np.pi,0),"g",label=r"Theorie")
# print(*params)
# plt.legend(loc="lower center")
# plt.tight_layout
# plt.text(10,5,r"$A=5.822$")
# plt.text(10,4.5,r"$\alpha=184.45$")
# plt.savefig("../Bilder/AusgangStoerung.pdf")
# plt.show()

#LED
# plt.ylim(-1,5)
# plt.xlim(0,1.4)
# plt.xlabel(r"Abstand [m]")
# plt.ylabel(r"Negative Spannung [V]")
# params,covar = curve_fit(exp,xled,-yled, p0=[1,-2])
# #plt.plot(xplot, exp(xplot, *params),"b",label=r"$f(x)=A \mathrm{cos(x+\alpha)}$")
# #print(*params)
# plt.plot(xled,-yled,"bx",label="Messdaten")
# plt.legend(loc="best")
# plt.tight_layout
# plt.savefig("../Bilder/LED.pdf")
# plt.show()

plt.ylim(0.0001,10)
plt.xlim(0.01,2)
plt.xlabel(r"Abstand [m]")
plt.ylabel(r"Betrag der Spannung [V]")
plt.yscale("log")
plt.xscale("log")
params,covar = curve_fit(exp,xled,-yled, p0=[0.001, -1.7])
plt.plot(xplot, exp(xplot, *params),"r",label=r"$y(x)=A x^b $")
plt.plot(xled,-yled,"bx",label="Messdaten")
print(*params)
plt.legend(loc="best")
plt.tight_layout
plt.text(0.8,0.8,r"$A=0.0048$")
plt.text(0.8,0.55,r"$b=-1.77$")
plt.savefig("../Bilder/LED_log.pdf")
plt.show()

# plt.ylim(-4.5,1)
# plt.xlim(0,18)
# plt.plot(np.exp((np.log(10)-1)*xled)*np.exp(xled),np.log10(-yled),"b+")
# params1,covar = curve_fit(exp,np.exp(xled),np.log10(-yled))#, p0=[-4.15148415205, 1.60640097894])
# plt.plot(xplot, exp(xplot, *params1),"r",label=r"$y_{\mathrm{lin}}(x)=m_{\mathrm{lin}}x+b_{\mathrm{lin}}$")
# plt.show()

# plt.xlabel(r"Abstand [m]")
# plt.ylabel(r"Negative Spannung [V]")
# plt.yscale("log")
# plt.xscale("log")
# params1,covar = curve_fit(lin,np.exp(xled),np.log(-yled))#, p0=[-4.15148415205, 1.60640097894])
# plt.plot(xplot, exp(xplot, 0.00456203284596,-1.78949240057),"r",label=r"$3 y_{\mathrm{lin}}(x)=m_{\mathrm{lin}}x+b_{\mathrm{lin}}$")
# plt.plot(xled,-yled,"bx",label="Messdaten")
# plt.legend(loc="best")
# print(*params1)
# print(params1[0])
# print(params1[1])
# plt.tight_layout
# plt.text(80,25,r"$m_{\mathrm{lin}}=0.00456$")
# plt.text(80,10,r"$b_{\mathrm{lin}}=-1.79$")
# #plt.savefig("../Bilder/LED_log.pdf")
# plt.show()
