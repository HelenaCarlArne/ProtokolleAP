import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy.constants as const
import uncertainties
from uncertainties import ufloat
import numpy as np 									                                                    
import matplotlib as mpl
import uncertainties.unumpy as unp
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as stds
from scipy.stats import sem
from numpy import *
from matplotlib.pyplot import *


#########################
#	DATEN EINLESEN	#
#########################

g1, b1, g2, b2  = np.genfromtxt('Werte_M1_M2.txt').T

g2_ohnenull=np.delete(g2,10)
b2_ohnenull=np.delete(b2,10)
ew, g31, b31, b32, g32 = np.genfromtxt('Werte_Bessel_w.txt').T

g31r, b31r, g32r, b32r, g31b, b31b, g32b, b32b, ef = np.genfromtxt('Werte_Bessel_b.txt').T

B_, g_ , b_ = np.genfromtxt('Werte_Abbe.txt').T


#################################
#	MESSUNG1 UND MESSUNG2	#
#################################

###### 	MITTELWERTE + FEHLER #####
g_1 = ufloat(np.mean(g1),sem(g1))  
b_1 = ufloat(np.mean(b1),sem(b1))

g_2 = ufloat(np.mean(g2_ohnenull),sem(g2_ohnenull))
b_2 = ufloat(np.mean(b2_ohnenull),sem(b2_ohnenull))

################PLOT1##################
nullen = zeros(len(g1))

plot(g1,nullen, "*")
plot(nullen,b1,"*") 

#Geraden durch die Punkte
def g(x,m,b):
    return m*x +b

m_1 = -b1 / g1

i = 1

# Ein Geradenplot wird ausserhalb der Schleife durchgefuehrt, um das Label fuer die Legende zu setzen!
I = linspace(0 , g1[1], 2)
plot(I, g(I,m_1[1],b1[1]), 'b', label = r"Linse mit Brennweite $f = 100\,\mathrm{mm}$")

for i in range(0 , len(g1)):
   I = linspace(0 , g1[i], 2)
   plot(I, g(I,m_1[i],b1[i]), 'b') 


plot([4.84], [4.84], 'g.', markersize=15.0)
plt.xlim(0,0.23)
plt.ylim(0, 0.55)
plt.xlabel('Gegenstandsweite $g$ /m')
plt.ylabel('Bildweite $b$ /m')
plt.legend(loc="best")
plt.tight_layout
plt.savefig('../Bilder/Messung1.pdf')
plt.show()

##################PLOT2#############
nullen = zeros(len(g2))

plot(g2,nullen, "*")
plot(nullen,b2,"*") 

#Geraden durch die Punkte
def g(x,m,b):
    return m*x +b

m_2 = -b2 / g2

i = 1

# Ein Geradenplot wird ausserhalb der Schleife durchgefuehrt, um das Label fuer die Legende zu setzen!
I = linspace(0 , g2[2], 2)
plot(I, g(I,m_2[2],b2[2]), 'b', label = r"Linse mit Brennweite $f = 50\,\mathrm{mm}$")

for i in range(0 , len(g2)):
   I = linspace(0 , g2[i], 2)
   plot(I, g(I,m_2[i],b2[i]), 'b') 


plot([4.84], [4.84], 'g.', markersize=15.0)
plt.xlim(0,0.17)
plt.ylim(0, 0.3)
plt.xlabel('Gegenstandsweite $g$ /m')
plt.ylabel('Bildweite $b$ /m')
plt.legend(loc="best")
plt.tight_layout
plt.savefig('../Bilder/Messung2.pdf')
plt.show()

######BERECHNUNG DER BRENNWEITEN#####
a=1/g_1 + 1/b_1 #Brennweite Messung 1
b=1/g_2 + 1/b_2 #Brennweite Messung 2
# Einzelbrennweite
a=1/g1 + 1/b1 #Brennweite Messung 1
b=1/g2_ohnenull + 1/b2_ohnenull #Brennweite Messung 2
f1=1/a
f2=1/b
print(ufloat(np.mean(f1),sem(f1)),ufloat(np.mean(f2),sem(f2)))
print(f2)
#####AUSGABE#####
print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
print('')
print('Messung 1 und 2 mit f1=100 1/mm und f2=50 1/mm')
print('')
print('Brennweite Messung1',f1,'/m')
print('Brennweite Messung2',f2,'/m')


#################################
#	MESSUNG3 - BESSEL	#
#################################

####BERECHNUNG DER BRENNWEITEN FueR WEISSES, ROTES UND BLAUES LICHT######
f31=(ew**2-(g31-b31)**2)/(4*ew)
print('f31',f31)
f31 = ufloat(np.mean(f31),sem(f31))
f32=(ew**2-(g32-b32)**2)/(4*ew)
print('f32',f32)
f32 = ufloat(np.mean(f32),sem(f32))

f31r=(ef**2-(g31r-b31r)**2)/(4*ef)
print('f31r',f31r)
f31r = ufloat(np.mean(f31r),sem(f31r))
f32r=(ef**2-(g32r-b32r)**2)/(4*ef)
print('f32r',f32r)
f32r = ufloat(np.mean(f32r),sem(f32r))

f31b=(ef**2-(abs(g31b-b31b))**2)/(4*ef)
print('f31b',f31b)
f31b = ufloat(np.mean(f31b),sem(f31b))
f32b=(ef**2-(abs(g32b-b32b))**2)/(4*ef)
print('f32b',f32b)
f32b = ufloat(np.mean(f32b),sem(f32b))

print(f31)
print(f32)
print((f31+f32)/2)
print(f31r)
print(f31b)
print(f32r)
print(f32b)
print((f31r+f32r)/2)
print((f31b+f32b)/2)
print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
print('')
print('Messung 3 nach Bessel mit f=100 1/mm')
print('')
print('Brennweite fuer weisses Licht 1 f=',f31,'1/m')
#print('Brennweite fuer weisses Licht 2 f=',f32,'1/m')
print('Brennweite fuer rotes Licht 1 f=',f31r,'1/m')
#print('Brennweite fuer rotes Licht 2 f=',f32r,'1/m')
print('Brennweite fuer blaues Licht 1 f=',f31b,'1/m')
#print('Brennweite fuer blaues Licht 2 f=',f32b,'1/m')


#################################
#	MESSUNG4 - ABBE		#
#################################

G=30
g=g_*10**3
b=b_*1e3
B=B_*1e3

# Auswertung
V = B/G
def f(x, m, b): return m*x+b     
params_1, cov_1 = curve_fit(f, 1+1/V, g)
params_2, cov_2 = curve_fit(f, 1+V, b)
f_1 = params_1[0] # <-- aus den Gegenstandsweiten
f_2 = params_2[0] # <-- aus den Bildweiten
h_1 = params_1[1]
h_2 = params_2[1]
df_1 = np.sqrt(cov_1[0][0])
df_2 = np.sqrt(cov_2[0][0])
dh_1 = np.sqrt(cov_1[1][1])
dh_2 = np.sqrt(cov_2[1][1])

# Graphen plotten    
x = np.linspace(min(V)+100, max(V)-2.35, 2)
plt.plot((1+1/V), g, 'rx', label='Messdaten')
plt.plot((1+1/x), f(1+1/x, f_1, h_1), 'k-', label='Ausgleichsgerade')
plt.legend(loc = 'best')
plt.xlabel(r'$1+\frac{1}{V}$')
plt.ylabel('Gegenstandsweite $g\'$ [mm]')
plt.xlim(min(1+1/V)-0.3, max(1+1/V)+0.3)
plt.savefig('../Bilder/Abbe1.pdf')
plt.close()

x = np.linspace(min(V)-1, max(V)+1, 2)
plt.plot((1+V), b, 'bx', label='Messdaten')
plt.plot((1+x), f(1+x, f_2, h_2), 'k-', label='Ausgleichsgerade')
plt.legend(loc = 'upper left')
plt.xlabel(r'$1+V$')
plt.ylabel('Bildweite $b\'$ [mm]')
plt.xlim(min(1+V)-0.5, max(1+V)+0.5)
plt.ylim(min(f(1+V, f_2, h_2))-100, max(f(1+V, f_2, h_2))+100)
plt.savefig('../Bilder/Abbe2.pdf')
plt.close()
print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
print('')
print('Messung nach ABBE')
print('')
print('V',V,'mm')
print('f1',f_1,'+/-',df_1,'mm')
print('f2',f_2,'+/-',df_2,'mm')
print('h1',h_1,'+/-',dh_1,'mm')
print('h2',h_2,'+/-',dh_2,'mm')