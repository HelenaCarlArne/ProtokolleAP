import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from scipy.optimize import curve_fit


# Werte einlesen
phi_0 = 323.6
beta = np.pi*45/180
Wellenlänge, Farbe, Intensität, Phi = np.genfromtxt('../Werte/daten_HgSpektrum.txt').T

########################
# GITTERKONSTANTE g    #
########################
print('>> Bestimmung der Gitterkonstanten g des Beugungsgitters')

# Auswertung
phi = abs(Phi-phi_0)
phi = np.pi*phi/180
phi = phi-beta
#print('Winkel phi',phi)
sin_phi = np.sin(phi)
#print('Sinus vom Winkel phi',sin_phi)
def f(x, m, b): return m*x+b     
params, cov = curve_fit(f, Wellenlänge, (sin_phi+np.sin(beta)))
m = params[0]
m_err = np.sqrt(cov[0][0])
b = params[1]
b_err = np.sqrt(cov[1][1])
g = 1/m
g_err = m_err/m**2

# Graphen plotten    
x = np.linspace(400, 600, 2)
plt.plot(Wellenlänge, (sin_phi+np.sin(beta)), 'gx', label='Messdaten')
plt.plot(x, f(x, m, b), 'k-', label='lineare Regression')

plt.legend(loc = 'best')
plt.xlabel(r'Wellenlänge $\lambda$ [nm]')
plt.ylabel(r'$\sin(\beta)+\sin(\varphi)$')
#plt.xlim(400, 700)
#plt.grid()
plt.savefig("../Bilder/plot_gitterkonstante.pdf")
print(m,'pm',m_err)
print(b,'pm',b_err)
print(g,'pm',g_err)

Eichung=0.0146853155864
Eichung_err=0
print('Eichung',Eichung)

#============================================#
# Untersuchung verschiedener Alkali-Spektren #
#============================================#
print('>> Untersuchung verschiedener Alkali-Spektren')

# Werte einlesen
h = 4.135e-12
c = 3e13
R = 13605.692
alpha = 7.297e-3
Phi_alkali, s = np.genfromtxt('../Werte/daten_AlkaliSpektren.txt').T
# Auswertung
phi_alkali = abs(Phi_alkali-phi_0)
phi_alkali = np.pi*phi_alkali/180
phi_alkali = phi_alkali-beta
print('phi_alkali',phi_alkali)
print('')
#s = abs(s_1-s_2)
Wellenlänge = (np.sin(phi_alkali)+np.sin(beta))*g 
Wellenlänge_err = (np.sin(phi_alkali)+np.sin(beta))*g_err
Wellenlängenunterschied = s*np.cos(phi_alkali)*Eichung
Wellenlängenunterschied_err = s*np.cos(phi_alkali)*Eichung_err
E = (h*c*Wellenlängenunterschied/Wellenlänge**2)*1e4 
E_err = (np.sqrt(((h*c/Wellenlänge**2)*Wellenlängenunterschied_err)**2+((2*h*c*Wellenlängenunterschied/Wellenlänge**3)*Wellenlänge_err)**2))*1e4 
Elemente = np.array(['Na', ' ', ' ',' ',' ', 'Ka', ' ', ' ', ' ', 'Rb'])
n = np.array([3, 3, 3, 4, 4, 4, 4, 5])
z = np.array([11, 11, 11, 19, 19, 19, 19, 37])
sigma = z-(2*n**3*E/(R*alpha**2))**(1/4)
sigma_err = n**3/(2*R*alpha)*((2*n**3*E)/(R*alpha**2))**(-3/4)*E_err
Na = np.mean(sigma[0:3])
K = np.mean(sigma[3:7])
Rb = np.mean(sigma[7:8])
Na_err = np.sqrt(sum((sigma_err[0:3])**2))/len(sigma[0:3])
K_err = np.sqrt(sum((sigma_err[3:7])**2))/len(sigma[3:7])
Rb_err = np.sqrt(sum((sigma_err[7:8])**2))/len(sigma[7:8])

print('Na:')
print(Na)
print('Fehler:')
print(Na_err)

print('Ka:')
print(K)
print('Fehler:')
print(K_err)

print('Rb:')
print(Rb)
print('Fehler:')
print(Rb_err)

