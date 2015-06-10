import matplotlib as mpl
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Nullmessung pro Sekunde:
null=189
t_N=900
#Fehler 
Dnull=np.sqrt(null)

#Einlesen der Werte
N  = np.genfromtxt('silber.txt', unpack=True)
t=9
#Fehler von n
DN=np.sqrt(N)

N=N/t-null/t_N
DN=np.sqrt((DN/t)**2+(Dnull/t_N)**2)
print('hier bitte nur positive Werte (N):')
print(N)
N=np.log(N)
DN=np.log(DN)


t=np.zeros(47)
for i in range(0, 47):
	t[i]=9+9*i


plt.errorbar(t, N, yerr=DN, fmt='rx', label='Messwerte - Nullmessung')
#_____________________________________________________________
#Einlesen der Werte
N  = np.genfromtxt('langlebigsilber.txt', unpack=True)
t=9
#Fehler von n
DN=np.sqrt(N)

N=N/t-null/t_N
DN=np.sqrt((DN/t)**2+(Dnull/t_N)**2)
print('hier bitte nur positive Werte (N):')
print(N)
N=np.log(N)
DN=np.log(DN)
t=np.zeros(23)
for i in range(0, 23):
	t[i]=9*23+9*i

#langlebige Ausgleichsgerade:
def f(t, a, b):
	return a*t+b

params, covariance = curve_fit(f, t, N)
errors = np.sqrt(np.diag(covariance))
#Ausgleichsgerade:
print('Ausgleichsgerade')
print('a =', params[0], 'pm', errors[0])   #a ist -lambda
print('b =', params[1], 'pm', errors[1])



X = np.linspace(0, 450)
plt.plot(X, f(X, *params), 'b-', label='langlebige Ausgleichsgerade')
plt.axvline(216+4.5, linestyle='--', color='black', label='t*')
plt.xlabel('Zeit $t$ in s')
plt.ylabel('logarithmierte Zaehlrate im Zeitintervall $\Delta T$')
plt.legend(loc="best")
plt.tight_layout()
plt.savefig('plotsilber1.pdf')













