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
N  = np.genfromtxt('indium.txt', unpack=True)
t=225
#Fehler von n
DN=np.sqrt(N)

N=N/t-null/t_N
DN=np.sqrt((DN/t)**2+(Dnull/t_N)**2)
print('hier bitte nur positive Werte (N):')
print(N)
N=np.log(N)
DN=np.log(DN)

t=np.zeros(17)
for i in range(0, 17):
	t[i]=225+225*i
	
plt.plot(t,N ,'r.', label='Messwerte')
plt.errorbar(t, N, yerr=DN, fmt='rx', label='Fehler')


# Ausgleichsgerade:
def f(t, a, b):
	return a*t+b

params, covariance = curve_fit(f, t, N)
errors = np.sqrt(np.diag(covariance))
print('a =', params[0], 'pm', errors[0])
print('b =', params[1], 'pm', errors[1])

X = np.linspace(0, 4000)
plt.plot(X, f(X, *params), 'b-', label='Ausgleichsgerade')

plt.xlabel('Zeit $t$ in s')
plt.ylabel('Logarithmierte Zaehlrate im Zeitintervall $\Delta T$')
plt.legend(loc="best")
plt.tight_layout()
plt.savefig('plotindium.pdf')