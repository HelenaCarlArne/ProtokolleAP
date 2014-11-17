import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy.constants as const

plt.rcParams['figure.figsize']= (10, 8)
plt.rcParams['font.size'] = 16

T = np.genfromtxt('Temperatur1_L.txt').T
T = const.C2K(T)
p = np.genfromtxt('Druck_b.txt').T

p = np.log(p)

_T=1/T
def f(_T, a, b):
	return a* _T + b
params, covariance = curve_fit(f, _T, p)
# covariance ist die Kovarianzmatrix

errors = np.sqrt(np.diag(covariance))

# print('a =', params[0], '±', errors[0])
# print('b =', params[1], '±', errors[1])

_T_plot = np.linspace(0.00339616, 0.00308499)


plt.plot(_T_plot, f(_T_plot, *params), 'b-', label='linearer Fit')
plt.legend(loc="best")
plt.plot(_T,p,'r.', label='$ln(p_b)$')

print('ln(p_b)',p)
print('Kehrwert der Temperatur',_T)
plt.xlabel('1/T / s')
plt.ylabel(r'$ln(p_b) / bar$')
plt.title('Dampfdruckkurve')
plt.legend(loc='best')
#plt.show()


L=143.9134164
MD1=1/L*13209*0.015
MD2=1/L*13209*0.031
MD3=1/L*13209*0.021
MD4=1/L*13209*0.011

d_MD1=np.sqrt((13209*0.002/L)**2+(13209*0.015*0.05*10**3/L**2)**2)
d_MD2=np.sqrt((13209*0.005/L)**2+(13209*0.031*0.05*10**3/L**2)**2)
d_MD3=np.sqrt((13209*0.009/L)**2+(13209*0.021*0.05*10**3/L**2)**2)
d_MD4=np.sqrt((13209*0.011/L)**2+(13209*0.011*0.05*10**3/L**2)**2)

print('Massendurchsätze')
print('MD1',MD1,'+/-',d_MD1)
print('MD2',MD2,'+/-',d_MD2)
print('MD3',MD3,'+/-',d_MD3)
print('MD4',MD4,'+/-',d_MD4)

