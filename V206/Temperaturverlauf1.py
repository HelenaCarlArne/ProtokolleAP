import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy.constants as const

plt.rcParams['figure.figsize']= (10, 8)
plt.rcParams['font.size'] = 16

t, T_1 = np.genfromtxt('Temperatur_1.txt').T
T_1 = const.C2K(T_1)
T_2 = np.genfromtxt('Temperatur_2.txt').T
T_2 = const.C2K(T_2)

plt.plot(t,T_1,'r.', label='$T_1$')
plt.plot(t,T_2,'b.',  label='$T_2$')
plt.xlabel('Zeit t  / s')
plt.ylabel('Temperatur T  / K')
plt.title('Temperaturverlauf')
plt.legend(loc='best')
plt.show()

#plt.savefig('Temperaturverlauf.pdf')

