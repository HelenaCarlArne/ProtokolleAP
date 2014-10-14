import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from uncertainties import ufloat
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 16
import numpy as np

d = np.genfromtxt('Durchmesser.txt').T
s1 = np.genfromtxt('Seite1.txt').T
s2 = np.genfromtxt('Seite2.txt').T

durch = ufloat(np.mean(d),np.std(d))
seite1 = ufloat(np.mean(s1),np.std(s1))
seite2 = ufloat(np.mean(s2),np.std(s2))


print("Der Durchmesser betraegt:",durch,"mm")
print("Die eine Seite betraegt:",seite1,"mm")
print("Der Durchmesser betraegt:",seite2,"mm")