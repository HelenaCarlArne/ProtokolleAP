import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from math import ceil

U,I	=np.genfromtxt("../Werte/Werte_gelb_2.txt").T
I=I*10**(-9)
plt.plot(U,I,"x",label="Messwerte")
plt.yticks([-0.5*10**(-9),0,0.5*10**(-9),1*10**(-9),1.5*10**(-9),2*10**(-9),2.5*10**(-9),3*10**(-9)],[-0.5,0,0.5,1,1.5,2,2.5,3])
plt.ylabel(r"$I_\mathrm{Ph} /(A\cdot10^{-9})$")
plt.xlabel(r"$U /V$")
plt.grid()
plt.legend(loc="best")
plt.savefig("../Bilder/messung2.png")
plt.show()