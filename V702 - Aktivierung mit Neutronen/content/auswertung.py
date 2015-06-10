import numpy as np
import matplotlib.pyplot as plt 


N_0=166
n_0=N_0/900
#
##	INDIUM
###
#

N_ind=np.genfromtxt("../Werte/Indium.txt").T 
n_ind=N_ind/250

plt.plot(np.arange(250,len(N_ind)*250+250,250),n_ind,"x",label="Indium")
plt.legend(loc="best")
plt.yscale("log")
plt.show()
#
##	RHODIUM
###
#

N_rho=np.genfromtxt("../Werte/Rhodium.txt").T 
n_rho=N_rho/20

plt.plot(np.arange(20,len(N_rho)*20+20,20),n_rho,"x",label="Rhodium")
plt.yscale("log")
plt.legend(loc="best")
plt.show()