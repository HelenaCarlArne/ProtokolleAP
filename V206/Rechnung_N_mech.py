import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy.constants as const


rho1=(2.6*5.51*273.15)/294.35
rho2=3.2*5.51*273.15/284.65
rho3=3.2*5.51*273.15/274.95
rho4=3.2*5.51*273.15/273.35


N1=(1/(1.14-1))*(7*((2.6/7)**(1/1.14))-2.6)*(1/rho1)*0.001377*100000
N2=(1/(1.14-1))*(9.5*((3.2/9.5)**(1/1.14))-3.2)*(1/rho2)*0.002845*100000
N3=(1/(1.14-1))*(11.5*((3.2/11.5)**(1/1.14))-3.2)*(1/rho3)*0.001927*100000
N4=(1/(1.14-1))*(12.5*((3.2/12.5)**(1/1.14))-3.2)*(1/rho4)*0.0011*100000

print('Dichte1',rho1)
print('Dichte2',rho2)
print('Dichte3',rho3)
print('Dichte4',rho4)


DN1=(1/(1.14-1))*(7*((2.6/7)**(1/1.14))-2.6)*(1/rho1)*0.000512*100000
DN2=(1/(1.14-1))*(9.5*((3.2/9.5)**(1/1.14))-3.2)*(1/rho2)*0.001090*100000
DN3=(1/(1.14-1))*(11.5*((3.2/11.5)**(1/1.14))-3.2)*(1/rho3)*0.001063*100000
DN4=(1/(1.14-1))*(12.5*((3.2/12.5)**(1/1.14))-3.2)*(1/rho4)*0.001100*100000

print('N1',N1,'+-',DN1)
print('N2',N2,'+-',DN2)
print('N3',N3,'+-',DN3)
print('N4',N4,'+-',DN4)

