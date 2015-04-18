import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy.constants as const
from uncertainties import ufloat
import uncertainties.unumpy as unp
from scipy.stats import sem
import sympy


N=np.genfromtxt('Kompressorleistung_N_el.txt').T
N=np.mean(N)

N_t=1/4*(175-N)**2+(205-N)**2+(210-N)**2+(210-N)**2
delta_N_t=np.sqrt(1/3*((175+420+205)/4))



delta_nu_real_1=np.sqrt(((13209*0.002)/N_t)**2+((13209*0.0223*delta_N_t)/(N_t**2))**2)
delta_nu_real_2=np.sqrt(((13209*0.004)/N_t)**2+((13209*0.032*delta_N_t)/(N_t**2))**2)
delta_nu_real_3=np.sqrt(((13209*0.007)/N_t)**2+((13209*0.027*delta_N_t)/(N_t**2))**2)
delta_nu_real_4=np.sqrt(((13209*0.008)/N_t)**2+((13209*0.023*delta_N_t)/(N_t**2))**2)

nu_real_1=13209*0.0223/(N_t)
nu_real_2=13209*0.032/(N_t)
nu_real_3=13209*0.027/(N_t)
nu_real_4=13209*0.023/(N_t)

T11=296.15
T21=294.35
T12=306.85	
T22=284.65
T13=317.35	
T23=274.95
T14=320.35	
T24=273.35

nu_ideal1=T11/(T11-T21)
nu_ideal2=T12/(T12-T22)
nu_ideal3=T13/(T13-T23)
nu_ideal4=T14/(T14-T24)


print(N)
print('N für die Berechnung zur den Zeiten t_i',N_t, '+-',delta_N_t)
print('')
print('nu real')
print(nu_real_1,'+-',delta_nu_real_1)
print(nu_real_2,'+-',delta_nu_real_2)
print(nu_real_3,'+-',delta_nu_real_3)
print(nu_real_4,'+-',delta_nu_real_4)
print('')
print('nu ideal')
print('nu_ideal1',nu_ideal1)
print('nu_ideal2',nu_ideal2)
print('nu_ideal3',nu_ideal3)
print('nu_ideal4',nu_ideal4)


