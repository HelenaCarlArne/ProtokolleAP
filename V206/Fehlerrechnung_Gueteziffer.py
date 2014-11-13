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

N_MW=np.mean(N)
N_Fehler=sem(N)

N_delta=1/4*(175+205+210*2)

delta_nu_real_1=np.sqrt(((13209*0.002)/N_delta)**2+((13209*0.0223*N_Fehler)/(N_delta**2))**2)
delta_nu_real_2=np.sqrt(((13209*0.005)/N_delta)**2+((13209*0.031*N_Fehler)/(N_delta**2))**2)
delta_nu_real_3=np.sqrt(((13209*0.009)/N_delta)**2+((13209*0.027*N_Fehler)/(N_delta**2))**2)
delta_nu_real_4=np.sqrt(((13209*0.013)/N_delta)**2+((13209*0.017*N_Fehler)/(N_delta**2))**2)

nu_real_1=13209*0.0223/(N_delta)
nu_real_2=13209*0.031/(N_delta)
nu_real_3=13209*0.027/(N_delta)
nu_real_4=13209*0.017/(N_delta)

T11=23.0	
T21=21.2
T12=33.7	
T22=11.5
T13=44.2	
T23=1.8
T14=49.8	
T24=-0.7

nu_ideal1=T11/(T11-T21)
nu_ideal2=T12/(T12-T22)
nu_ideal3=T13/(T13-T23)
nu_ideal4=T14/(T14-T24)

print('Mittelwertleistung',N_MW,'+/-',N_Fehler)
print('N für die Berechnung zur den Zeiten t_i',N_delta)
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
print('nu ideal')


