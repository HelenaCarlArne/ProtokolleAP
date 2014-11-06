#
# Header
#
import numpy as np 									                                                    
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
import uncertainties.unumpy as unp
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as stds
from uncertainties import ufloat
from scipy.stats import sem
import peakdetect
#
# Einladen der Daten
#
ID_M1, T1_M1, T2_M1, T3_M1, T4_M1, T5_M1, T6_M1, T7_M1, T8_M1 = np.genfromtxt("../Originalwerte/MessungStat").T
ID_M2, T1_M2, T5_M2, T2_M2, T6_M2, Junk_M2 = np.genfromtxt("../Originalwerte/Messung80").T 
ID_M3, T1_M3, T2_M3, T3_M3, T4_M3, T5_M3, T6_M3, T7_M3, T8_M3, Junk_M3= np.genfromtxt("../Originalwerte/Messung200",delimiter = ',').T

peakdetect()