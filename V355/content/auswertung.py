import numpy as np
import math
import uncertainties.unumpy as unp
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as stds
from uncertainties import ufloat

# Daten
C=ufloat(798,2)*10**-12

Csp=ufloat(37,1)*10**-12

L=ufloat(31.90,0.05)*10**-3

R= ufloat(50,0)

# Resonanzfrequenz
fres=unp.sqrt(1/(L*C)-R**2/(4*L**2))

# Ausgabe
print("""
Die Resonanzfrequenz ist {} Hertz

""".format(fres,))