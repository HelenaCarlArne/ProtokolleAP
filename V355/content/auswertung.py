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

uCkopplung=unp.uarray([9.99,8.00,6.47,5.02,4.00,3.00,2.03,1.01],[9.99*10**-2,8.00*10**-2,6.47*10**-2,5.02*10**-2,4.00*10**-2,3.00*10**-2,2.03*10**-2,1.01*10**-2])
Ckopplung=uCkopplung*10**(-9)

Cgesamt=(C*Ckopplung)/(2*C+Ckopplung)+Csp

vau1m=np.array([30.56,30.57,30.57,30.57,30.57,30.57,30.58,30.58])*1000
vau2m=np.array([32.85,33.38,33.98,34.88,35.86,37.40,40.12,47.23])*1000

alpham=np.array([14,11,9,7,6,5,4,2])

# Resonanzfrequenz
fres=unp.sqrt(1/(L*(C+Csp))-R**2/(4*L**2))/(2*np.pi)

# Fundamentalfrequenz
vau1 =1/(2*np.pi*unp.sqrt(L*(C+Csp)))
vau2 = 1/(2*np.pi*unp.sqrt(L*Cgesamt))

#Abweichung der Fundamentalfrequenzen
vau1dev=(vau1-vau1m)*100/vau1
vau2dev=(vau2-vau2m)*100/vau2

#Frequenzverhältnisse
alpha=-(vau1+vau2)/(2*(vau1-vau2))
alphav=(alpha-alpham)*100/alpham

# Ausgabe
print("""
#################################################################################################################################################
Die Resonanzfrequenz ist {} Hertz

#################################################################################################################################################
Die erste Fundamentalfrequenz ist Hertz:
{} 
Die zweite Fundamentalfrequenz ist in Hertz:
{}
#################################################################
Die Abweichungen der Fundamentalfrequenzen 1 von der Theorie in %
{}
Die Abweichungen der Fundamentalfrequenzen 2 von der Theorie in %
{}
#################################################################################################################################################
Die Frequenzverhaeltnisse sind:
{}
Deren Verhaeltnisse sind:
{}
""".format(fres,vau1,vau2,vau1dev,vau2dev,alpha,alphav))