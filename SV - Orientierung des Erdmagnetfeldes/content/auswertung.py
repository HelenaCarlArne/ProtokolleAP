import os.path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from scipy.optimize import curve_fit
from matrix2latex import matrix2latex
a, U1, U2, U3, U4 = np.genfromtxt('../Werte/werte.txt').T

a=np.radians(a)
U=(U1+U2+U3+U4)/4
#dU=np.std(U)



plt.plot(a, U)
plt.savefig('../Bilder/cos.pdf')
plt.show()
plt.close()

plt.plot(a, U1,'gx')
plt.plot(a, U2,'kx')
plt.plot(a, U3,'yx')
plt.plot(a, U4,'rx')
plt.savefig('../Bilder/cos1.pdf')
plt.show()
plt.close()


def f(a,u,b):
    return u*np.cos(a)+b

params, covariance = curve_fit(f, a, U)
# covariance ist die Kovarianzmatrix

errors = np.sqrt(np.diag(covariance))

print('u =', params[0], '±', errors[0])
print('b =', params[1], '±', errors[1])

a_plot = np.linspace(0, 10)

plt.plot(a, U, 'rx', label="example data")
plt.plot(a_plot, f(a_plot, *params), 'b-', label='linearer Fit')
plt.legend(loc="best")
plt.savefig('../Bilder/fit.pdf')
plt.show()
plt.close()
