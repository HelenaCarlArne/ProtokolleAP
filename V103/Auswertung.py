import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 16
import numpy as np

a, b ,c = np.genfromtxt('Zweiseitig.txt').T

print(a)
print(b)
print(c)