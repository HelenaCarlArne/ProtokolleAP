import numpy as np
import matplotlib.pyplot as plt

x=np.array([0.02,0.05,0.08,0.11,0.14,0.17,0.20,0.23,0.26,0.29,0.39,0.49,0.59,0.69,0.79,0.99,1.19])
y=np.array([-5,-1,-0.5, -0.2,-0.1,-0.09, -0.006,-0.0045,-0.0035,-0.0025,-0.001,-0.0007,-0.0005,-0.0003,-0.00025,-0.00012,-0.0001])
plt.ylim(-5,1)
plt.plot(x,y,"bx",label="Messdaten")
plt.legend(loc="best")
plt.show()