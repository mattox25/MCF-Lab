import numpy as np
from main import Temperatures, T_errors
import matplotlib.pyplot as plt

radius = np.loadtxt('radius.dat')
mask = np.where((Temperatures > 0) & (Temperatures < 500))
Temperatures = Temperatures[mask]
radius = radius[mask]
T_errors = T_errors[mask]

plt.errorbar(radius, Temperatures, xerr=None, yerr=T_errors, fmt='o', ecolor='r', markersize=2, capsize=2)
plt.title('Temperature Profile')
plt.ylabel('Electron Temperature (eV)')
plt.xlabel('Radius (m)')
plt.savefig('temp_profile_plt_with_error')
plt.close()

plt.scatter(radius, Temperatures, s=2)
plt.title('Temperature Profile')
plt.ylabel('Electron Temperature (eV)')
plt.xlabel('Radius (m)')
plt.savefig('temp_profile_plt')
plt.close()
