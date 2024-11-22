import numpy as np
import matplotlib.pyplot as plt
from main import *

# Cross section from intensity data
plt.plot(intensity_cs)
plt.title('Position vs Intensity')
plt.xlabel('Position')
plt.ylabel('Intensity')
plt.savefig('Position Intensity plot')
plt.close()

# Cross section from lambda data
plt.plot(lambda_cs, intensity_cs)
plt.title('Wavelength vs Intensity')
plt.ylabel('Intensity')
plt.xlabel('Wavelength (Angstroms)')
plt.savefig('Wavelength Intensity plot')
plt.close()

# Cross section from lambda data WITH FIT
plt.plot(lambda_cs, intensity_cs)
plt.plot(x_range, gaussian(x_range, *params), label='Gaussian Fit', color='r')
plt.legend()
plt.title('Wavelength vs Intensity')
plt.ylabel('Intensity')
plt.xlabel('Wavelength (Angstroms)')
plt.savefig('Wavelength Intensity fit')
plt.close()

plt.scatter(lambda_dat_mask, intens_dat_mask, s=2)
plt.title('Wavelength vs Intensity: Masked Data')
plt.ylabel('Intensity')
plt.xlabel('Wavelength (Angstroms)')
plt.savefig('Wavelength Intensity Masked Scatter')
plt.close()

plt.scatter(lambda_dat_mask, intens_dat_mask, s=2)
plt.plot(x_range, gaussian(x_range, *params), label='Gaussian Fit', color='r')
plt.legend()
plt.title('Wavelength vs Intensity: Masked Data with Fit')
plt.ylabel('Intensity')
plt.xlabel('Wavelength (Angstroms)')
plt.savefig('Wavelength Intensity Masked Scatter with Fit')
plt.close()


# Create the color plots
plt.imshow(intens_dat, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title('Intensity Heat Plot')
plt.xlabel('X Index')
plt.ylabel('Y Index')
plt.savefig('Intensity heatmap.png')
plt.close()    

plt.imshow(lambda_dat, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title('Wavelength Heat Plot')
plt.xlabel('X Index')
plt.ylabel('Y Index')
plt.savefig('Lambda heatmap.png')
plt.close()
