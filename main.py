import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Function to read 2D data from a text file and store it in a numpy array
def twoD_data_to_array(file_path):
    try:
        # Read the data from the file into a 2D numpy array
        data_array = np.loadtxt(file_path)
        return data_array
    except Exception as e:
        print(f"Error reading the file: {e}")
        return None

# Function to slice for a constant y-value
def slice(data, val):
    return data[val, :]

def gaussian(x, *params):

	A = params[0]
	x0 = params[1]
	c = params[2]
	y0 = params[3]
		
	return y0 + A*np.exp(-(x-x0)**2/(2*c*c))


# Example usage
intens_dat = twoD_data_to_array('intensity.dat')
lambda_dat = twoD_data_to_array('lambda.dat')

# Gather cross section data
intensity_cs = slice(intens_dat, 150) # From intensity data
lambda_cs = slice(lambda_dat, 150) # From lambda data

# Mask data
# IMPORTANT: Mask must be changed manually
mask_arr = np.array([[6400, 6700], [6800, 7100]])

# mask the data to pass into the fitting function
mask1 = (lambda_cs >= mask_arr[0][0]) & (lambda_cs <= mask_arr[0][1])
lambda_dat_half_mask = lambda_cs[~mask1]
intens_dat_half_mask = intensity_cs[~mask1]

mask2 = (lambda_dat_half_mask >=mask_arr[1][0]) & (lambda_dat_half_mask <= mask_arr[1][1])
lambda_dat_mask = lambda_dat_half_mask[~mask2]
intens_dat_mask = intens_dat_half_mask[~mask2]

# Fit the data
guess = np.array([1700, 6900, 270, 16])
params, params_covariance = curve_fit(gaussian, lambda_dat_mask, intens_dat_mask, p0=guess)
x_range = np.arange(min(lambda_dat_mask), max(lambda_dat_mask), 10)
print(params)