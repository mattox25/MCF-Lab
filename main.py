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
angle_dat = np.loadtxt('angle.dat')

Temperatures = []
T_errors = []
for i in range(len(angle_dat)):
    # Gather cross section data
    intensity_cs = slice(intens_dat, i) # From intensity data
    lambda_cs = slice(lambda_dat, i) # From lambda data
    angle_cs = angle_dat[i]

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
    perr = np.sqrt(np.diag(params_covariance))
    lambda_i = params[1]
    lambda_i_err = perr[1]
    sigma_lambda = params[2]
    sigma_lambda_err = perr[2]
    # constants 
    k_b = 1.38e-23 #J/K
    m_e = 9.1e-31 #kg
    c = 2.99e8 #m/s
    energy = m_e * c**2 #joules
    K = energy/(4*k_b*np.sin(angle_cs/2)**2)
    beta = sigma_lambda / (lambda_i * np.sqrt(2) * np.sin(angle_cs/2))

    temp = (beta**2 * energy/(2 * k_b))/11600
    T_error = 2 * K * np.sqrt(sigma_lambda ** 2/lambda_i ** 4 * sigma_lambda_err ** 2 + sigma_lambda ** 4 / lambda_i ** 6 * lambda_i_err ** 2)
    # print(f'Electron Temp = {(beta**2 * energy/(2 * k_b))/11600} +/- {T_error/11600} eV')
    Temperatures.append(temp)
    T_errors.append(T_error/11600)

Temperatures = np.array(Temperatures)
T_errors = np.array(T_errors)