import numpy
import cv2

def linear_scale (array, low_out, high_out):
	
	low_in = numpy.amin(array)
	high_in = numpy.amax(array)
	
	array = array + (-1)*low_in
	array = numpy.multiply(array, (high_out - low_out)/(high_in - low_in))
	array = array + low_out

	return (array)

# This functions makes image in visible range go all white
# def gamma_correction(array):

# 	max_val = numpy.amax(array)

# 	array = numpy.power(array, 1/2.4)
# 	array = numpy.multiply(array, max_val/numpy.power(max_val, 1/2.2))

# 	return array


def gamma_function (u):

	if (u <= 0.0031308):
		return (12.92*u)

	else:
		return ((1.055 * numpy.power(u, 1/2.4)) - 1.055)

def gamma_correction (array):

	max_val = numpy.amax(array)
	array = numpy.multiply(array, 1/max_val)

	vector_gamma = numpy.vectorize(gamma_function)
	array = vector_gamma(array)

	array = numpy.multiply(array, max_val)

	return (array)


def get_luminance (array):

	L = numpy.zeros((array.shape[0], array.shape[1]))

	max_L = numpy.amin(array)
	min_L = numpy.amax(array)

	for i in range (0, array.shape[0]):

		for j in range (0, array.shape[1]):

			# array is in BGR format instead of RGB
			L[i][j] = 0.114*array[i][j][0] + 0.587*array[i][j][1] + 0.299*array[i][j][2]

			if(L[i][j] != 0):
				
				if(L[i][j] > max_L):
					max_L = L[i][j]

				if(L[i][j] < min_L):
					min_L = L[i][j]

	for i in range (0, array.shape[0]):

		for j in range (0, array.shape[1]):

			if (L[i][j] == 0):

				L[i][j] = min_L

	return (L)


def log_luminance (array):

	L = get_luminance(array)
	logL = numpy.log10(L)

	min_L = numpy.amin(logL)
	max_L = numpy.amax(logL)

	mean = (min_L + max_L)/2

	# we need 1:100 pixel range
	logLnew = linear_scale(logL, mean-1 , mean+1)

	Lnew = numpy.power(10, logLnew)

	new_array = numpy.zeros((array.shape[0], array.shape[1], array.shape[2]))

	for i in range (0, array.shape[0]):

		for j in range (0, array.shape[1]):

			# array is in BGR format instead of RGB
			
			new_array[i][j][0] = (array[i][j][0]*Lnew[i][j])/L[i][j]
			new_array[i][j][1] = (array[i][j][1]*Lnew[i][j])/L[i][j]
			new_array[i][j][2] = (array[i][j][2]*Lnew[i][j])/L[i][j]

	return (new_array)


def Histogram_equalization (array):

	L = get_luminance(array)

	L = numpy.around(L, decimals=3)

	min_L = numpy.amin(L)
	max_L = numpy.amax(L)

	bin_number = int((max_L - min_L)*1000)

	hist, bin_edges = numpy.histogram(L, bins = bin_number, density = False)
	cummulative = numpy.cumsum(hist)

	cummulative = numpy.hstack((0, cummulative))

	total = float(L.shape[0]*L.shape[1])

	Lnew = numpy.zeros((L.shape[0], L.shape[1]))

	for i in range (0, L.shape[0]):

		for j in range (0, L.shape[1]):

			Lnew[i][j] = (cummulative[ int((L[i][j] - min_L)*1000) ]/(total))*(max_L-min_L) + min_L


	new_array = numpy.zeros((array.shape[0], array.shape[1], array.shape[2]))

	for i in range (0, array.shape[0]):

		for j in range (0, array.shape[1]):

			# array is in BGR format instead of RGB
			
			new_array[i][j][0] = (array[i][j][0]*Lnew[i][j])/L[i][j]
			new_array[i][j][1] = (array[i][j][1]*Lnew[i][j])/L[i][j]
			new_array[i][j][2] = (array[i][j][2]*Lnew[i][j])/L[i][j]

	return (new_array)


def Histogram_equalization_log (array):

	L = get_luminance(array)
	
	logL = numpy.log10(L)

	logL = numpy.around(logL, decimals=3)

	min_L = numpy.amin(logL)
	max_L = numpy.amax(logL)

	bin_number = int((max_L - min_L)*1000)

	hist, bin_edges = numpy.histogram(logL, bins = bin_number, density = False)
	cummulative = numpy.cumsum(hist)

	cummulative = numpy.hstack((0, cummulative))

	total = float(logL.shape[0]*logL.shape[1])

	logLnew = numpy.zeros((logL.shape[0], logL.shape[1]))

	for i in range (0, logL.shape[0]):

		for j in range (0, logL.shape[1]):

			logLnew[i][j] = (cummulative[ int((logL[i][j] - min_L)*1000) ]/(total))*(max_L-min_L) + min_L

	Lnew = numpy.power(10, logLnew)

	new_array = numpy.zeros((array.shape[0], array.shape[1], array.shape[2]))

	for i in range (0, array.shape[0]):

		for j in range (0, array.shape[1]):

			# array is in BGR format instead of RGB
			
			new_array[i][j][0] = (array[i][j][0]*Lnew[i][j])/L[i][j]
			new_array[i][j][1] = (array[i][j][1]*Lnew[i][j])/L[i][j]
			new_array[i][j][2] = (array[i][j][2]*Lnew[i][j])/L[i][j]

	return (new_array)

def add_filter (fil, array):

	fil_array = numpy.zeros((array.shape[0], array.shape[1]))

	for i in range (1, array.shape[0]-1):

		for j in range (1, array.shape[1]-1):

			fil_array[i][j] = fil[0][0]*array[i-1][j-1] + fil[0][1]*array[i-1][j] + fil[0][2]*array[i-1][j+1] 
			fil_array[i][j] = fil[1][0]*array[i][j-1] + fil[1][1]*array[i][j] + fil[1][2]*array[i][j+1] + fil_array[i][j]
			fil_array[i][j] = fil[2][0]*array[i+1][j-1] + fil[2][1]*array[i+1][j] + fil[2][2]*array[i+1][j+1] + fil_array[i][j]

	return (fil_array)


def sharpning (array):

	L = get_luminance (array)

	# logL = numpy.log10(L)
	logL = L

	fil = numpy.array([[0.0,-1.0,0.0], [-1.0,4.0,-1.0], [0.0,-1.0,0.0]])
	logLnew = add_filter(fil, logL)

	Lnew = logLnew
	# Lnew = numpy.power(10, logLnew)

	new_array = numpy.zeros((array.shape[0], array.shape[1], array.shape[2]))

	for i in range (0, array.shape[0]):

		for j in range (0, array.shape[1]):

			# array is in BGR format instead of RGB
			
			new_array[i][j][0] = (array[i][j][0]*Lnew[i][j])/L[i][j] + array[i][j][0] 
			new_array[i][j][1] = (array[i][j][1]*Lnew[i][j])/L[i][j] + array[i][j][1]
			new_array[i][j][2] = (array[i][j][2]*Lnew[i][j])/L[i][j] + array[i][j][2]

	return (new_array)


def mean (array):

	L = get_luminance (array)

	# logL = numpy.log10(L)
	logL = L

	fil = numpy.array([[1/16, 2/16, 1/16], [2/16, 4/16, 2/16], [1/16, 2/16, 1/16]])
	logLnew = add_filter(fil, logL)

	Lnew = logLnew
	# Lnew = numpy.power(10, logLnew)

	new_array = numpy.zeros((array.shape[0], array.shape[1], array.shape[2]))

	for i in range (0, array.shape[0]):

		for j in range (0, array.shape[1]):

			# array is in BGR format instead of RGB
			
			new_array[i][j][0] = (array[i][j][0]*Lnew[i][j])/L[i][j] 
			new_array[i][j][1] = (array[i][j][1]*Lnew[i][j])/L[i][j] 
			new_array[i][j][2] = (array[i][j][2]*Lnew[i][j])/L[i][j] 

	return (new_array)


def median_filter (L):

	newL = numpy.zeros((L.shape[0], L.shape[1]))

	for i in range (1, L.shape[0]-1):

		for j in range (1, L.shape[1]-1):

			median = numpy.array( [L[i-1][j-1], L[i-1][j], L[i-1][j+1], L[i][j-1], L[i][j], L[i][j+1], L[i+1][j-1], L[i+1][j], L[i+1][j+1]] )

			newL[i][j] = numpy.median(median)

	return (newL)

def median (array):

	L = get_luminance (array)

	# logL = numpy.log10(L)
	logL = L

	logLnew = median_filter (logL)

	Lnew = logLnew
	# Lnew = numpy.power(10, logLnew)

	new_array = numpy.zeros((array.shape[0], array.shape[1], array.shape[2]))

	for i in range (0, array.shape[0]):

		for j in range (0, array.shape[1]):

			# array is in BGR format instead of RGB
			
			new_array[i][j][0] = (array[i][j][0]*Lnew[i][j])/L[i][j] 
			new_array[i][j][1] = (array[i][j][1]*Lnew[i][j])/L[i][j] 
			new_array[i][j][2] = (array[i][j][2]*Lnew[i][j])/L[i][j]

	return (new_array)


######################################################################################################

# Imread reads in BGR format
# -1 reads the image in unmodified format

img = cv2.imread('memorial.hdr', -1)
img_arr = numpy.array(img)

# cv2.imshow('input',img_arr)
# cv2.waitKey(0)

# Linear rescaling with and without gamma

low = linear_scale(img_arr, 0, 50)
cv2.imshow('LR - low', low)
cv2.waitKey(0)

low_gamma = gamma_correction(low)
cv2.imshow('LR - low with gamma', low_gamma)
cv2.waitKey(0)

medium = linear_scale(img_arr, 0, 255)
cv2.imshow('LR - visible', medium)
cv2.waitKey(0)

medium_gamma = gamma_correction(medium)
cv2.imshow('LR - visible with gamma', medium_gamma)
cv2.waitKey(0)

high = linear_scale(img_arr, 0, 1000)
cv2.imshow('LR - high', high)
cv2.waitKey(0)

high_gamma = gamma_correction(high)
cv2.imshow('LR - high with gamma', high_gamma)
cv2.waitKey(0)

# In order to save images - jpg, jpeg, png formats will fail because majority values of array are float
# We need extensions like .hdr and .exr
# Example - cv2.imwrite('Low.exr', low)


# Log luminance

L_image = log_luminance(img_arr)
cv2.imshow('Log luminance', L_image)
cv2.waitKey(0)

# Gamma correction on log luminance rescaled image is useless

# L_image_gamma = gamma_correction(L_image)
# cv2.imshow('Log luminanace with gamma', L_image_gamma)
# cv2.waitKey(0)


# Enhancing the logorithmicly rescaled image

# Histogram equilization

# Luminance domain

hist_img = Histogram_equalization (L_image)
cv2.imshow('Histogram equilization', hist_img)
cv2.waitKey(0)

# Log luminance domain

hist_img_log = Histogram_equalization_log (L_image)
cv2.imshow('Histogram equilization log', hist_img_log)
cv2.waitKey(0)


# Sharpning Filter

sharp = sharpning(L_image)
cv2.imshow('Unsharp Masking', sharp)
cv2.waitKey(0)

# Mean Filter

mean = mean(sharp)
cv2.imshow('Mean Filter', mean)
cv2.waitKey(0)

# Median Filter

median = median(sharp)
cv2.imshow('Median Filter', median)
cv2.waitKey(0)
