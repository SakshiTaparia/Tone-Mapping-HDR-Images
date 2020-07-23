import numpy
import cv2

def get_luminance (array):

	L = numpy.zeros((array.shape[0], array.shape[1]))

	max_L = numpy.amin(array)
	min_L = numpy.amax(array)

	for i in range (0, array.shape[0]):

		for j in range (0, array.shape[1]):

			# array is in BGR format instead of RGB

			L[i][j] = 0.06*array[i][j][0] + 0.67*array[i][j][1] + 0.27*array[i][j][2]

			if(L[i][j] != 0):
				
				if(L[i][j] > max_L):
					max_L = L[i][j]

				if(L[i][j] < min_L):
					min_L = L[i][j]

	for i in range (0, array.shape[0]):

		for j in range (0, array.shape[1]):

			if (L[i][j] == 0):

				L[i][j] = min_L/10

	return (L)


def log_average (L):

	logL = numpy.log10(L)

	array_sum = numpy.sum(logL)
	
	return ( numpy.power(10, array_sum/(L.shape[0]*L.shape[1] )) )


def L_d_function (element, minimum):

	return ( (element*(1 + (element/(minimum*minimum))))/(1 + element) )


def undo_luminance (array, L, Lnew):

	new_array = numpy.zeros((array.shape[0], array.shape[1], array.shape[2]))

	for i in range (0, array.shape[0]):

		for j in range (0, array.shape[1]):

			# array is in BGR format instead of RGB
			
			new_array[i][j][0] = (array[i][j][0]*Lnew[i][j])/L[i][j]
			new_array[i][j][1] = (array[i][j][1]*Lnew[i][j])/L[i][j]
			new_array[i][j][2] = (array[i][j][2]*Lnew[i][j])/L[i][j]

	return (new_array)


def R_1 (x, y, s, alpha_1):

	array = numpy.zeros((x,y))

	for i in range (0,x):

		for j in range (0,y):

			exponent = numpy.exp( (-1)*(numpy.power(i,2) + numpy.power(j,2))/numpy.power( (alpha_1*s) ,2))

			array[i][j] = exponent/(numpy.power( (alpha_1*s) ,2) * numpy.pi )

	return (array)

def R_2 (x, y, s, alpha_2):

	array = numpy.zeros((x,y))

	for i in range (0,x):

		for j in range (0,y):

			exponent = numpy.exp( (-1)*(numpy.power(i,2) + numpy.power(j,2))/numpy.power( (alpha_2*s) ,2))

			array[i][j] = exponent/(numpy.power( (alpha_2*s) ,2) * numpy.pi )

	return (array)


def compute_V (L_array, R_array):

	# Computing convolution using fast fourier transform 

	L = numpy.fft.fft2(L_array)
	R = numpy.fft.fft2(R_array)

	return ( numpy.fft.ifft2(L*R) )


######################################################################################################

img = cv2.imread('memorial.hdr', -1)
img_arr = numpy.array(img)

# cv2.imshow('input',img_arr)
# cv2.waitKey(0)

max_pixel = numpy.amax(img_arr)

# Constants 

key_value = 0.36

alpha_1 = 0.35

alpha_2 = 0.56

epsilon = 0.05

Gamma = 10

s_values = numpy.array( [1, 1.6, 2.56, 4.10, 6.55, 10.49, 16.77, 26.84, 42.95, 68.72, 109.95, 175.92, 281.48, 450.39] )

# Implementation of Photographic Tone Reproduction for Digital Images

luminance = get_luminance(img_arr)

log_avg = log_average(luminance)

scaled_luminance = numpy.multiply(luminance, key_value/log_avg)


# PART ONE - Images with dynamic range in fewer zones

L_white = numpy.max(scaled_luminance)

L_d_function_vector = numpy.vectorize(L_d_function)

L_d_tone_mapping = L_d_function_vector(scaled_luminance, L_white)

new_img_arr = undo_luminance (img_arr, luminance, L_d_tone_mapping)

cv2.imshow('Lower dynamic ranges', new_img_arr)
cv2.waitKey(0)


# PART TWO - Dodging and burning very for very high dynamic ranges

X_max = scaled_luminance.shape[0]
Y_max = scaled_luminance.shape[1]

print(X_max)
print(Y_max)

final_V1 = numpy.zeros((X_max, Y_max))

for s in s_values:

	print(s)

	R1 = R_1 (X_max, Y_max, s, alpha_1)

	V1 = compute_V (scaled_luminance, R1)

	R2 = R_2 (X_max, Y_max, s, alpha_2)

	V2 = compute_V (scaled_luminance, R2)

	const = numpy.power(2,Gamma) * ( key_value/(s*s) )

	V = numpy.divide( V1 - V2, V1 + const) 

	count = 0

	for i in range (0, X_max):

		for j in range (0, Y_max):

			if ( final_V1[i][j] == 0 ):

				if ( numpy.abs(V[i][j]) <= epsilon):

					final_V1[i][j] = V1[i][j]

					count = count + 1

	print(count)


L_d_tone_mapping_new = numpy.divide(scaled_luminance, 1 + final_V1)

new_img_arr_high = undo_luminance (img_arr, luminance, L_d_tone_mapping_new)

cv2.imshow('Very high dynamic ranges', new_img_arr_high)
cv2.waitKey(0)
