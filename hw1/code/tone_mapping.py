import math
import numpy as np
import cv2

def Reinhard_global(HDR_image , a = 0.18 , L_white = None):
	L_w = 0.06 * HDR_image[ : , : , 0] + 0.67 * HDR_image[ : , : , 1] + 0.27 * HDR_image[ : , : , 2]

	L_w_average = np.exp(np.mean(np.log(L_w + 1e-8)))
	L_m = a * L_w / L_w_average
	if L_white is None:
		L_white = np.max(L_m)
	L_d = L_m * (1 + L_m / L_white**2) / (1 + L_m)
	
	LDR_image = np.zeros(HDR_image.shape)
	for i in range(3):
		LDR_image[ : , : , i] = HDR_image[ : , : , i] / L_w * L_d

	LDR_image = np.clip(255 * LDR_image , 0 , 255).astype(np.uint8)
	return LDR_image

def Reinhard_local(HDR_image , a = 0.72 , phi = 8 , eps = 0.05):
	L_w = 0.06 * HDR_image[ : , : , 0] + 0.67 * HDR_image[ : , : , 1] + 0.27 * HDR_image[ : , : , 2]

	L_w_average = np.exp(np.mean(np.log(L_w + 1e-8)))
	L_m = a * L_w / L_w_average

	previous_L_blur = L_m
	L_d = np.zeros(L_m.shape)
	finish = np.full(L_m.shape , False)
	for scale in range(1 , 9):
		s_1 = 2 * math.ceil(1.6**(scale - 1) / math.sqrt(2)) + 1
		s_2 = 2 * math.ceil(1.6**scale / math.sqrt(2)) + 1
		L_blur_1 = cv2.GaussianBlur(L_m , (s_1 , s_1) , 0)
		L_blur_2 = cv2.GaussianBlur(L_m , (s_2 , s_2) , 0)
		V = (L_blur_1 - L_blur_2) / (2**phi * a / s_1**2 + L_blur_1)
		
		index = np.logical_and(V >= eps , np.logical_not(finish))
		L_d[index] = (L_m / (1 + previous_L_blur))[index]
		finish[index] = True
		
		previous_L_blur = L_blur_1
		
	index = np.logical_not(finish)
	L_d[index] = (L_m / (1 + previous_L_blur))[index]
	
	LDR_image = np.zeros(HDR_image.shape)
	for i in range(3):
		LDR_image[ : , : , i] = HDR_image[ : , : , i] / L_w * L_d

	LDR_image = np.clip(255 * LDR_image , 0 , 255).astype(np.uint8)
	return LDR_image