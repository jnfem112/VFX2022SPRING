import math
import numpy as np
import cv2
from time import time

class Debevec:
	def __init__(self , number_of_sample_vertical = 10 , number_of_sample_horizontal = 10 , lambd = 10):
		self.number_of_sample_vertical = number_of_sample_vertical
		self.number_of_sample_horizontal = number_of_sample_horizontal
		self.number_of_sample = number_of_sample_vertical * number_of_sample_horizontal
		self.lambd = lambd
		self.weight = 127 - np.floor(np.abs(np.arange(256) - 127.5))

	def solve_response_curve(self , images , exposure_times):
		number_of_image = len(images)
		images = [cv2.resize(image , (self.number_of_sample_horizontal , self.number_of_sample_vertical) , cv2.INTER_NEAREST) for image in images]

		self.response_curves = []

		for i in range(3):
			A = np.zeros((number_of_image * self.number_of_sample + 255 , self.number_of_sample + 256))
			b = np.zeros((number_of_image * self.number_of_sample + 255))

			for j in range(number_of_image):
				for k in range(self.number_of_sample):
					value = images[j][k // self.number_of_sample_horizontal][k % self.number_of_sample_horizontal][i]
					A[j * self.number_of_sample + k][value] = self.weight[value] * 1
					A[j * self.number_of_sample + k][k + 256] = self.weight[value] * -1
					b[j * self.number_of_sample + k] = self.weight[value] * np.log(exposure_times[j])

			for j in range(254):
				A[number_of_image * self.number_of_sample + j][j] = self.lambd * self.weight[j + 1] * 1
				A[number_of_image * self.number_of_sample + j][j + 1] = self.lambd * self.weight[j + 1] * -2
				A[number_of_image * self.number_of_sample + j][j + 2] = self.lambd * self.weight[j + 1] * 1

			A[-1][127] = 1

			x = np.linalg.lstsq(A , b , rcond = None)[0]
			self.response_curves.append(x[ : 256])

	def reconstruct_HDR_image(self , images , exposure_times):
		HDR_image = np.zeros(images[0].shape)
		for i in range(3):
			HDR_image[ : , : , i] = np.exp(np.sum([np.array(list(map(lambda value : self.weight[value] * (self.response_curves[i][value] - np.log(exposure_time)) , image[ : , : , i]))) for image , exposure_time in zip(images , exposure_times)] , axis = 0) / \
										  (np.sum([np.array(list(map(lambda value : self.weight[value] , image[ : , : , i]))) for image in images] , axis = 0) + 1e-8))
		return HDR_image.astype(np.float32)

class Robertson:
	def __init__(self , iteration = 5):
		self.iteration = iteration
		self.weight = (np.exp(4) / (np.exp(4) - 1)) * np.exp(-(4 * np.arange(256) / 255 - 2)**2) + (1 / (1 - np.exp(4)))

	def solve_response_curve(self , images , exposure_times):
		number_of_image = len(images)
		height , width , _ = images[0].shape
		self.response_curves = []
		for i , channel in enumerate(['blue' , 'green' , 'red']):
			print(f'        channel: {channel}')
			response_curve = np.arange(256) / 128
			for j in range(self.iteration):
				print(f'            iteration: {j + 1} ' , end = '')
				start = time()

				radiance_map = np.sum([self.weight[image[ : , : , i]] * response_curve[image[ : , : , i]] * exposure_time for image , exposure_time in zip(images , exposure_times)] , axis = 0) / \
							   np.sum([self.weight[image[ : , : , i]] * exposure_time**2 for image , exposure_time in zip(images , exposure_times)] , axis = 0)
				
				response_curve = np.zeros((256))
				for k in range(256):
					cardinality = 0
					for image , exposure_time in zip(images , exposure_times):
						response_curve[k] += np.sum(radiance_map[image[ : , : , i] == k] * exposure_time)
						cardinality += np.sum(image[ : , : , i] == k)
					response_curve[k] /= (cardinality + 1e-8)
				response_curve /= response_curve[128]

				error = np.sum([self.weight[image[ : , : , i]] * (response_curve[image[ : , : , i]] - radiance_map * exposure_time)**2 for image , exposure_time in zip(images , exposure_times)])
				end = time()
				print(f'({int(end - start)}s) error: {error}')

			self.response_curves.append(response_curve)

	def reconstruct_HDR_image(self , images , exposure_times):
		HDR_image = np.zeros(images[0].shape)
		for i in range(3):
			HDR_image[ : , : , i] = np.sum([self.weight[image[ : , : , i]] * self.response_curves[i][image[ : , : , i]] * exposure_time for image , exposure_time in zip(images , exposure_times)] , axis = 0) / \
									np.sum([self.weight[image[ : , : , i]] * exposure_time**2 for image , exposure_time in zip(images , exposure_times)] , axis = 0)
		return HDR_image.astype(np.float32)