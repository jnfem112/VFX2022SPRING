import math
import numpy as np
import cv2

class MTB:
	def __init__(self , max_offset = 0 , tolerance = 10):
		self.max_offset = max_offset
		self.max_depth = int(math.log2(max_offset + 1))
		self.tolerance = tolerance

	def get_bitmap_and_mask(self , image):
		threshold = np.median(image)
		bitmap = image <= threshold
		mask = np.logical_or(image < threshold - self.tolerance , image > threshold + self.tolerance)
		return bitmap , mask

	def shift(self , image , offset_x , offset_y):
		translation_matrix = np.array([[1.0 , 0.0 , offset_x] , [0.0 , 1.0 , offset_y]])
		return cv2.warpAffine(image.astype(np.uint8) , translation_matrix , (image.shape[1] , image.shape[0]))

	def count_difference(self , bitmap_1 , bitmap_2 , mask_1 , mask_2):
		return np.sum(np.logical_and(np.logical_and(np.logical_xor(bitmap_1 , bitmap_2) , mask_1) , mask_2))

	def find_best_offset(self , source , target , previous_offset_x = 0 , previous_offset_y = 0):
		bitmap_source , mask_source = self.get_bitmap_and_mask(source)
		bitmap_target , mask_target = self.get_bitmap_and_mask(target)
		min_difference , best_offset_x , best_offset_y = np.inf , None , None
		for offset_x in [-1 , 0 , 1]:
			for offset_y in [-1 , 0 , 1]:
				shifted_bitmap_source = self.shift(bitmap_source , previous_offset_x + offset_x , previous_offset_y + offset_y)
				shifted_mask_source = self.shift(mask_source , previous_offset_x + offset_x , previous_offset_y + offset_y)
				difference = self.count_difference(shifted_bitmap_source , bitmap_target , shifted_mask_source , mask_target)
				if difference < min_difference:
					min_difference = difference
					best_offset_x = previous_offset_x + offset_x
					best_offset_y = previous_offset_y + offset_y
		return best_offset_x , best_offset_y

	def align_image(self , source , target , depth = None):
		if depth is None:
			depth = self.max_depth

		if depth == 0:
			offset_x , offset_y = 0 , 0
		else:
			previous_offset_x, previous_offset_y = self.align_image(cv2.pyrDown(source) , cv2.pyrDown(target) , depth - 1)
			offset_x , offset_y = self.find_best_offset(source , target , 2 * previous_offset_x , 2 * previous_offset_y)

		return offset_x , offset_y
	
	def align_images(self , images):
		target = cv2.cvtColor(images[0] , cv2.COLOR_BGR2GRAY)
		result , offset = [images[0]] , [(0 , 0)]
		for i in range(1 , len(images)):
			source = cv2.cvtColor(images[i] , cv2.COLOR_BGR2GRAY)
			offset_x , offset_y = self.align_image(source , target)
			result.append(self.shift(images[i] , offset_x , offset_y))
			offset.append((offset_x , offset_y))
		return result , offset