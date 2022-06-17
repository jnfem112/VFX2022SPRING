import numpy as np
import cv2

class Keypoint:
	def __init__(self , x , y , orientation , octave , size , response):
		self.x = x
		self.y = y
		self.orientation = orientation
		self.octave = octave
		self.size = size
		self.response = response
	
	def __eq__(self , another):
		return self.x == another.x and self.y == another.y and self.orientation == another.orientation and self.octave == another.octave and self.size == another.size and self.response == another.response
	
	def __lt__(self , another):
		return (self.x < another.x) or \
			(self.x == another.x and self.y < another.y) or \
			(self.x == another.x and self.y == another.y and self.orientation < another.orientation) or \
			(self.x == another.x and self.y == another.y and self.orientation == another.orientation and self.octave < another.octave) or \
			(self.x == another.x and self.y == another.y and self.orientation == another.orientation and self.octave == another.octave and self.size < another.size) or \
			(self.x == another.x and self.y == another.y and self.orientation == another.orientation and self.octave == another.octave and self.size == another.size and self.response < another.response)

class SIFT:
	def __init__(self , sigma = 1.6 , number_of_interval = 3 , contrast_threshold = 0.04 , eigenvalue_ratio = 10):
		self.sigma = sigma
		self.number_of_interval = number_of_interval
		self.contrast_threshold = contrast_threshold
		self.edge_threshold = (eigenvalue_ratio + 1)**2 / eigenvalue_ratio
		self.threshold = int(127.5 * contrast_threshold / number_of_interval)

	def generate_base_image(self , image):
		image = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY).astype(np.float32)
		image = cv2.resize(image , (0 , 0) , fx = 2 , fy = 2 , interpolation = cv2.INTER_LINEAR)
		kernel_size = np.sqrt(self.sigma**2 - 1)
		image = cv2.GaussianBlur(image , (0 , 0) , sigmaX = kernel_size , sigmaY = kernel_size)
		self.base_image = image

	def construct_pyramid(self):
		number_of_octave = int(np.round(np.log2(min(self.base_image.shape)) - 1))
		Gaussian_pyramid , DoG_pyramid = [] , []
		previous_image = self.base_image
		for i in range(number_of_octave):
			Gaussian_images , DoG_images = [previous_image] , []
			for j in range(self.number_of_interval + 2):
				kernel_size = np.sqrt((self.sigma * 2**((j + 1) / self.number_of_interval))**2 - (self.sigma * 2**(j / self.number_of_interval))**2)
				image = cv2.GaussianBlur(previous_image , (0 , 0) , sigmaX = kernel_size , sigmaY = kernel_size)
				Gaussian_images.append(image)
				DoG_images.append(image - previous_image)
				previous_image = image
			Gaussian_pyramid.append(np.array(Gaussian_images))
			DoG_pyramid.append(np.array(DoG_images))
			previous_image = Gaussian_images[-3]
			previous_image = cv2.resize(previous_image , (previous_image.shape[1] // 2 , previous_image.shape[0] // 2) , interpolation = cv2.INTER_NEAREST)
		self.number_of_octave = number_of_octave
		self.Gaussian_pyramid = Gaussian_pyramid
		self.DoG_pyramid = DoG_pyramid

	def keypoint_localization(self , border = 5):
		keypoints = []
		for octave_index in range(self.number_of_octave):
			DoG_images = self.DoG_pyramid[octave_index]
			for image_index in range(1 , self.number_of_interval + 1):
				DoG_image = DoG_images[image_index]
				height , width = DoG_image.shape
				for y in range(border , height - border):
					for x in range(border , width - border):
						if abs(DoG_image[y][x]) > self.threshold and \
						((DoG_image[y][x] >= 0 and np.all(DoG_image[y][x] >= DoG_images[image_index - 1 : image_index + 2 , y - 1 : y + 2 , x - 1 : x + 2])) or \
							(DoG_image[y][x] < 0 and np.all(DoG_image[y][x] <= DoG_images[image_index - 1 : image_index + 2 , y - 1 : y + 2 , x - 1 : x + 2]))):
							ret = self.accurate_keypoint_localization(DoG_images , octave_index , image_index , x , y , border)
							if ret is not None:
								keypoint , accurate_image_index = ret
								keypoints += self.find_orientation(keypoint , self.Gaussian_pyramid[octave_index][accurate_image_index] , octave_index)

		keypoints.sort()
		unique_keypoints = [keypoints[0]]
		for i in range(1 , len(keypoints)):
			if keypoints[i] != unique_keypoints[-1]:
				unique_keypoints.append(keypoints[i])

		for i in range(len(unique_keypoints)):
			unique_keypoints[i].x /= 2
			unique_keypoints[i].y /= 2
			unique_keypoints[i].size /= 2
			unique_keypoints[i].octave = (unique_keypoints[i].octave & ~255) | ((unique_keypoints[i].octave - 1) & 255)

		return unique_keypoints

	def accurate_keypoint_localization(self , DoG_images , octave_index , image_index , x , y , border = 5 , iteration = 5):
		for _ in range(iteration):
			gradient = self.compute_gradient(DoG_images / 255 , image_index , x , y)
			Hessian = self.compute_Hessian(DoG_images / 255 , image_index , x , y)
			offset = -np.linalg.lstsq(Hessian , gradient , rcond = None)[0]
			if np.all(np.abs(offset) < 0.5):
				break
			image_index += int(np.round(offset[0]))
			x += int(np.round(offset[2]))
			y += int(np.round(offset[1]))
			if (image_index < 1 or image_index > self.number_of_interval) or (x < border or x >= DoG_images.shape[2] - border) or (y < border or y >= DoG_images.shape[1] - border):
				return None
		else:
			return None

		contrast_response = DoG_images[image_index][y][x] / 255 + 0.5 * np.dot(gradient , offset)
		edge_response = np.trace(Hessian[1 : , 1 : ])**2 / (np.linalg.det(Hessian[1 : , 1 : ]) + 1e-7)
		if abs(contrast_response) * self.number_of_interval >= self.contrast_threshold and (np.linalg.det(Hessian[1 : , 1 : ]) > 0 and edge_response < self.edge_threshold):
			keypoint = Keypoint(
				(x + offset[2]) * 2**octave_index , 
				(y + offset[1]) * 2**octave_index , 
				None , 
				octave_index + 256 * image_index + 65536 * int(np.round(255 * (offset[0] + 0.5))) , 
				self.sigma * 2**((image_index + offset[0]) / self.number_of_interval) * 2**(octave_index + 1) , 
				abs(contrast_response)
			)
			return keypoint , image_index
		else:
			return None

	def compute_gradient(self , DoG_images , image_index , x , y):
		dx = (DoG_images[image_index + 1][y][x] - DoG_images[image_index - 1][y][x]) / 2
		dy = (DoG_images[image_index][y + 1][x] - DoG_images[image_index][y - 1][x]) / 2
		dz = (DoG_images[image_index][y][x + 1] - DoG_images[image_index][y][x - 1]) / 2
		return np.array([dx , dy , dz])
	
	def compute_Hessian(self , DoG_images , image_index , x , y):
		dxx = DoG_images[image_index + 1][y][x] + DoG_images[image_index - 1][y][x] - 2 * DoG_images[image_index][y][x]
		dyy = DoG_images[image_index][y + 1][x] + DoG_images[image_index][y - 1][x] - 2 * DoG_images[image_index][y][x]
		dzz = DoG_images[image_index][y][x + 1] + DoG_images[image_index][y][x - 1] - 2 * DoG_images[image_index][y][x]
		dxy = ((DoG_images[image_index + 1][y + 1][x] - DoG_images[image_index - 1][y + 1][x]) - (DoG_images[image_index + 1][y - 1][x] - DoG_images[image_index - 1][y - 1][x])) / 4
		dyz = ((DoG_images[image_index][y + 1][x + 1] - DoG_images[image_index][y - 1][x + 1]) - (DoG_images[image_index][y + 1][x - 1] - DoG_images[image_index][y - 1][x - 1])) / 4
		dzx = ((DoG_images[image_index + 1][y][x + 1] - DoG_images[image_index + 1][y][x - 1]) - (DoG_images[image_index - 1][y][x + 1] - DoG_images[image_index - 1][y][x - 1])) / 4
		return np.array([[dxx , dxy , dzx] , 
						[dxy , dyy , dyz] , 
						[dzx , dyz , dzz]])
	
	def find_orientation(self , keypoint , Gaussian_image , octave_index , number_of_bin = 36 , peak_ratio = 0.8):
		scale = 1.5 * (keypoint.size / 2**(octave_index + 1))
		radius = int(np.round(3 * scale))
		weight_factor = -0.5 / scale** 2
		histogram = np.zeros(number_of_bin)
		smooth_histogram = np.zeros(number_of_bin)
		height , width = Gaussian_image.shape
		x = int(np.round(keypoint.x / 2**octave_index))
		y = int(np.round(keypoint.y / 2**octave_index))
		for i in range(-radius , radius + 1):
			for j in range(-radius , radius + 1):
				if x + i <= 0 or x + i >= width - 1 or y + j <= 0 or y + j >= height - 1:
					continue
				dx = Gaussian_image[y + j , x + i + 1] - Gaussian_image[y + j , x + i - 1]
				dy = Gaussian_image[y + j - 1 , x + i] - Gaussian_image[y + j + 1 , x + i]
				gradient_magnitude = np.sqrt(dx**2 + dy**2)
				gradient_degree = np.rad2deg(np.arctan2(dy , dx))
				weight = np.exp(weight_factor * (i**2 + j**2))
				histogram[int(np.round(gradient_degree / (360 / number_of_bin)))] += weight * gradient_magnitude
		
		for i in range(number_of_bin):
			smooth_histogram[i] = (histogram[i - 2] + 4 * histogram[i - 1] + 6 * histogram[i] + 4 * histogram[(i + 1) % number_of_bin] + histogram[(i + 2) % number_of_bin]) / 16

		keypoints = []
		for i in range(number_of_bin):
			if smooth_histogram[i] > smooth_histogram[i - 1] and smooth_histogram[i] > smooth_histogram[(i + 1) % number_of_bin] and smooth_histogram[i] >= peak_ratio * np.max(smooth_histogram):
				orientation = 360 - (360 / number_of_bin) * ((i + 0.5 * (smooth_histogram[i - 1] - smooth_histogram[(i + 1) % number_of_bin]) / (smooth_histogram[i - 1] - 2 * smooth_histogram[i] + smooth_histogram[(i + 1) % number_of_bin])) % number_of_bin)
				if abs(orientation - 360) < 1e-7:
					orientation = 0
				oriented_keypoint = Keypoint(
					keypoint.x , 
					keypoint.y , 
					orientation , 
					keypoint.octave , 
					keypoint.size , 
					keypoint.response
				)
				keypoints.append(oriented_keypoint)
		return keypoints

	def keypoint_detection(self , image):
		self.generate_base_image(image)
		self.construct_pyramid()
		keypoints = self.keypoint_localization()
		return keypoints

	def keypoint_description(self , keypoints , window_size = 4 , number_of_bin = 8):
		descriptors = []
		for keypoint in keypoints:
			octave_index = keypoint.octave & 255
			image_index = (keypoint.octave >> 8) & 255
			if octave_index >= 128:
				octave_index = octave_index | -128
			scale = 1 / (1 << octave_index) if octave_index >= 0 else 1 << -octave_index

			Gaussian_image = self.Gaussian_pyramid[octave_index + 1][image_index]
			height , width = Gaussian_image.shape
			x = int(np.round(scale * keypoint.x))
			y = int(np.round(scale * keypoint.y))
			degree = 360 - keypoint.orientation

			histogram_width = 1.5 * scale * keypoint.size
			radius = int(np.round(np.sqrt(0.5) * histogram_width * (window_size + 1)))
			radius = int(min(radius , np.sqrt(height**2 + width**2)))
			bin_list , magnitude_list , degree_list = [] , [] , []
			for i in range(-radius , radius + 1):
				for j in range(-radius , radius + 1):
					rotated_i = j * np.sin(np.deg2rad(degree)) + i * np.cos(np.deg2rad(degree))
					rotated_j = j * np.cos(np.deg2rad(degree)) - i * np.sin(np.deg2rad(degree))
					bin_i = (rotated_i / histogram_width) + 0.5 * (window_size - 1)
					bin_j = (rotated_j / histogram_width) + 0.5 * (window_size - 1)
					if bin_i <= -1 or bin_i >= window_size or bin_j <= -1 or bin_j >= window_size:
						continue
					window_i = int(round(y + i))
					window_j = int(round(x + j))
					if window_i <= 0 or window_i >= height - 1 or window_j <= 0 or window_j >= width - 1:
						continue
					dx = Gaussian_image[window_i , window_j + 1] - Gaussian_image[window_i , window_j - 1]
					dy = Gaussian_image[window_i - 1 , window_j] - Gaussian_image[window_i + 1 , window_j]
					gradient_magnitude = np.sqrt(dx**2 + dy**2)
					gradient_degree = np.rad2deg(np.arctan2(dy , dx)) % 360
					weight = np.exp((-2 / window_size**2) * ((rotated_i / histogram_width)**2 + (rotated_j / histogram_width)**2))
					bin_list.append((bin_i , bin_j))
					magnitude_list.append(weight * gradient_magnitude)
					degree_list.append((gradient_degree - degree) * (number_of_bin / 360))

			histogram = np.zeros((window_size + 2 , window_size + 2 , number_of_bin))
			for (bin_i , bin_j) , magnitude , degree in zip(bin_list , magnitude_list , degree_list):
				bin_i , bin_i_remain = int(np.floor(bin_i)) , bin_i - int(np.floor(bin_i))
				bin_j , bin_j_remain = int(np.floor(bin_j)) , bin_j - int(np.floor(bin_j))
				degree , degree_remain = int(np.floor(degree)) % number_of_bin , degree - int(np.floor(degree))
				c1 = magnitude * bin_i_remain
				c0 = magnitude * (1 - bin_i_remain)
				c11 = c1 * bin_j_remain
				c10 = c1 * (1 - bin_j_remain)
				c01 = c0 * bin_j_remain
				c00 = c0 * (1 - bin_j_remain)
				c111 = c11 * degree_remain
				c110 = c11 * (1 - degree_remain)
				c101 = c10 * degree_remain
				c100 = c10 * (1 - degree_remain)
				c011 = c01 * degree_remain
				c010 = c01 * (1 - degree_remain)
				c001 = c00 * degree_remain
				c000 = c00 * (1 - degree_remain)
				histogram[bin_i + 1 , bin_j + 1 , degree] += c000
				histogram[bin_i + 1 , bin_j + 1 , (degree + 1) % number_of_bin] += c001
				histogram[bin_i + 1 , bin_j + 2 , degree] += c010
				histogram[bin_i + 1 , bin_j + 2 , (degree + 1) % number_of_bin] += c011
				histogram[bin_i + 2 , bin_j + 1 , degree] += c100
				histogram[bin_i + 2 , bin_j + 1 , (degree + 1) % number_of_bin] += c101
				histogram[bin_i + 2 , bin_j + 2 , degree] += c110
				histogram[bin_i + 2 , bin_j + 2 , (degree + 1) % number_of_bin] += c111

			descriptor = histogram[1 : -1 , 1 : -1 , : ].reshape(-1)
			threshold = 0.2 * np.linalg.norm(descriptor)
			descriptor[descriptor > threshold] = threshold
			descriptor /= max(np.linalg.norm(descriptor) , 1e-7)
			descriptor = np.round(512 * descriptor)
			descriptor[descriptor < 0] = 0
			descriptor[descriptor > 255] = 255
			descriptors.append(descriptor)
		descriptors = np.array(descriptors)
		return descriptors

def keypoint_matching(images , descriptors_list):
	all_matches_list = []
	match_count = np.zeros((len(images) , len(images)))
	for i in range(len(descriptors_list)):
		for j in range(len(descriptors_list)):
			all_matches = []
			if i != j:
				matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
				match_result = matcher.knnMatch(descriptors_list[i].astype(np.float32) , descriptors_list[j].astype(np.float32) , k = 2)
				for match_1 , match_2 in match_result:
					if match_1.distance < 0.7 * match_2.distance:
						all_matches.append(match_1)
						match_count[i][j] += 1
			all_matches_list.append(all_matches)
	return all_matches_list , match_count