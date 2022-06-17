import numpy as np

def cylindrical_projection(image , focal_length):
	height , width , channel = image.shape
	new_height , new_width = height , int(2 * (focal_length * np.arctan2(width / 2 , focal_length)))
	new_image = np.zeros((new_height , new_width , channel))
	mask = np.zeros((new_height , new_width , channel))
	for new_i in range(new_height):
		for new_j in range(new_width):
			for k in range(channel):
				j = focal_length * np.tan((new_j - new_width / 2) / focal_length) + width / 2
				i = height / 2 - (new_height / 2 - new_i) / focal_length * np.sqrt((j - width / 2)**2 + focal_length**2)
				if i >= 0 and i <= height - 1 and j >= 0 and j <= width - 1:
					a00 = int(image[int(np.floor(i))][int(np.floor(j))][k])
					a01 = int(image[int(np.floor(i))][int(np.ceil(j))][k])
					a10 = int(image[int(np.ceil(i))][int(np.floor(j))][k])
					a11 = int(image[int(np.ceil(i))][int(np.ceil(j))][k])
					r0 = j - int(j)
					r1 = i - int(i)
					b0 = a00 + r0 * (a01 - a00)
					b1 = a10 + r0 * (a11 - a10)
					new_image[new_i][new_j][k] = b0 + r1 * (b1 - b0)
					mask[new_i][new_j] = 1
	return new_image.astype(np.uint8) , mask