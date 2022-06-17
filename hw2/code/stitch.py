import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

def RANSAC(keypoints_1 , keypoints_2 , all_matches , iteration = 1000 , threshold = 5):
	max_number_of_inlier , best_offset_x , best_offset_y = -np.inf , None , None
	for _ in range(iteration):
		selected_match = random.choice(all_matches)
		offset_x = keypoints_1[selected_match.queryIdx].x - keypoints_2[selected_match.trainIdx].x
		offset_y = keypoints_1[selected_match.queryIdx].y - keypoints_2[selected_match.trainIdx].y
		number_of_inlier = 0
		for match in all_matches:
			error = (keypoints_1[match.queryIdx].x - (keypoints_2[match.trainIdx].x + offset_x))**2 + (keypoints_1[match.queryIdx].y - (keypoints_2[match.trainIdx].y + offset_y))**2
			if error < threshold:
				number_of_inlier += 1
		if number_of_inlier > max_number_of_inlier:
			max_number_of_inlier = number_of_inlier
			best_offset_x = offset_x
			best_offset_y = offset_y
	
	good_matches = []
	for match in all_matches:
		error = (keypoints_1[match.queryIdx].x - (keypoints_2[match.trainIdx].x + best_offset_x))**2 + (keypoints_1[match.queryIdx].y - (keypoints_2[match.trainIdx].y + best_offset_y))**2
		if error < threshold:
			good_matches.append(match)
			
	return int(np.round(best_offset_x)) , int(np.round(best_offset_y)) , good_matches

def image_match_verification(all_matches , good_matches):
	return len(all_matches) > 5.9 + 0.22 * len(good_matches)

def linear_blending(image_1 , image_2 , mask_1 , mask_2 , offset_x , offset_y):
	result_width = image_2.shape[1] + abs(offset_x) if offset_x > 0 else image_1.shape[1] + abs(offset_x)
	result_height = image_2.shape[0] + abs(offset_y) if offset_y > 0 else image_1.shape[0] + abs(offset_y)

	translation_matrix = np.array([[1 , 0 , max(0 , -offset_x)] , [0 , 1 , max(0 , -offset_y)]]).astype(np.float32)
	translated_image_1 = cv2.warpAffine(image_1 , translation_matrix , (result_width , result_height))
	translated_mask_1 = cv2.warpAffine(mask_1 , translation_matrix , (result_width , result_height))

	translation_matrix = np.array([[1 , 0 , max(0 , offset_x)] , [0 , 1 , max(0 , offset_y)]]).astype(np.float32)
	translated_image_2 = cv2.warpAffine(image_2 , translation_matrix , (result_width , result_height))
	translated_mask_2 = cv2.warpAffine(mask_2 , translation_matrix , (result_width , result_height))

	if offset_x > 0:
		weight = np.concatenate([np.ones(abs(offset_x)) , np.linspace(1 , 0 , image_1.shape[1] - abs(offset_x)) , np.zeros(abs(offset_x) + image_2.shape[1] - image_1.shape[1])] , axis = 0)
	else:
		weight = np.concatenate([np.zeros(abs(offset_x)) , np.linspace(0 , 1 , image_2.shape[1] - abs(offset_x)) , np.ones(abs(offset_x) + image_1.shape[1] - image_2.shape[1])] , axis = 0)
	weight = np.repeat(np.expand_dims(np.repeat(np.expand_dims(weight , axis = 0) , result_height , axis = 0) , axis = -1) , 3 , axis = 2)

	result = weight * translated_image_1 + (1 - weight) * translated_image_2

	result[np.logical_and(translated_mask_1 , np.logical_not(translated_mask_2))] = translated_image_1[np.logical_and(translated_mask_1 , np.logical_not(translated_mask_2))]
	result[np.logical_and(translated_mask_2 , np.logical_not(translated_mask_1))] = translated_image_2[np.logical_and(translated_mask_2 , np.logical_not(translated_mask_1))]
	result_mask = np.logical_or(translated_mask_1 , translated_mask_2)

	return result.astype(np.uint8) , result_mask.astype(np.uint8)

def stitch_images(images , masks , keypoints_list , all_matches_list , match_count , potential_image_match = 6 , start_index = 0):
	next_image_index = [start_index]
	used = np.zeros(len(images))
	used[start_index] = 1
	total_offset_x = np.zeros(len(images)).astype(int)
	total_offset_y = np.zeros(len(images)).astype(int)
	result = images[start_index].copy()
	result_mask = masks[start_index].copy()
	while next_image_index:
		i = next_image_index[0]
		next_image_index.pop(0)
		for j in np.flip(np.argsort(match_count[i]))[ : potential_image_match]:
			if used[j]:
				continue
			all_matches = all_matches_list[i * len(images) + j]
			offset_x , offset_y , good_matches = RANSAC(keypoints_list[i] , keypoints_list[j] , all_matches)
			if image_match_verification(all_matches , good_matches):
				offset_x += total_offset_x[i]
				offset_y += total_offset_y[i]
				result , result_mask = linear_blending(result , images[j] , result_mask , masks[j] , offset_x , offset_y)
				for k in range(len(images)):
					if used[k]:
						total_offset_x[k] += max(0 , -offset_x)
						total_offset_y[k] += max(0 , -offset_y)
				total_offset_x[j] = max(0 , offset_x)
				total_offset_y[j] = max(0 , offset_y)
				used[j] = 1
				next_image_index.append(j)

				plt.figure(figsize = (18 , 6))
				plt.title(f'Stitch Image {j + 1} with Image {i + 1}' , fontsize = 18 , y = 1.05)
				plt.imshow(cv2.cvtColor(result , cv2.COLOR_BGR2RGB))
				plt.axis('off')
				plt.show()

	return result

def global_alignment(image , border = 5):
	height , width , _ = image.shape
	x1 , y1 = 0 , np.where(np.sum(image , axis = 2)[ : , 0] != 0)[0][0]
	x2 , y2 = 0 , np.where(np.sum(image , axis = 2)[ : , 0] != 0)[0][-1]
	x3 , y3 = width - 1 , np.where(np.sum(image , axis = 2)[ : , -1] != 0)[0][-1]
	x4 , y4 = width - 1 , np.where(np.sum(image , axis = 2)[ : , -1] != 0)[0][0]
	new_height = min(abs(y1 - y2) , abs(y3 - y4))
	new_width = min(abs(x1 - x4) , abs(x2 - x3))
	source_points = np.array([[x1 , y1] , [x2 , y2] , [x3 , y3] , [x4 , y4]]).astype(np.float32)
	destination_points = np.array([[0 , 0] , [0 , new_height - 1] , [new_width - 1 , new_height - 1] , [new_width - 1 , 0]]).astype(np.float32)
	transform_matrix = cv2.getPerspectiveTransform(source_points , destination_points)
	return cv2.warpPerspective(image , transform_matrix , (new_width , new_height))[border : -border , border : -border , : ]