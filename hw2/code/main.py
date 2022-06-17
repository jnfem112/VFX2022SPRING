import argparse
import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from time import time
from warp import cylindrical_projection
from feature import SIFT , keypoint_matching
from stitch import stitch_images , global_alignment

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_directory' , type = str , default = '../data/input/scene_1/')
	parser.add_argument('--scale' , type = float , default = 0.2)
	parser.add_argument('--focal_length' , type = int , default = 7893)
	parser.add_argument('--global_alignment' , type = int , default = 1)
	parser.add_argument('--output_image' , type = str , default = '../result.png')
	args = parser.parse_args()

	# Load images.
	print('Load images...')
	images = []
	for file_name in os.listdir(args.input_directory):
		if file_name.endswith('jpg') or file_name.endswith('JPG') or file_name.endswith('png') or file_name.endswith('PNG'):
			image = cv2.imread(os.path.join(args.input_directory , file_name) , cv2.IMREAD_COLOR)
			image = cv2.resize(image , (int(args.scale * image.shape[1]) , int(args.scale * image.shape[0])) , cv2.INTER_LINEAR)
			images.append(image)
	
	fig , axes = plt.subplots(int(np.ceil(len(images) / 6)) , 6 , figsize = (12 , 6))
	for i , image in enumerate(images):
		ax = axes[i // 6 , i % 6]
		ax.set_title(f'Image {i + 1}')
		ax.imshow(cv2.cvtColor(image , cv2.COLOR_BGR2RGB))
		ax.set_axis_off()
	plt.suptitle('Original Images' , fontsize = 18 , y = 0.975)
	plt.savefig('../data/output/original_images.jpg')
	plt.show()

	# Cylindrical projection...
	if args.focal_length:
		print('Cylindrical projection...')
		new_images , masks = [] , []
		for i in range(len(images)):
			print(f'    Process image {i + 1} ' , end = '')
			start = time()
			new_image , mask = cylindrical_projection(images[i] , int(args.scale * args.focal_length))
			end = time()
			print(f'({int(end - start)}s)')
			new_images.append(new_image)
			masks.append(mask)
		images = new_images
	else:
		masks = [np.ones(image.shape) for image in images]
	
	if args.focal_length:
		fig , axes = plt.subplots(int(np.ceil(len(images) / 6)) , 6 , figsize = (12 , 6))
		for i , image in enumerate(images):
			ax = axes[i // 6 , i % 6]
			ax.set_title(f'Image {i + 1}')
			ax.imshow(cv2.cvtColor(image , cv2.COLOR_BGR2RGB))
			ax.set_axis_off()
		plt.suptitle('Cylindrical Projection' , fontsize = 18 , y = 0.975)
		plt.savefig('../data/output/cylindrical_projection.jpg')
		plt.show()
	
	# Feature detection & feature description.
	print('Feature detection & feature description...')
	keypoints_list , descriptors_list = [] , []
	for i , image in enumerate(images):
		print(f'    Image {i + 1}:')
		sift = SIFT()
		print('        Keypoint detection ' , end = '')
		start = time()
		keypoints = sift.keypoint_detection(image)
		end = time()
		print(f'({int(end - start)}s)')
		keypoints_list.append(keypoints)
		print('        Keypoint description ' , end = '')
		start = time()
		descriptors = sift.keypoint_description(keypoints)
		end = time()
		print(f'({int(end - start)}s)')
		descriptors_list.append(descriptors)
	
	fig , axes = plt.subplots(int(np.ceil(len(images) / 6)) , 6 , figsize = (12 , 6))
	for i , (image , keypoints) in enumerate(zip(images , keypoints_list)):
		SIFT_image = image.copy()
		for keypoint in keypoints:
			cv2.circle(SIFT_image , (int(keypoint.x) , int(keypoint.y)) , 1 , (0 , 0 , 255) , 2)
		ax = axes[i // 6 , i % 6]
		ax.set_title(f'Image {i + 1}')
		ax.imshow(cv2.cvtColor(SIFT_image , cv2.COLOR_BGR2RGB))
		ax.set_axis_off()
	plt.suptitle('SIFT' , fontsize = 18 , y = 0.975)
	plt.savefig('../data/output/SIFT.jpg')
	plt.show()

	# Feature matching.
	print('Feature matching...')
	all_matches_list , match_count = keypoint_matching(images , descriptors_list)

	plt.figure(figsize = (7.5 , 7.5))
	plt.title('Number of Keypoint Match' , fontsize = 18 , y = 1.02)
	plt.imshow(match_count)
	plt.xticks(np.arange(len(images)) , [f'Image {i + 1}' for i in range(len(images))] , rotation = 90)
	plt.yticks(np.arange(len(images)) , [f'Image {i + 1}' for i in range(len(images))] , rotation = 0)
	plt.savefig('../data/output/keypoint_matching.jpg')
	plt.show()

	# Image matching & image blending.
	print('Image matching & image blending...')
	result = stitch_images(images , masks , keypoints_list , all_matches_list , match_count , potential_image_match = 2 , start_index = 1)
	
	plt.figure(figsize = (18 , 6))
	plt.title('Image Stitching' , fontsize = 18 , y = 1.05)
	plt.imshow(cv2.cvtColor(result , cv2.COLOR_BGR2RGB))
	plt.axis('off')
	plt.savefig('../data/output/image_stitching.jpg')
	plt.show()

	# End-to-end alignment & crop.
	if args.global_alignment:
		print('End-to-end alignment & crop...')
		result = global_alignment(result)

		plt.figure(figsize = (18 , 6))
		plt.title('Global Alignment & Crop' , fontsize = 18 , y = 1.05)
		plt.imshow(cv2.cvtColor(result , cv2.COLOR_BGR2RGB))
		plt.axis('off')
		plt.savefig('../data/output/global_alignment.jpg')
		plt.show()
	
	cv2.imwrite(args.output_image , result)