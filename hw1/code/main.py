import os
import pandas as pd
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from align import MTB
from HDR import Debevec , Robertson
from tone_mapping import Reinhard_global , Reinhard_local

if __name__ == '__main__':
	# Load images.
	print('Load images...')
	df = pd.read_csv('../data/image_list.csv')
	df = df.sort_values('exposure_time' , ascending = False)
	images , exposure_times = [] , []
	for image_name , exposure_time in zip(df['image_name'].values , df['exposure_time'].values):
		image = cv2.imread(os.path.join('../data/' , image_name) , cv2.IMREAD_COLOR)
		images.append(image)
		exposure_times.append(exposure_time)

	fig , axes = plt.subplots(math.ceil(len(images) / 4) , 4 , figsize = (10 , 6))
	for i , (image , exposure_time) in enumerate(zip(images , exposure_times)):
		ax = axes[i // 4 , i % 4]
		ax.set_title(f'Exposure Time : {exposure_time}' , fontsize = 8)
		ax.imshow(cv2.cvtColor(image , cv2.COLOR_BGR2RGB))
		ax.set_axis_off()
	fig.suptitle('Original Images' , fontsize = 16 , y = 0.96)
	fig.savefig('../data/original_images.png')
	plt.show()

	# Align images.
	print('Align images...')
	alignment_algorithm = MTB(max_offset = 16)
	aligned_images , offset = alignment_algorithm.align_images(images)

	fig , axes = plt.subplots(math.ceil(len(images) / 4) , 4 , figsize = (10 , 6))
	for i , (image , (offset_x , offset_y)) in enumerate(zip(aligned_images , offset)):
		ax = axes[i // 4 , i % 4]
		ax.set_title(f'Shift : ({offset_x} , {offset_y})' , fontsize = 8)
		ax.imshow(cv2.cvtColor(image , cv2.COLOR_BGR2RGB))
		ax.set_axis_off()
	fig.suptitle('MTB Algorithm' , fontsize = 16 , y = 0.96)
	fig.savefig('../data/MTB.png')
	plt.show()

	# HDR reconstruction.
	print('HDR reconstruction...')

	print('    Debevec\'s method...')
	HDR_algorithm = Debevec()
	HDR_algorithm.solve_response_curve(images , exposure_times)
	HDR_Debevec = HDR_algorithm.reconstruct_HDR_image(images , exposure_times)
	cv2.imwrite('../data/HDR_Debevec.hdr' , HDR_Debevec)

	fig , axes = plt.subplots(1 , 3 , figsize = (12 , 4))
	for i , channel in enumerate(['Blue' , 'Green' , 'Red']):
		ax = axes[i]
		ax.set_title(f'Radiance Map ({channel})')
		im = ax.imshow(np.log(HDR_Debevec[ : , : , i]) , cmap = 'jet')
		ax.set_axis_off()
		divider = make_axes_locatable(ax)
		cax = divider.append_axes('right' , size = '5%' , pad = 0.1)
		fig.colorbar(im , cax = cax , orientation = 'vertical')
	fig.suptitle('Debevec\'s Method' , fontsize = 16 , y = 0.9)
	fig.savefig('../data/radiance_map_Debevec.png')
	plt.show()

	fig , axes = plt.subplots(1 , 3 , figsize = (16 , 6))
	for i , channel in enumerate(['Blue' , 'Green' , 'Red']):
		ax = axes[i]
		ax.set_title(f'Response Curve ({channel})')
		ax.plot(HDR_algorithm.response_curves[i] , np.arange(256) , c = channel)
		ax.set_xlabel('Log Exposure')
		ax.set_ylabel('Pixel Value')
		ax.grid()
	fig.suptitle('Debevec\'s Method' , fontsize = 16)
	fig.savefig('../data/response_curve_Debevec.png')
	plt.show()

	print('    Robertson\'s method...')
	HDR_algorithm = Robertson()
	HDR_algorithm.solve_response_curve(images , exposure_times)
	HDR_Robertson = HDR_algorithm.reconstruct_HDR_image(images , exposure_times)
	cv2.imwrite('../data/HDR_Robertson.hdr' , HDR_Robertson)

	fig , axes = plt.subplots(1 , 3 , figsize = (12 , 4))
	for i , channel in enumerate(['Blue' , 'Green' , 'Red']):
		ax = axes[i]
		ax.set_title(f'Radiance Map ({channel})')
		im = ax.imshow(np.log(HDR_Robertson[ : , : , i]) , cmap = 'jet')
		ax.set_axis_off()
		divider = make_axes_locatable(ax)
		cax = divider.append_axes('right' , size = '5%' , pad = 0.1)
		fig.colorbar(im , cax = cax , orientation = 'vertical')
	fig.suptitle('Robertson\'s Method' , fontsize = 16 , y = 0.9)
	fig.savefig('../data/radiance_map_Robertson.png')
	plt.show()

	fig , axes = plt.subplots(1 , 3 , figsize = (16 , 6))
	for i , channel in enumerate(['Blue' , 'Green' , 'Red']):
		ax = axes[i]
		ax.set_title(f'Response Curve ({channel})')
		ax.plot(np.log(HDR_algorithm.response_curves[i] + 1e-8) , np.arange(256) , c = channel)
		ax.set_xlabel('Log Exposure')
		ax.set_ylabel('Pixel Value')
		ax.grid()
	fig.suptitle('Robertson\'s Method' , fontsize = 16)
	fig.savefig('../data/response_curve_Robertson.png')
	plt.show()

	# Tone mapping.
	print('Tone mapping...')
	LDR_Debevec_global = Reinhard_global(HDR_Debevec)
	LDR_Debevec_local = Reinhard_local(HDR_Debevec)
	LDR_Robertson_global = Reinhard_global(HDR_Robertson)
	LDR_Robertson_local = Reinhard_local(HDR_Robertson)
	cv2.imwrite('../data/LDR_Debevec_global.png' , LDR_Debevec_global)
	cv2.imwrite('../data/LDR_Debevec_local.png' , LDR_Debevec_local)
	cv2.imwrite('../data/LDR_Robertson_global.png' , LDR_Robertson_global)
	cv2.imwrite('../data/LDR_Robertson_local.png' , LDR_Robertson_local)

	fig , axes = plt.subplots(2 , 2 , figsize = (8 , 6))
	ax = axes[0 , 0]
	ax.set_title('Debevec + Reinhard (Global)' , fontsize = 8)
	ax.imshow(cv2.cvtColor(LDR_Debevec_global , cv2.COLOR_BGR2RGB))
	ax.set_axis_off()
	ax = axes[0 , 1]
	ax.set_title('Debevec + Reinhard (Local)' , fontsize = 8)
	ax.imshow(cv2.cvtColor(LDR_Debevec_local , cv2.COLOR_BGR2RGB))
	ax.set_axis_off()
	ax = axes[1 , 0]
	ax.set_title('Robertson + Reinhard (Global)' , fontsize = 8)
	ax.imshow(cv2.cvtColor(LDR_Robertson_global , cv2.COLOR_BGR2RGB))
	ax.set_axis_off()
	ax = axes[1 , 1]
	ax.set_title('Robertson + Reinhard (Local)' , fontsize = 8)
	ax.imshow(cv2.cvtColor(LDR_Robertson_local , cv2.COLOR_BGR2RGB))
	ax.set_axis_off()
	fig.suptitle('Tone Mapping' , fontsize = 16 , y = 0.96)
	fig.savefig('../data/tone_mapping.png')
	plt.show()

	print('========== Compare to OpenCV ==========')

	# HDR reconstruction.
	print('HDR reconstruction using OpenCV...')

	print('    Debevec\'s method...')
	calibrate_Debevec = cv2.createCalibrateDebevec()
	response_curve_Debevec = calibrate_Debevec.process(np.array(images).copy() , times = np.array(exposure_times , dtype = np.float32).copy())
	HDR_algorithm = cv2.createMergeDebevec()
	HDR_Debevec = HDR_algorithm.process(np.array(images).copy() , times = np.array(exposure_times , dtype = np.float32).copy() , response = response_curve_Debevec)
	cv2.imwrite('../data/HDR_Debevec_OpenCV.hdr' , HDR_Debevec)

	fig , axes = plt.subplots(1 , 3 , figsize = (12 , 4))
	for i , channel in enumerate(['Blue' , 'Green' , 'Red']):
		ax = axes[i]
		ax.set_title(f'Radiance Map ({channel})')
		im = ax.imshow(np.log(HDR_Debevec[ : , : , i]) , cmap = 'jet')
		ax.set_axis_off()
		divider = make_axes_locatable(ax)
		cax = divider.append_axes('right' , size = '5%' , pad = 0.1)
		fig.colorbar(im , cax = cax , orientation = 'vertical')
	fig.suptitle('Debevec\'s Method (Using OpenCV)' , fontsize = 16 , y = 0.9)
	fig.savefig('../data/radiance_map_Debevec_OpenCV.png')
	plt.show()

	fig , axes = plt.subplots(1 , 3 , figsize = (16 , 6))
	for i , channel in enumerate(['Blue' , 'Green' , 'Red']):
		ax = axes[i]
		ax.set_title(f'Response Curve ({channel})')
		ax.plot(np.log(response_curve_Debevec[ : , : , i].reshape(-1)) , np.arange(256) , c = channel)
		ax.set_xlabel('Log Exposure')
		ax.set_ylabel('Pixel Value')
		ax.grid()
	fig.suptitle('Debevec\'s Method (Using OpenCV)' , fontsize = 16)
	fig.savefig('../data/response_curve_Debevec_OpenCV.png')
	plt.show()

	print('    Robertson\'s method...')
	calibrate_Robertson = cv2.createCalibrateRobertson()
	response_curve_Robertson = calibrate_Robertson.process(np.array(images).copy() , times = np.array(exposure_times , dtype = np.float32).copy())
	HDR_algorithm = cv2.createMergeRobertson()
	HDR_Robertson = HDR_algorithm.process(np.array(images).copy() , times = np.array(exposure_times , dtype = np.float32).copy() , response = response_curve_Robertson)
	cv2.imwrite('../data/HDR_Robertson_OpenCV.hdr' , HDR_Robertson)

	fig , axes = plt.subplots(1 , 3 , figsize = (12 , 4))
	for i , channel in enumerate(['Blue' , 'Green' , 'Red']):
		ax = axes[i]
		ax.set_title(f'Radiance Map ({channel})')
		im = ax.imshow(np.log(HDR_Robertson[ : , : , i]) , cmap = 'jet')
		ax.set_axis_off()
		divider = make_axes_locatable(ax)
		cax = divider.append_axes('right' , size = '5%' , pad = 0.1)
		fig.colorbar(im , cax = cax , orientation = 'vertical')
	fig.suptitle('Robertson\'s Method (Using OpenCV)' , fontsize = 16 , y = 0.9)
	fig.savefig('../data/radiance_map_Robertson_OpenCV.png')
	plt.show()

	fig , axes = plt.subplots(1 , 3 , figsize = (16 , 6))
	for i , channel in enumerate(['Blue' , 'Green' , 'Red']):
		ax = axes[i]
		ax.set_title(f'Response Curve ({channel})')
		ax.plot(np.log(response_curve_Robertson[ : , : , i].reshape(-1)) , np.arange(256) , c = channel)
		ax.set_xlabel('Log Exposure')
		ax.set_ylabel('Pixel Value')
		ax.grid()
	fig.suptitle('Robertson\'s Method (Using OpenCV)' , fontsize = 16)
	fig.savefig('../data/response_curve_Robertson_OpenCV.png')
	plt.show()

	# Tone mapping.
	print('Tone mapping using OpenCV...')
	tone_mapping_algorithm = cv2.createTonemapReinhard(1.5 , 0 , 0 , 0)
	LDR_Debevec_global = tone_mapping_algorithm.process(HDR_Debevec.copy())
	LDR_Debevec_global = np.clip(255 * LDR_Debevec_global , 0 , 255).astype(np.uint8)
	LDR_Robertson_global = tone_mapping_algorithm.process(HDR_Robertson.copy())
	LDR_Robertson_global = np.clip(255 * LDR_Robertson_global , 0 , 255).astype(np.uint8)
	cv2.imwrite('../data/LDR_Debevec_global_OpenCV.png' , LDR_Debevec_global)
	cv2.imwrite('../data/LDR_Robertson_global_OpenCV.png' , LDR_Robertson_global)

	fig , axes = plt.subplots(1 , 2 , figsize = (8 , 4))
	ax = axes[0]
	ax.set_title('Debevec + Reinhard (Global)')
	ax.imshow(cv2.cvtColor(LDR_Debevec_global , cv2.COLOR_BGR2RGB))
	ax.set_axis_off()
	ax = axes[1]
	ax.set_title('Robertson + Reinhard (Global)')
	ax.imshow(cv2.cvtColor(LDR_Robertson_global , cv2.COLOR_BGR2RGB))
	ax.set_axis_off()
	fig.suptitle('Tone Mapping (Using OpenCV)' , fontsize = 16 , y = 0.92)
	fig.savefig('../data/tone_mapping_OpenCV.png')
	plt.show()