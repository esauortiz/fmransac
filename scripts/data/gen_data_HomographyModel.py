from estimation.myfit import HomographyModel, is_inlier, get_residuals
from gen_data import plane_nd_data, get_bbox_limits, is_in_bbox, gen_uniform_noise

import numpy as np
import random
import cv2 as cv
import sys
import yaml
import io

from multiprocessing import Pool

def _check_dim_match(data, target_dim):
	"""Checks dimension match beetween data and requested target dimensions
	Parameters
	----------
	data : (N, dim) array
	    N points with dim dimensions
	target_dim : tuple
		Indicating the dimensions to which noise is added
	Returns
	-------
	target_dim : tuple
	Raises
	------
	ValueError
	    If there is not a match
	"""
	target_dim = list(set(target_dim))

	if max(target_dim) >= np.shape(data)[1]:
		raise ValueError('The dimension cannot be greater than %d' % (np.shape(data)[1] - 1) )
	else:
		return target_dim

def gen_gauss_noise(data, model_class, model_params, residual_threshold, points_num, 
					max_trials, mu, sigma, target_dim, outlier_ratio, seed, bbox_limits = None, 
					bbox_limits_tolerance = 0):
	"""Adds gaussian distributed noise
	Parameters
	----------
	Returns
	-------
	noisy data : (N, dim) array
	"""
	random_state = np.random.RandomState(seed)
	np.random.seed(seed)
	
	dim = 2
	target_dim = _check_dim_match(data, target_dim)
	idx = 0
	trials = 0
	
	while idx < points_num and trials < max_trials:
		
		point = data[idx]
		# Adding Gaussian noise to both dimensions x and y
		for dim in target_dim:
			noise = np.random.normal(mu, sigma)
			point[dim] += noise

		# Check if noisy point is inside of camera bbox limits
		if is_in_bbox([point], bbox_limits):
			data[idx] = point
			idx +=1

		trials += 1
		if trials == max_trials:
			raise RuntimeError('Number of trials has been exceeded. Noise has not been generated')

	return data

def gen_uniform_noise(matching_matrix, inliers_num, seed):
	"""Adds uniform noise changing matches as outliers
	Parameters
	----------
	matching_matrix: (N, 2) array
	Returns
	----------
	noisy_macthing_matrix: (N, 2) array
	"""
	"""
	random_state = np.random.RandomState(seed)
	matches_num = matching_matrix.shape[0]
	outliers_num = matches_num - inliers_num

	outliers_idxs = random_state.choice(matches_num, int(outliers_num), replace=False)
	# reversed outliers idxs
	r_outliers_idxs = outliers_idxs[::-1]
	matching_matrix[outliers_idxs,1]=matching_matrix[r_outliers_idxs,0]
	"""

	matches_num = matching_matrix.shape[0]
	outliers_idxs = matching_matrix[inliers_num:,1]
	# reversed outliers idxs
	r_outliers_idxs = outliers_idxs[::-1]
	matching_matrix[inliers_num:,1]=r_outliers_idxs
	return matching_matrix

def fit_plane_in_bbox(data, points_num, seed, bbox_limits = None):
	"""Returns plane points inside the bbox
	Points in data are uniformly distributed (both in the space and
	in the array indexes.)
	Parameters
	----------
	Returns
	-------
	Plane points inside bbox : (N, 3) array
	"""
	max_trials = points_num*100
	dim = 3

	i = 0
	trials = 0
	points = np.empty((int(points_num), dim), dtype = float)

	for point in data:
		points[i] = point
		if is_in_bbox(np.asarray([point]), bbox_limits):
			i += 1

		trials += 1
		if trials == max_trials:
			raise RuntimeError('Number of trials has been exceeded. Noise has not been generated')
		if i == points_num:
			break

	return np.stack(points, axis = 0)

def is_data_in_bbox(data, bbox):
	for coord in data:
		if not is_in_bbox([coord], bbox):
			return False
	return True

def get_homography(cam_pose1, cam_pose2):
	""" Computes homography from two cam poses
	cam pose is defined by [tx, ty, tz, theta]
	"""
	n = np.array([0,0,1])
	theta = cam_pose2[3] - cam_pose1[3]

	R = np.array([	[np.cos(theta), -np.sin(theta), 0], 
					[np.sin(theta),  np.cos(theta), 0],
					[0,				 0,				1]])

	t = cam_pose2[0:3] - cam_pose1[0:3]

	
	
	f = 0.010
	sz = 1/(10*10**(-6))
	ox, oy = [512, 384]

	k = np.array([	[f*sz, 	0, 		ox],
					[0, 	f*sz, 	oy],
					[0,		0,		1]])

	l = np.array([	[1, 0,	0, 0], 
					[0, 1,	0, 0],
					[0,	0,	1, 0]])

	tf = np.matmul(k, l)
	trans = np.dot(np.array([*t, 1]), tf.T)
	#trans /= trans[2]
	t[0:2] = trans[0:2]
	#print(t/cam_pose1[2])
	H = R
	H[:,2] += t/cam_pose1[2] # simplifying t \cdot n
	#H = np.linalg.inv(H)
	H /= H[2,2]
	#print(H)

	return H

def add_ones_column(data):
	"""Adds ones column in data
	data.shape is suposed to be (points_num, N)
	"""
	points_num, N = data.shape
	ones = np.ones((points_num, N+1))
	ones[:,:-1] = data

	return ones

def rmv_ones_column(data):
	# normalize data
	points_num, N = data.shape
	data /= data[:,N-1].reshape(points_num, 1)
	# remove "ones" column
	data = data[:,:-1]
	return data

def get_valid_camera_pose(tx_rng, ty_rng, tz_rng, theta_rng, plane_data, k, image_bbox, seed):
	"""Returns camera pose which includes all points of "plane_data"		
	"""

	is_valid_pose = False
	points_num = plane_data.shape[0]
	plane_data = add_ones_column(plane_data)

	#random.seed(None)

	while is_valid_pose == False:
		# generar extrinsics
		# fraction of pi from 0 to 2*pi, random fraction is generated rand(0,k)/k
		theta = np.deg2rad(random.randint(*theta_rng))
		tx = random.randint(*tx_rng)
		ty = random.randint(*ty_rng)
		tz = random.randint(*tz_rng)*50

		t = np.array([	[np.cos(theta), -np.sin(theta),	0, tx], 
						[np.sin(theta),  np.cos(theta),	0, ty],
						[0,				 0,				1, tz]])

		tf = np.matmul(k, t)
		# project plane_data to the camera position
		proj_data = np.dot(plane_data, tf.T)
		proj_data = rmv_ones_column(proj_data)
		# check if proj_data is in image
		if is_data_in_bbox(proj_data, image_bbox):
			is_valid_pose = True

	cam_pose = np.array([tx,ty,tz,theta])
	#print(cam_pose)
	return cam_pose, proj_data

def is_the_same_pose(pose1, pose2):
	for i in range(np.size(pose1)):
		if pose1[i] != pose2[i]:
			return False
	return True

def get_data_from_real_source(src_img, dst_img):

	path_header = '/home/esau/tfm/codigo_fuente/homography_dataset/source_img'
	MIN_MATCH_COUNT = 10
	img1 = cv.imread(f'{path_header}/{src_img}.JPG',0) # queryImage
	img2 = cv.imread(f'{path_header}/{dst_img}.JPG',0) # trainImage
	# Initiate SIFT detector
	sift = cv.SIFT_create()
	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)

	FLANN_INDEX_KDTREE = 1
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 50)
	flann = cv.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des1,des2,k=2)
	# store all the good matches as per Lowe's ratio test.
	good = []
	for m,n in matches:
	    if m.distance < 0.7*n.distance:
	        good.append(m)

	if len(good)>MIN_MATCH_COUNT:
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,2)
		M, inliers = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

	else:
	    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))

	return src_pts, dst_pts, inliers

def get_original_model(src_img, dst_img, inliers):
	path_header = '/home/esau/tfm/codigo_fuente/homography_dataset/source_img'
	src_img_kp = np.loadtxt(f'{path_header}/{src_img}_kp.txt') # source image keypoint coordinates
	dst_img_kp = np.loadtxt(f'{path_header}/{dst_img}_kp.txt') # second image keypoint coordinates 
	data = np.column_stack((src_img_kp[inliers],dst_img_kp[inliers]))

	model = HomographyModel()
	model.estimate(data)
	residuals = np.abs(model.residuals(*data))
	inliers = residuals < 5 # euristic threshold value

	return model, inliers
	

if __name__ == '__main__':

	# fixed plane bbox
	plane_bbox = [[-2,2], [-2,2], [-2,2]]
	image_bbox = [[0.,3000.], [0.,3000.]]
	# intrinsic matrix k
	f = 0.010
	sz = 1/(10*10**(-6))
	ox, oy = [512, 384]

	k = np.array([	[f*sz, 	0, 		ox],
					[0, 	f*sz, 	oy],
					[0,		0,		1]])

	test_num = int(sys.argv[1])
	yaml_file_path = sys.argv[2] + '/yaml/test_'
	
	def generate_data(test_id):

		# set random seed
		#random.seed(test_id)
		#np.random.seed(test_id)
		synthetic_data = True
		# if synthetic_data == False we should provide src and dst image to find real correspondences
		src_img = 'IMG_3326'
		dst_img = 'IMG_3327'

		yaml_file = yaml_file_path + str(test_id) + '.yaml'
		# Read test params
		with open(yaml_file, 'r') as stream:
			test_params = yaml.safe_load(stream)

		# test params
		data_params = test_params['data_params']
		theta_rng, tx_rng, ty_rng, tz_rng, origin, direction = data_params['model_params']
	
		# force xy planes
		direction = [0,0,1]

		data_len = data_params['data_len']
		data = [[]]

		#seed = test_params['seed']
		seed = 50

		# noise params
		gn_params = data_params['gn_params']
		un_params = data_params['un_params']

		if synthetic_data == True:
			# generating 3D points and fitting them in a bounding box
			# test_id as random seed # more than points_num are generated
			plane_data = plane_nd_data(data_len[0], [100*int(*data_len[1])], int(1), origin, direction)

			try:
				# fit plane in plane_bbox, only points_num will be selected
				plane_data = fit_plane_in_bbox(plane_data, points_num = int(*data_len[1]), seed = int(1), bbox_limits = plane_bbox)
				#g = np.array([np.linspace(2,4,10),np.linspace(2,6,10),np.repeat(0,10)]).T
				#plane_data[:10,:]=g
				
				# reset random seed
				random.seed(seed)
				np.random.seed(seed)
				
				# setting camera poses
				cam_pose1, proj_data1 = get_valid_camera_pose(tx_rng, ty_rng, tz_rng, theta_rng, plane_data, k, image_bbox, seed)
				cam_pose2 = cam_pose1
				while is_the_same_pose(cam_pose1, cam_pose2):
					cam_pose2, proj_data2 = get_valid_camera_pose(tx_rng, ty_rng, tz_rng, theta_rng, plane_data, k, image_bbox, seed)
				# computing homography from two projected data in two camera positions
				model = HomographyModel()
				model.estimate(np.column_stack((proj_data1,proj_data2)))
				#H =  get_homography(cam_pose1, cam_pose2)
				#print(H)
				#print(np.array(model.H))
				model_original_params = model.params
				
				# proj_data will be contaminated with gaussian noise in x and y
				#proj_data1 = gen_gauss_noise(proj_data1, None, None, *gn_params, seed, bbox_limits = image_bbox)
				"""
				f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, sharex = True)
				# First subplot
			
				ax1.plot(proj_data1[:,0], proj_data1[:,1], 'bo')
				ax2.plot(proj_data2[:,0], proj_data2[:,1], 'bo')

				proj2 = model.get_projection(proj_data1)
				ax3.plot(proj2[:,0], proj2[:,1], 'bo')

				plt.show()
				"""
				proj_data2 = gen_gauss_noise(proj_data2, None, None, *gn_params, seed, bbox_limits = image_bbox)
				# se cambia el orden de las correspondencias para emular outliers
				correspondences = np.array([range(0, proj_data1.shape[0]), range(0, proj_data1.shape[0])]).T
				inliers_num = int(*data_len[1]) - un_params[1] # total - outliers_num
				correspondences = gen_uniform_noise(correspondences, inliers_num = inliers_num, seed = seed)
				# sort proj_data i.e. correspondences are in the same index in both array
				proj_data1 = np.array([x for _,x in sorted(zip(correspondences[:,0],proj_data1))])
				proj_data2 = np.array([x for _,x in sorted(zip(correspondences[:,1],proj_data2))])
				inliers = correspondences[:,0] == correspondences[:,1]
		
			except RuntimeError:
				print('Data for ' + test_params['file_name'] + ' has not been generated')
		
		elif synthetic_data == False:
			proj_data1, proj_data2 = get_data_from_real_source(src_img, dst_img)
			model, inliers = get_original_model(src_img, dst_img)
			correspondences = np.array([range(0, proj_data1.shape[0]), range(0, proj_data1.shape[0])]).T # correspondences are already in the correct order
			model_original_params = model.params			

		# write data, correspondences and original inliesr
		np.savetxt((test_params['save_path'] + '/data/' + test_params['file_name'] + '_correspondences.txt'), correspondences)
		np.savetxt((test_params['save_path'] + '/data/' + test_params['file_name'] + '_proj1.txt'), proj_data1)
		np.savetxt((test_params['save_path'] + '/data/' + test_params['file_name'] + '_proj2.txt'), proj_data2)
		np.savetxt((test_params['save_path'] + '/data/' + test_params['file_name'] + '_inliers.txt'), inliers)
		# data has been ordered, regenerate correspondences correspondences[i] = (idx, idx)
		#correspondences = np.array([range(0, proj_data1.shape[0]), range(0, proj_data1.shape[0])]).T
		
		residuals = get_residuals(np.column_stack((proj_data1, proj_data2)), HomographyModel, model.params)
		np.savetxt((test_params['save_path'] + '/data/' + test_params['file_name'] + '_residuals.txt'), residuals)
		
		data_params['model_params'] = np.asarray(model_original_params).tolist()
		# Write YAML file (dumps H*)
		with io.open(yaml_file, 'w', encoding='utf8') as outfile:
			yaml.dump(test_params, outfile, default_flow_style=False, allow_unicode=True)

		return True

	#generate_data(test_num)
	with Pool(16) as p:
		p.map(generate_data, range(1, test_num + 1))