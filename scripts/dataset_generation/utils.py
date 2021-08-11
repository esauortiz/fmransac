import numpy as np
import random

from estimation.utils import _is_inlier

def _check_dim_match(data, noise_dim):
	"""Checks dimension match beetween data and requested target dimensions
	Parameters
	----------
	data : (N, dim) array
	    N points with dim dimensions
	noise_dim : tuple
		Indicating the dimensions to which noise is added
	Returns
	-------
	noise_dim : tuple
	Raises
	------
	ValueError
	    If there is not a match
	"""
	noise_dim = list(set(noise_dim))

	if max(noise_dim) >= np.shape(data)[1]:
		raise ValueError('The dimension cannot be greater than %d' % (np.shape(data)[1] - 1) )
	else:
		return True

def _is_in_bbox(coord, bbox):
	for axis_bbox, component in zip(bbox, coord):
		if (component <= axis_bbox[0]) or (component >= axis_bbox[1]):
			return False
	return True
	
def _uniform_noise(data, original_model, dataset_params, seed = None):
	""" Uniformly distributed noise inside
	dataset_params['uniform_noise_bbox'] limits
	Parameters
	----------
	Returns
	-------
	noisy data : (N, dim) array
	"""
	np.random.seed(None)
	random.seed(None)

	origin = data.mean(axis=0)
	dataset_dim = np.size(origin)
	dataset_bbox = _get_bbox_from_value(dataset_params['dataset_bbox'], dataset_dim)

	n_points = dataset_params['n_points']
	n_outliers = int(n_points * dataset_params['outlier_ratio'])
	residual_threshold = dataset_params['sd'] * dataset_params['residual_kappa']

	_check_dim_match(data, dataset_params['noise_dim'])
	noisy_data = np.empty((n_outliers, dataset_dim), dtype = float)

	i = 0
	trials = 0

	while i < n_outliers:

		for dim in dataset_params['noise_dim']:
			noise = np.random.uniform(-dataset_params['uniform_noise_bbox'], dataset_params['uniform_noise_bbox'])
			noisy_data[i][dim] = origin[dim] + noise

		if _is_in_bbox(noisy_data[i], dataset_bbox):
			if not _is_inlier(noisy_data[i], original_model, residual_threshold):
				i += 1

		trials += 1

		if trials == dataset_params['max_trials']:
			raise RuntimeError('Number of trials has been exceeded. Gaussian noise has not been generated')
			
	#return np.stack(noisy_data, axis = 0)
	return noisy_data

def _get_bbox_from_value(value, dim):
	bbox = []
	for i in range(dim):
		bbox.append([-value, value])
	return bbox

if __name__ == '__main__':
	True