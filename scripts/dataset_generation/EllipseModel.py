from dataset_generation.utils import _is_inlier, _is_in_bbox, _check_dim_match, _get_bbox_from_value
from estimation.fit import EllipseModel
import numpy as np
import random

def _gaussian_noise(data, original_model, dataset_params, seed = None):
	"""Adds gaussian distributed noise
	Parameters
	----------
	Returns
	-------
	noisy data : (N, dim) array
	"""
	np.random.seed(seed)
	random.seed(seed)

	dataset_dim = 2
	dataset_bbox = _get_bbox_from_value(dataset_params['dataset_bbox'], dataset_dim)
	n_points = int(dataset_params['n_points'])
	n_inliers = int(n_points * (1 - dataset_params['outlier_ratio']))
	residual_threshold = dataset_params['sd'] * dataset_params['residual_kappa']

	_check_dim_match(data, dataset_params['noise_dim'])
	noisy_data = np.empty((n_inliers, dataset_dim), dtype = float)

	i = 0
	trials = 0

	while i < n_inliers:
		
		noisy_data[i] = np.array(random.choice(data))
		gaussian_sample = np.random.normal(0, dataset_params['sd'])

		x, y = noisy_data[i]
		a, b, c, d, f, g = original_model.get_general_params()
		
		# gradient(general_ellipse_equation)
		normal_vector = np.array([2*(a*x+b*y+d), 2*(b*x+c*y+f)])
		
		noisy_data[i] += (normal_vector * gaussian_sample)

		if _is_in_bbox(noisy_data[i], dataset_bbox):
			if _is_inlier(noisy_data[i], original_model, residual_threshold):
				i += 1
		
		trials += 1

		if trials == dataset_params['max_trials']:
			raise RuntimeError('Number of trials has been exceeded. Gaussian noise has not been generated')
			
	#return np.stack(noisy_data, axis = 0)
	return noisy_data

def _get_model_dataset(model_params, model_samples, model_bbox, seed = None, print_model_params = False):
	model = EllipseModel(model_params)
	model_bbox = _get_bbox_from_value(model_bbox, dim = 2)
	if print_model_params == True:
		a, b, c, d, f, g = model.get_general_params(model_params)
		print(f'Original model params: {"%.3f" % a} | {"%.3f" % b} | {"%.3f" % c} | {"%.3f" % d} | {"%.3f" % f} | {"%.3f" % g}')
	return model.predict_xy(t = np.linspace(0, 2 * np.pi, model_samples), model_bbox = model_bbox)

if __name__ == '__main__':
	True