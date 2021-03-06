from matplotlib.patches import ConnectionPatch
import matplotlib.cm as cmap
import matplotlib.colors as colors
import numpy as np

# single_test.py

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def get_thresh_lines_dataset(model_params, model_bbox, _get_model_dataset, threshold, sign):
	""" returns 2D threshold lines (two 2D points) at both sides
		of the estimated model with model_params parameters	
	Parameters
	----------
	model_params : (2, 2) array
		set of [origin, normal_vector] parameters
	model_bbox : int
		Half the length of the side of the square 
		containing the dataset
	_get_model_dataset : function
		returns model dataset i.e. a 2D line following 
		model_params parameters
	threshold : float
		residual threshold
	sign : int
		determines the side at which the threshold line is
	Returns
	-------
	residuals : (2, 2) array
		2D threshold lines (two 2D points) at both sides
		of the estimated model with model_params parameters	
	"""
	origin, normal_vector = np.copy(model_params)
	normal_vector /= np.linalg.norm(normal_vector)
	origin += normal_vector * threshold * sign
	data = _get_model_dataset([origin, normal_vector], 10000, model_bbox, seed = 0, output = None)
	return np.array((data[1,:], data[-1,:]))

def plot_residuals_lines(ax, model_params, dataset, residuals, color):
	""" plots 2D lines between each dataset's point and the model
	Parameters
	----------
	ax :
		ax where to plot
	model_params : (2, 2) array
		set of [origin, normal_vector] parameters
	dataset : (N, 2) array
		noisy dataset
	residuals : (N, 2) array
		fitting errors
	color : str or rgb as (3,) array
		lines color
	Returns
	-------
	"""
	origin, normal_vector = model_params
	normal_vector /= np.linalg.norm(normal_vector)
	for point, residual in zip(dataset, residuals):
		model_point = point - residual * normal_vector
		residual_line = np.array((point, model_point))
		ax.plot(residual_line[:, 0], residual_line[:, 1], color=color, alpha=1, linestyle='-', linewidth=0.75, zorder = -1)

def plot_matchings(ax1, ax2, fp, tp, inliers, inliers_flag, rgba_colors = None):

	if rgba_colors is not None:
		for P, Q, color in zip(fp, tp, rgba_colors):
			con= ConnectionPatch(xyA=Q, xyB=P, coordsA="data", coordsB="data", axesA=ax2, axesB=ax1, arrowstyle="-", color=color, alpha=1, linewidth=0.5)
			ax2.add_artist(con)
	else:
		for P, Q, is_inlier in zip(fp, tp, inliers):
			outlier_color = (1,0.5,0.5)
			inlier_color = (0,0,1)
	
			modelo_original_color = (0.65,0.65,0.65)

			if inliers_flag == "original":
				color = modelo_original_color
			elif is_inlier:
				continue
			else:
				color = outlier_color
			con= ConnectionPatch(xyA=Q, xyB=P, coordsA="data", coordsB="data", axesA=ax2, axesB=ax1, arrowstyle="-", color=color, alpha=1, linewidth=0.5)
			ax2.add_artist(con)

		for P, Q, is_inlier in zip(fp, tp, inliers):
			inlier_color = (0,0,1)
			if is_inlier:
				con= ConnectionPatch(xyA=Q, xyB=P, coordsA="data", coordsB="data", axesA=ax2, axesB=ax1, arrowstyle="-", color=inlier_color, alpha=1, linewidth=0.5)
				ax2.add_artist(con)

	return True

def get_projection(params, data, inliers):
	s_cos_phi, s_sin_phi, tx, ty = params
	H = np.array([[s_cos_phi, -s_sin_phi, tx],
	[s_sin_phi, s_cos_phi,  ty],
	[0,         0,          1]])
	ones = np.ones((np.size(inliers),3))
	ones[:,:-1]=data
	estimated_projection = np.dot(H, ones.T).T
	estimated_projection[:,0] /= estimated_projection[:,2]
	estimated_projection[:,1] /= estimated_projection[:,2]
	return estimated_projection[:,:-1]

# percentiles.py
def get_results_data(estimators, metric, results_path):
	""" returns estimator's tests results for a metric
	Parameters
	----------
	estimators : (N, ) list
		list of estimators
	metric : str
		metric in {'abs_errors', 'rel_errors', 'estimation_errors'}
	results_path : str
		folder containing results
	Returns
	-------
	results_data : (N, M) array
	"""
	results_data = []
	for estimator in estimators:
		data = np.loadtxt(f'{results_path}/{estimator}/00_{metric}.txt')
		results_data.append(data)
	return np.array(results_data)