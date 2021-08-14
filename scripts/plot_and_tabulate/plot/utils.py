import matplotlib.colors as colors
from matplotlib.patches import ConnectionPatch
import matplotlib.cm as cmap

import numpy as np

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

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