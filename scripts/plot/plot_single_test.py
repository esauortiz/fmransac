import matplotlib
import matplotlib.transforms as mtransforms
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
import matplotlib.cm as cmap

import numpy as np
import sys
import yaml
import io

from matplotlib.patches import ConnectionPatch
from matplotlib import pyplot as plt
from data.gen_data import plane_nd_data, ellipse_data
from estimation.myfit import PlaneModelND, EllipseModel
from estimation.estimation_results import get_homography_params, get_hplane_params, check_antiparallelism, get_ellipse_params

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

def get_bbox_limits(data, tolerance):
	# 'dim' rows and 2 columns (min, max)
	limits = np.empty((np.shape(data)[1], 2), dtype = float)
	i = 0
	for col in np.transpose(data):
		limits[i] = np.array([min(col) - tolerance, max(col) + tolerance])
		i += 1
	return limits

def get_projection(params, data):
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

model_class = 'HomographyModel' # {LineModelND, CircleModel, EllipseModel, PlaneModelND}
save_path = '/home/esau/tfm/Tests/' + model_class

model_class_dict = {
	"EllipseModel" : "EM",
	"PlaneModelND" : "PM",
	"HomographyModel" : "HM"
}
# usage
# python test_plot_single_test.py #batch #test model_e
# python test_plot_single_test.py 1 58 fun4
# python test_plot_single_test.py 1 58 None

################################################################
## PLOT SETTINGS
################################################################
cm = 1/2.54  # centimeters in inches
#small size
#myfigsize=(4*cm,4*cm)
#medium size
#myfigsize = (5.75*cm,5.75*cm)
#big size
#myfigsize=(7.5*cm,7.5*cm)
#estylf small size. x-axis should be 1.25 times wider if print_colormap == True
myfigsize=(3.75*cm,3.75*cm)

plot_wheights = False
view_axis_labels = False
#file_path = "/home/esau/tfm/memoria/CAP_1/figure_format_figs/"
#file_path = "/home/esau/Documents/00_publicaciones/On the Use of Fuzzy Metrics for Robust Model Estimation A RANSAC-based Approach/FIG/PDF/"
#file_path = "/home/esau/tfm/memoria/CAP_5/FIG/"
file_path = '/home/esau/Documents/00_publicaciones/paper_v20210507/paper_v20210507/FIG/PDF/'
#file_path = file_path + "tmp/"
#file_path = "/home/esau/Documents/estylf/llncs2e/FIG/"
plot_title = f'{model_class_dict[model_class]}_B{str(sys.argv[1])}T{str(sys.argv[2])}_{str(sys.argv[3])}'
print_legend = False
print_colormap = False
print_model_params = False
string_model_params = None
save_figure = False # If False plot will be only shown (plt.show())

################################################################
## inplot strings and some related variables
################################################################

# orignal/estimated models
str_modelo_original = "$M_{\\widehat{\Theta}}$"
str_modelo_estimado = "$M_{{\Theta}^*}$"
modelo_original_color = (0.65,0.65,0.65)
modelo_estimado_color = (0.00,0.00,0.00)
modelo_original_linestyle = '-'
modelo_estimado_linestyle = '-'

modelo_original_linewidth = 2.55
modelo_estimado_linewidth = 2.55

print_model_o_params = True
print_model_r_params = False

# inlier/outlier
str_inlier = "\\textit{Inlier}"
str_outlier = "\\textit{Outlier}"
outlier_color = (0.8,0.8,0.8)
#outlier_color = (0.0,0.0,0.0)
#inlier_color = (0.0,0.0,0.0)
outlier_color = (1,0,0)
inlier_color = (0,0,1)
point_size = 2

################################################################
## Using latex text format
################################################################
if save_figure == True:
	matplotlib.rcParams['text.usetex'] = True

	plt.rcParams.update({
	    "text.usetex": True,
	    "font.family": "serif"

	})
################################################################
## Select test to be ploted
################################################################

#test_id = str(1)
tests = [int(sys.argv[2])]
#batch = "batch_187"
batch = "batch_" + str(sys.argv[1])
model_e = str(sys.argv[3]) #{ransac, msac, fun1, fun2, fun3, fun4} or None
try:
	inliers_flag = str(sys.argv[4])
except:
	inliers_flag = "estimated"

if model_e == "None":
	model_e = None
if model_e == 'ransac' or model_e == 'msac':
	plot_wheights = False

################################################################
## Plot
################################################################

if model_class == "HomographyModel":

	test_id = str(tests[0])

	fp = np.loadtxt(save_path + '/' + batch + '/data/test_' + test_id + '_proj1.txt')
	tp = np.loadtxt(save_path + '/' + batch + '/data/test_' + test_id + '_proj2.txt')
	correspondences = np.loadtxt(save_path + '/' + batch + '/data/test_' + test_id + '_correspondences.txt')

	yaml_file = save_path + '/' + batch + '/yaml/test_' + str(test_id) + '.yaml'
	# Read test params
	with open(yaml_file, 'r') as stream:
		test_params = yaml.safe_load(stream)
	data_params = test_params['data_params']

	# original inliers when plotting original projection or original inliers/outliers
	inliers = np.loadtxt(save_path + '/' + batch + '/data/test_' + test_id + '_inliers.txt').astype(bool)
	
	if model_e is not None:
		file_name = model_e + '/test_' + test_id 
		inliers = np.loadtxt(save_path + '/' + batch + '/results/' + file_name + '_inliers.txt').astype(bool)
	
		params_e = np.loadtxt(save_path + '/' + batch + '/results/' + file_name + '_params.txt')
		params_e_vector = get_homography_params(params_e)
	
		estimated_projection = get_projection(params_e, fp)
		#add original mismatches in order tu plot outliers properly
		#this is beacause only tp is shuffled and, when generating a "new" tp (i.e. estimated projection) correspondences are not properly ordered
		estimated_projection = np.array([x for _,x in sorted(zip(correspondences[:,1],estimated_projection))])

	outliers = inliers == False	
	params_o = data_params['model_params']
	s, phi, tx, ty = get_homography_params(params_o)
	string_model_params = f's: {s} | phi: {phi} | tx: {tx} | ty: {ty}'
	#print(f'model_e: {model_e} with params: {params_e}')
	#print(f'original params: {params_o}')
	#params_o_vector = get_homography_params(params_o)
	#print(f'diff: {((params_o_vector-params_e_vector)/params_o_vector)*100}')
	#print(params_o_vector)
	

	#theres no need to add mismatches for plotting pourposes in original projection
	original_projection = get_projection(params_o, fp)
	
	fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=myfigsize)
	
	if plot_wheights == False:
		if inliers_flag == "estimated" and model_e is not None:
			# First subplot
			#ax1 = fig.add_subplot(1, 2, 1)

			# Plotting original and estimated model first with empty data to reorder legend items
			ax1.scatter(fp[outliers,0], fp[outliers,1], 2*point_size, marker = 's', color = outlier_color, alpha=1, label=str_outlier)
			ax1.scatter(fp[inliers,0], fp[inliers,1], 2*point_size, marker = 's', color = inlier_color, alpha=1, label=str_inlier)

			# Second subplot
			#ax2 = fig.add_subplot(1, 2, 2)
			#estimated_projection=tp
			ax2.scatter(original_projection[:,0], original_projection[:,1], 2*point_size, marker = 's', facecolors='none', color=modelo_original_color, alpha=1, label=str_modelo_original)
			#legend order
			ax2.scatter([], [], 2*point_size, marker = 's', color = inlier_color, alpha=1, label=str_inlier)
			ax2.scatter([], [], 2*point_size, marker = 's', color = outlier_color, alpha=1, label=str_outlier)
			#plot inliers and outliers without labels (already set)
			ax2.scatter(estimated_projection[outliers,0], estimated_projection[outliers,1], 2*point_size, marker = 's', color = outlier_color, alpha=1)
			ax2.scatter(estimated_projection[inliers,0], estimated_projection[inliers,1], 2*point_size, marker = 's', color = inlier_color, alpha=1)
			plot_matchings(ax1, ax2, fp, estimated_projection, inliers, inliers_flag)
		
		elif inliers_flag == "original":
			ax1.plot(fp[:,0], fp[:,1], 'bo', color='lime', alpha=0.8, marker = '.')
			ax2.plot(original_projection[:,0], original_projection[:,1], 'bo', color='lime', alpha=0.8, marker = '.')
			plot_matchings(ax1, ax2, fp, original_projection, inliers, inliers_flag)

		elif inliers_flag == "gauss": # original inliers with gaussian noise
			colors = cmap.Greys(np.linspace(0,1,fp.shape[0]))
			indices = np.arange(0,len(colors))

			ax1.scatter(fp[:,0], fp[:,1], 2*point_size, marker = 's', color=colors[indices], alpha=1)
			ax2.scatter(tp[:,0], tp[:,1], 2*point_size, marker = 's', color=colors[indices], alpha=1)
			ax2.scatter(original_projection[:,0], original_projection[:,1], 2.5*point_size, facecolors='none', marker = 's', color=modelo_original_color, alpha=1, label=str_modelo_original, zorder = 0)
			
			plot_matchings(ax1, ax2, fp, tp, inliers, inliers_flag, rgba_colors = colors[indices])


	elif plot_wheights == True:

		weights = np.loadtxt(save_path + '/' + batch + '/results/' + file_name + '_best_scores.txt')
		# printing w as alpha
		length = np.size(weights)
		rgba_colors = np.zeros((length,4))

		# reorder weights and data to print higher weigths at top
		idx = weights.argsort()
		weights, fp, original_projection, estimated_projection = weights[idx], fp[idx], original_projection[idx], estimated_projection[idx]

		# set min score for outliers
		min_weigth = 0.1
		weights[weights < min_weigth] = min_weigth
		rgba_colors = np.zeros((length,4))
		# if weigth == 1 -> inlier represented with black color
		rgba_colors[:,0] = (1 - weights)
		rgba_colors[:,1] = (1 - weights)
		rgba_colors[:,2] = (1 - weights)
		# alphas
		rgba_colors[:, 3] = 1

		# Plotting original and estimated model first with empty data to reorder legend items
		ax1.scatter(fp[:,0], fp[:,1], 2*point_size, marker='s', color = rgba_colors, alpha=1, label=str_inlier)

		# Second subplot
		#ax2 = fig.add_subplot(1, 2, 2)
		#estimated_projection=tp
		ax2.scatter(original_projection[:,0], original_projection[:,1], 2*point_size, facecolors='none', marker = 's', color=modelo_original_color, alpha=1, label=str_modelo_original)
		ax2.scatter(estimated_projection[:,0], estimated_projection[:,1], 2*point_size, marker = 's', color = rgba_colors, alpha=1)
		plot_matchings(ax1, ax2, fp, estimated_projection, inliers, inliers_flag, rgba_colors)

		#ax1.plot(fp[inliers,0], fp[inliers,1], 'bo', color='blue', alpha=1, marker = '.', label='Inliers')
		#ax1.plot(fp[outliers,0], fp[outliers,1], 'bo', color='red', alpha=1, marker = '.', label='Outliers')

		#ax2.plot(tp[outliers,0], tp[outliers,1], 'bo' ,color = 'red', marker = '.', alpha = 0.4, label='Outliers')
		#ax2.scatter(tp[:,0], tp[:,1], color = rgba_colors)
		#ax2.plot(original_projection[:,0], original_projection[:,1], 'bo', color='lime', marker = '.', alpha=0.8, label='Modelo\noriginal')

	if view_axis_labels == False:
		ax1.get_xaxis().set_ticks([])
		ax1.get_yaxis().set_ticks([])
		ax2.get_xaxis().set_ticks([])
		ax2.get_yaxis().set_ticks([])

else:
	for id in tests:
	
		test_id = str(id)

		# fig settings
		fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(myfigsize))
		ax.axis('equal')

		# noisy dataset
		data_noisy = np.loadtxt(save_path + '/' + batch + '/data/test_' + test_id + '.txt')
		lims = get_bbox_limits(data_noisy, tolerance = 0)
		lims[0] = [-10,10]

		# Read test params
		yaml_file = save_path + '/' + batch + '/yaml/test_' + str(test_id) + '.yaml'
		with open(yaml_file, 'r') as stream:
			test_params = yaml.safe_load(stream)

		# test params
		data_params = test_params['data_params']
		model_o_params = data_params['model_params']
		data_len = data_params['data_len']

		# generate original model data
		if model_class == 'PlaneModelND':
			data_original = plane_nd_data(*data_len, test_id, *model_o_params)

		elif model_class == 'EllipseModel':
			data_original = ellipse_data(*data_len, *model_o_params)
			
		if model_e is not None:

			# read needed files
			file_name = model_e + '/test_' + test_id 
			try:
				inliers = np.loadtxt(save_path + '/' + batch + '/results/' + file_name + '_inliers.txt').astype(bool)
			except OSError:
				inliers = np.ones(data_noisy.shape[0], dtype = bool)
			
			model_r_params = np.loadtxt(save_path + '/' + batch + '/results/' + file_name + '_params.txt')
			outliers = inliers == False

			if model_class == 'EllipseModel':
				model_e_data = ellipse_data(*data_len, *model_r_params)
				if print_model_r_params:
					a,b,c,d,f,g = get_ellipse_params(model_r_params)
					string_model_params = f'${"%.3f" % a} | {"%.3f" % b} | {"%.3f" % c} | {"%.3f" % d} | {"%.3f" % f} | {"%.3f" % g} $'
									
			elif model_class == 'PlaneModelND':
				model_e_data = plane_nd_data(*data_len, test_id, *model_r_params)
				if print_model_r_params:
					Ao, Bo, Co = get_hplane_params(model_r_params)
					string_model_params = f'${"%.2f" % Ao}x{"%.2f" % Bo}y+{"%.2f" % Co}=0$'

				#if check_antiparallelism(np.array([Ao,Bo]), np.array([A,B])) == True:
				#	A = -A
				#	B = -B

			# 2D LINE
			if(data_noisy.shape[1] == 2):
			
				if plot_wheights == True:
					weights = np.loadtxt(save_path + '/' + batch + '/results/' + file_name + '_best_scores.txt')
					length = np.size(weights)
					#sorting lowest weigths to be represented in "top layer"
					idx = weights.argsort()
					weights, data_noisy = weights[idx], data_noisy[idx]

					# set min score for outliers
					min_weigth = 0.1
					weights[weights < min_weigth] = min_weigth
					rgba_colors = np.zeros((length,4))
					# if weigth == 1 -> inlier represented with black color
					rgba_colors[:,0] = 1 - weights
					rgba_colors[:,1] = 1 - weights
					rgba_colors[:,2] = 1 - weights
					# alphas
					rgba_colors[:, 3] = 1

					#setting axis limits
					ax.set_xlim(lims[0])
					ax.set_ylim(lims[1])
					# Plotting original and estimated model first with empty data to reorder legend items
					ax.plot([], [], color=modelo_original_color, alpha=1, linestyle = modelo_original_linestyle ,linewidth=modelo_original_linewidth, label=str_modelo_original)
					ax.plot([], [], color=modelo_estimado_color, alpha=1, linestyle = modelo_estimado_linestyle, linewidth=modelo_estimado_linewidth, label=str_modelo_estimado)
					ax.scatter(data_noisy[:, 0], data_noisy[:, 1], c = rgba_colors, s = point_size)
					ax.scatter([], [], point_size, color = inlier_color, alpha=1, label=str_inlier)
					ax.scatter([], [], point_size, color = outlier_color, alpha=1, label=str_outlier)
					ax.plot(data_original[:, 0], data_original[:, 1], color=modelo_original_color, alpha=1, linestyle = modelo_original_linestyle ,linewidth=modelo_original_linewidth)
					ax.plot(model_e_data[:, 0], model_e_data[:, 1], color=modelo_estimado_color, alpha=1, linestyle = modelo_estimado_linestyle, linewidth=modelo_estimado_linewidth)

				else: 
					#setting axis limits
					#ax.set_xlim(lims[0]+(20,25))
					#ax.set_ylim(lims[1]-(5,-7.5))
					ax.set_xlim(lims[0])
					ax.set_ylim(lims[1])
					# Plotting original and estimated model first with empty data to reorder legend items
					ax.plot([], [], color=modelo_original_color, alpha=1, linestyle = modelo_original_linestyle ,linewidth=modelo_original_linewidth, label=str_modelo_original)
					ax.plot([], [], color=modelo_estimado_color, alpha=1, linestyle = modelo_estimado_linestyle, linewidth=modelo_estimado_linewidth, label=str_modelo_estimado)
					ax.scatter(data_noisy[inliers, 0], data_noisy[inliers, 1], point_size, color = inlier_color, alpha=1,  zorder = 0)
					ax.scatter(data_noisy[outliers, 0], data_noisy[outliers, 1], point_size, color = outlier_color, alpha=1,  zorder = 0)
					ax.plot(data_original[:, 0], data_original[:, 1], color=modelo_original_color, alpha=1, linestyle = modelo_original_linestyle ,linewidth=modelo_original_linewidth, zorder = 1)
					ax.plot(model_e_data[:, 0], model_e_data[:, 1], color=modelo_estimado_color, alpha=1, linestyle = modelo_estimado_linestyle, linewidth=modelo_estimado_linewidth)

			# 3D LINE
			else: 
					ax = fig.add_subplot(111, projection='3d')
					#plt3d = plt.figure().gca(projection='3d')
					#ax = plt.gca()
					
					def plt_surface(params, ax, color, label):

						params = params.reshape(2,-1)
						point, normal = params
						d = -point.dot(normal)
						# create x,y
						xx, yy = np.meshgrid(range(-10,10), range(-10,10))
						# calculate corresponding z
						z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]
						# Ensure that the next plot doesn't overwrite the first plot
						ax.scatter(xx, yy, z, alpha=0.5, color = color, label = label)
						#plot_wireframe or plot_surface
						return True

					#plt_surface(np.asarray(model_r_params), ax, modelo_estimado_color, 'Modelo estimado')
					#normal_vector = np.array([0,0,0,*model_r_params[0]])
					#ax.quiver(*normal_vector)
					#plt_surface(np.asarray(model_o_params), ax, modelo_original_color, 'Modelo original')
					#ax.hold(True)					
					ax.scatter3D(data_noisy[inliers, 0], data_noisy[inliers, 1],  data_noisy[inliers, 2], color = inlier_color, alpha=1, label=str_inlier)
					ax.scatter3D(data_noisy[outliers, 0], data_noisy[outliers, 1], data_noisy[outliers, 2], color = outlier_color, alpha=1, label=str_outlier)
					#ax.scatter3D(data_original[:, 0], data_original[:, 1], data_original[:, 2], color='lime', alpha=0.8, linewidth=1, label='Modelo original')
					#ax.scatter3D(model_e_data[:, 0], model_e_data[:, 1], model_e_data[:, 2], color='blue', alpha=0.8, linewidth=1, label='Modelo estimado')

					#ax.view_init(azim=54, elev=10)
					#ax.set_xlim([-2,2])
					#ax.set_ylim([-2,2])
					#ax.set_zlim([-2,2])

		# Model estimator: None -> Plot raw data
		else:
			# else only plot noisy dataset
			if(data_noisy.shape[1] == 2):
				#original_inliers = np.loadtxt(save_path + '/' + batch + '/data/test_' + test_id + '_inliers.txt').astype(bool)
				#outliers = original_inliers == False
				# print(get_ellipse_params(model_o_params))
				#ax.plot(data_noisy[original_inliers, 0], data_noisy[original_inliers, 1], '.b', alpha=0.25, label='Inlier data')
				ax.scatter(data_noisy[:, 0], data_noisy[:, 1], point_size, color = 'black', alpha=1,  zorder = 0)
				#ax.plot(data_original[:, 0], data_original[:, 1], color='lime', alpha=0.8, linewidth=2., label='Original Model')
				#ax.plot(data_noisy[outliers, 0], data_noisy[outliers, 1], '.r', alpha=0.6, label='Outlier data')

				#ax.legend(loc='lower right')
				
				ax.set_xlim(lims[0])
				ax.set_ylim(lims[1])
			else:
				#print("printing in 3d")
				ax = fig.add_subplot(111, projection='3d')
				ax.scatter3D(data_noisy[:, 0], data_noisy[:, 1], data_noisy[:, 2])
		
		if view_axis_labels == False and False:
			ax.get_xaxis().set_ticks([])
			ax.get_yaxis().set_ticks([])

if plot_wheights and print_colormap:
	cmap = plt.get_cmap('Greys')
	my_cmap = truncate_colormap(cmap, min_weigth, 1)
	sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=plt.Normalize(vmin=min_weigth, vmax=1))
	# fake up the array of the scalar mappable
	sm._A = []
	cbar = plt.colorbar(sm, ticks=[min_weigth, 1], shrink=0.9, aspect=15)
	cbar.ax.set_yticklabels(['$\\phi=0$', '$\\phi=1$'])

if print_legend == True:
	leg = plt.legend(bbox_to_anchor=(1.06,1), loc="upper left", borderaxespad=0, framealpha=1, fancybox = False)
	leg.get_frame().set_edgecolor('black')
	leg.get_frame().set_linewidth(0.5)

if print_model_params == True:
	ax.text(0.5, 0.90, '$M(\\hat{\\theta})=$'+string_model_params, fontsize=4,
        bbox={'facecolor': (0.9,0.9,0.9), 'alpha': 0.9, 'pad': 2, 'linewidth':0.5}, 
        horizontalalignment='center',
        verticalalignment='center', 
        transform=ax.transAxes)

if save_figure == True:
	print(f'Table saved in: {file_path}')
	print(f'Tex file name: {plot_title} with .pgf extension PRINTED')
	if print_model_r_params or print_model_o_params:
		if model_class == 'PlaneModelND':
			if print_model_o_params:
				Ao, Bo, Co = get_hplane_params(model_o_params)
				string_model_params = f'${"%.2f" % Ao}x{"%.2f" % Bo}y+{"%.2f" % Co}=0$'
		if model_class == 'EllipseModel':
			if print_model_o_params:	
				a,b,c,d,f,g = get_ellipse_params(model_o_params)
				#string_model_params = f'${"%.3f" % a} | {"%.3f" % b} | {"%.3f" % c} | {"%.3f" % d} | {"%.3f" % f} | {"%.3f" % g} $'
				string_model_params = f'{"%.3f" % a}*x^2 + 2*{"%.3f" % b}*x*y + {"%.3f" % c}*y^2 + 2*{"%.3f" % d}*x + 2*{"%.3f" % f}*y + {"%.3f" % g} = 0'
	print('$M(\\hat{\\theta})=$'+string_model_params)
	plt.savefig(f'{file_path}{plot_title}.pdf', bbox_inches='tight', pad_inches=0.01, dpi=300)
else:
	plt.show()