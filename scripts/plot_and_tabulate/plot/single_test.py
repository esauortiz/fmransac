from matplotlib import pyplot as plt
from estimation.fit import PlaneModelND
from plot_and_tabulate.plot.utils import truncate_colormap, get_thresh_lines_dataset, plot_residuals_lines
from test_configuration.utils import _read_yaml
import matplotlib.cm as cmap
import numpy as np
import matplotlib
import sys, os

if __name__ == '__main__':

	# usage
	# python single_test.py batch_id test_id estimator figure_label

	model_class = sys.argv[1]
	batch_id = sys.argv[2]
	test_id = int(sys.argv[3][5:])
	estimator = sys.argv[4]
	if estimator == "None": estimator = None
	figure_label = sys.argv[5]

	# read batch parameters, plot parameters and naming convention
	current_path = os.path.dirname(os.path.realpath(__file__))
	scripts_path = current_path[:-22]
	tests_path = _read_yaml(f'{scripts_path}/test_configuration/params/tests_path.yaml')['path']
	batch_save_path = f'{tests_path}/{model_class}/{batch_id}'
	batch_params = _read_yaml(f'{batch_save_path}/batch_params.yaml')
	plot_params = _read_yaml(f'{current_path}/plot_params.yaml')
	nc = _read_yaml(f'{scripts_path}/plot_and_tabulate/naming_convention.yaml')

	# plot parameters
	cm = 1/2.54  # centimeters in inches
	save_path = plot_params['save_path'] + f'/{estimator}'
	save_extension = plot_params['save_extension']
	save_figure = plot_params['save_figure']
	output = plot_params['output']
	model_label = nc['text'][model_class]
	#plot_title = 'dataset_w_outliers_w_ori_model_w_residuals'
	plot_title = f'{model_label}_B{str(batch_id)[6:]}T{str(test_id)}_{str(estimator)}_{figure_label}' # e.g. PM_B1T12_RANSAC_{label}.png
	
	fig_size = plot_params['fig_size'] # [width, height] size in cm
	fig_size[0] *= cm
	fig_size[1] *= cm

	model_ori_color = plot_params['model_original_color']
	model_ori_line_width = plot_params['model_original_line_width']
	model_ori_line_style = plot_params['model_original_line_style']
	model_est_color = plot_params['model_estimated_color']
	model_est_line_width = plot_params['model_estimated_line_width']
	model_est_line_style = plot_params['model_estimated_line_style']
	model_samples = plot_params['model_samples']
	
	outlier_color = plot_params['outlier_color']
	inlier_color = plot_params['inlier_color']
	point_size = plot_params['point_size']
	force_plot_legend = plot_params['force_plot_legend']

	#plot_params = _read_yaml(f'{current_path}/plot_params_{figure_label}.yaml')
	
	plot_axis_ticks = plot_params['plot_axis_ticks']
	plot_color_map_bar = plot_params['plot_color_map_bar']
	plot_compatibilities = plot_params['plot_compatibilities']
	plot_legend = plot_params['plot_legend']
	if estimator in ['RANSAC', 'MSAC']: plot_compatibilities = False

	plot_model_params = plot_params['plot_model_params']
	plot_only_inliers = plot_params['plot_only_inliers']

	plot_mss = plot_params['plot_mss']
	plot_threshold_lines = plot_params['plot_threshold_lines']
	plot_inliers_outliers = plot_params['plot_inliers_outliers']
	plot_model_ori = plot_params['plot_model_original']
	plot_model_est = plot_params['plot_model_estimated']
	plot_residuals = plot_params['plot_residuals']
	print_info = plot_params['print_info']
	
	# using latex text format
	if save_figure == True:
		matplotlib.rcParams['text.usetex'] = True

		plt.rcParams.update({
			"text.usetex": True,
			"font.family": "sans-serif",
			#"font.sans-serif": "Arial"
		})

	# plot
	"""
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
			min_compatibility_value = 0.1
			weights[weights < min_compatibility_value] = min_compatibility_value
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
	"""
	
	if True:
	
		fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(fig_size))
		ax.axis('equal')

		# noisy dataset
		noisy_data = np.loadtxt(f'{batch_save_path}/datasets/test_{test_id}.txt')
		dataset_bbox = batch_params['dataset_params']['dataset_bbox']
		model_bbox = batch_params['dataset_params']['model_bbox']*20
		residual_threshold = batch_params['ransac_params']['residual_threshold']
		plot_limits = [-dataset_bbox, dataset_bbox]

		# read test params
		test_params = _read_yaml(f'{batch_save_path}/tests_params/test_{test_id}.yaml')
		model_ori_params = test_params['model_params']['params']

		# generate original model data
		if model_class == 'PlaneModelND':
			from dataset_generation.PlaneModelND import _get_model_dataset
		elif model_class == 'EllipseModel':
			from dataset_generation.EllipseModel import _get_model_dataset
		model_ori_dataset = _get_model_dataset(model_ori_params, model_samples, model_bbox, seed = test_id, print_model_params = True,  output = output)
							
		if estimator == 'default':
			residuals = np.loadtxt(f'{batch_save_path}/results/{estimator}/test_{test_id}_residuals.txt')
			ax.plot([], [], color=model_ori_color, alpha=1, linestyle=model_ori_line_style, linewidth=model_ori_line_width, label='$'+nc['tex_eq']['model_original']+'$')
			ax.plot([], [], color=model_est_color, alpha=1, linestyle=model_est_line_style, linewidth=model_est_line_width, label='$'+nc['tex_eq']['model_estimated']+'$')

			model_est_params = np.loadtxt(f'{batch_save_path}/results/{estimator}/test_{test_id}_params.txt')
			model_est_dataset = _get_model_dataset(model_est_params, model_samples, model_bbox, seed = test_id, print_model_params = True,  output = output)
			ax.scatter(noisy_data[:, 0], noisy_data[:, 1], point_size-1, color = 'black', alpha=1,  zorder = 0)
			model_ori_dataset = np.array([model_ori_dataset[0,:],model_ori_dataset[-1,:]])
			model_est_dataset = np.array([model_est_dataset[0,:],model_est_dataset[-1,:]])
			ax.plot(model_ori_dataset[:, 0], model_ori_dataset[:, 1], color=model_ori_color, alpha=1, linestyle=model_ori_line_style, linewidth=model_ori_line_width, zorder = -1)
			#ax.plot(model_est_dataset[:, 0], model_est_dataset[:, 1], color=model_est_color, alpha=1, linestyle=model_est_line_style, linewidth=model_est_line_width, zorder = 1)
			plot_residuals_lines(ax, model_est_params, noisy_data[:], residuals[:], 'black')
			ax.set_xlim(plot_limits)
			ax.set_ylim(plot_limits)

		elif estimator is not None:
			try:
				inliers = np.loadtxt(f'{batch_save_path}/results/{estimator}/test_{test_id}_inliers.txt').astype(bool)
				compatibilities = np.loadtxt(f'{batch_save_path}/results/{estimator}/test_{test_id}_scores.txt')
				residuals = np.loadtxt(f'{batch_save_path}/results/{estimator}/test_{test_id}_residuals.txt')
			except OSError:
				inliers = np.ones(noisy_data.shape[0], dtype = bool)
				compatibilities = np.zeros(noisy_data.shape[0], dtype = bool)
			
			model_est_params = np.loadtxt(f'{batch_save_path}/results/{estimator}/test_{test_id}_params.txt')
			outliers = inliers == False

			# generate estimated model data
			model_est_dataset = _get_model_dataset(model_est_params, model_samples, model_bbox, seed = test_id, print_model_params = True, output = output)

			if(noisy_data.shape[1] == 2):
			
				if plot_compatibilities == True:
					length = np.size(compatibilities)
					#sorting lowest weigths to be represented in "top layer"
					idx = compatibilities.argsort()
					compatibilities, noisy_data, inliers = compatibilities[idx], noisy_data[idx], inliers[idx]

					# set min score for outliers
					min_compatibility_value = 0.1
					compatibilities_to_plot = np.copy(compatibilities)
					compatibilities_to_plot[compatibilities < min_compatibility_value] = min_compatibility_value
					rgba_colors = np.zeros((length,4))
					# if weigth == 1 -> point represented with black color
					rgba_colors[:,0] = 1 - compatibilities_to_plot
					rgba_colors[:,1] = 1 - compatibilities_to_plot
					rgba_colors[:,2] = 1 - compatibilities_to_plot
					rgba_colors[:, 3] = 1

					#setting axis limits
					ax.set_xlim(plot_limits)
					ax.set_ylim(plot_limits)
					# Plotting original and estimated model first with empty data to reorder legend items
					ax.plot([], [], color=model_ori_color, alpha=1, linestyle = model_ori_line_style, linewidth=model_ori_line_width, label='$'+nc['tex_eq']['model_original']+'$')
					ax.plot([], [], color=model_est_color, alpha=1, linestyle = model_est_line_style, linewidth=model_est_line_width, label='$'+nc['tex_eq']['model_estimated']+'$')
					if force_plot_legend:
						ax.scatter([], [], point_size+1, color='lime', alpha=1, label='MSS')
						#ax.scatter([], [], point_size+1, color=inlier_color, alpha=1, label='Inlier')
						#ax.scatter([], [], point_size+1, color=outlier_color, alpha=1, label='Outlier')

					if plot_only_inliers:
						ax.scatter(noisy_data[inliers, 0], noisy_data[inliers, 1], c = rgba_colors[inliers], s = point_size)
					else:
						ax.scatter(noisy_data[:, 0], noisy_data[:, 1], c = rgba_colors, s = point_size)

					model_ori_dataset = np.array([model_ori_dataset[0,:],model_ori_dataset[-1,:]])
					model_est_dataset = np.array([model_est_dataset[0,:],model_est_dataset[-1,:]])
					if plot_model_ori: ax.plot(model_ori_dataset[:, 0], model_ori_dataset[:, 1], color=model_ori_color, alpha=1, linestyle=model_ori_line_style ,linewidth=model_ori_line_width, zorder = 0)
					if plot_model_est: ax.plot(model_est_dataset[:, 0], model_est_dataset[:, 1], color=model_est_color, alpha=1, linestyle=model_est_line_style, linewidth=model_est_line_width, zorder = 1)
					
					if plot_threshold_lines == True:
						thresh_line = get_thresh_lines_dataset(model_est_params, model_bbox, _get_model_dataset,1.15, -1)
						ax.plot(thresh_line[:, 0], thresh_line[:, 1], color=model_est_color, alpha=1, linestyle='dashed', linewidth=(model_est_line_width - 1.25))
						thresh_line = get_thresh_lines_dataset(model_est_params, model_bbox, _get_model_dataset,1.15, 1)
						ax.plot(thresh_line[:, 0], thresh_line[:, 1], color=model_est_color, alpha=1, linestyle='dashed', linewidth=(model_est_line_width - 1.25))
	
				else: 
					#setting axis limits
					ax.set_xlim(plot_limits)
					ax.set_ylim(plot_limits)
					# Plotting original and estimated model first with empty data to reorder legend items
					ax.plot([], [], color=model_ori_color, alpha=1, linestyle=model_ori_line_style, linewidth=model_ori_line_width, label='$'+nc['tex_eq']['model_original']+'$')
					ax.plot([], [], color=model_est_color, alpha=1, linestyle=model_est_line_style, linewidth=model_est_line_width, label='$'+nc['tex_eq']['model_estimated']+'$')
					ax.scatter([], [], point_size+1, color='lime', alpha=1, label='MSS')
					ax.scatter([], [], point_size+1, color=inlier_color, alpha=1, label='Inlier')
					ax.scatter([], [], point_size+1, color=outlier_color, alpha=1, label='Outlier')
					if plot_inliers_outliers == True:
						ax.scatter(noisy_data[inliers, 0], noisy_data[inliers, 1], point_size, color=inlier_color, alpha=1,  zorder=0)
						ax.scatter(noisy_data[outliers, 0], noisy_data[outliers, 1], point_size, color=outlier_color, alpha=1,  zorder=0)
					else:
						ax.scatter(noisy_data[:, 0], noisy_data[:, 1], point_size, color = 'black', alpha=1,  zorder = 0)

					if plot_mss == True:
						mss = np.loadtxt(f'/home/esau/tfm/slides/mss/{estimator}/G{test_id + 1}.txt')
						ax.scatter(mss[:, 0], mss[:, 1], 10, color = 'lime', alpha=1,  zorder = 2)

					if plot_model_ori == True:
						model_ori_dataset = np.array([model_ori_dataset[0,:],model_ori_dataset[-1,:]])
						ax.plot(model_ori_dataset[:, 0], model_ori_dataset[:, 1], color=model_ori_color, alpha=1, linestyle=model_ori_line_style ,linewidth=model_ori_line_width, zorder = -1)
					if plot_model_est == True:
						model_est_dataset = np.array([model_est_dataset[0,:],model_est_dataset[-1,:]])
						ax.plot(model_est_dataset[:, 0], model_est_dataset[:, 1], color=model_est_color, alpha=1, linestyle=model_est_line_style, linewidth=model_est_line_width, zorder = 1)

					if plot_threshold_lines == True:
						thresh_line = get_thresh_lines_dataset(model_est_params, model_bbox, _get_model_dataset,1.15, -1)
						ax.plot(thresh_line[:, 0], thresh_line[:, 1], color=model_est_color, alpha=1, linestyle='dashed', linewidth=(model_est_line_width - 1.25))
						thresh_line = get_thresh_lines_dataset(model_est_params, model_bbox, _get_model_dataset,1.15, 1)
						ax.plot(thresh_line[:, 0], thresh_line[:, 1], color=model_est_color, alpha=1, linestyle='dashed', linewidth=(model_est_line_width - 1.25))

					if plot_residuals == True:
						if plot_inliers_outliers == False:
							plot_residuals_lines(ax, model_est_params, noisy_data[:], residuals[:], 'black')
						else:
							plot_residuals_lines(ax, model_est_params, noisy_data[inliers], residuals[inliers], inlier_color)
							plot_residuals_lines(ax, model_est_params, noisy_data[outliers], residuals[outliers], outlier_color)
							

			else: 
					ax = fig.add_subplot(111, projection='3d')
					ax.scatter3D(noisy_data[inliers, 0], noisy_data[inliers, 1],  noisy_data[inliers, 2], color = inlier_color, alpha=1, label='Inlier')
					ax.scatter3D(noisy_data[outliers, 0], noisy_data[outliers, 1], noisy_data[outliers, 2], color = outlier_color, alpha=1, label='Outlier')
					#ax.scatter3D(data_original[:, 0], data_original[:, 1], data_original[:, 2], color='lime', alpha=0.8, linewidth=1, label='Modelo original')
					#ax.scatter3D(model_e_data[:, 0], model_e_data[:, 1], model_e_data[:, 2], color='blue', alpha=0.8, linewidth=1, label='Modelo estimado')
					#ax.view_init(azim=54, elev=10)
					
		# Model estimator: None -> Plot raw data
		else:
			# else only plot noisy dataset
			if(noisy_data.shape[1] == 2):
				ax.scatter(noisy_data[:, 0], noisy_data[:, 1], point_size, color = 'black', alpha=1,  zorder = 0)
				ax.set_xlim(plot_limits)
				ax.set_ylim(plot_limits)
			else:
				#print("printing in 3d")
				ax = fig.add_subplot(111, projection='3d')
				ax.scatter3D(noisy_data[:, 0], noisy_data[:, 1], noisy_data[:, 2])
		
		if plot_axis_ticks == False:
			ax.get_xaxis().set_ticks([])
			ax.get_yaxis().set_ticks([])

	if plot_compatibilities and plot_color_map_bar:
		cmap = plt.get_cmap('Greys')
		my_cmap = truncate_colormap(cmap, min_compatibility_value, 1)
		sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=plt.Normalize(vmin=min_compatibility_value, vmax=1))
		# fake up the array of the scalar mappable
		sm._A = []
		cbar = plt.colorbar(sm, ticks=[min_compatibility_value, 1], shrink=0.9, aspect=15)
		cbar.ax.set_yticklabels(['$\\phi=0$', '$\\phi=1$'])

	if plot_legend == True:
		leg = plt.legend(bbox_to_anchor=(1.06,1), loc="upper left", fontsize = 12, borderaxespad=0, framealpha=1, fancybox = False, handlelength = 1.25)
		leg.get_frame().set_edgecolor('black')
		leg.get_frame().set_linewidth(0.5)

	if save_figure == True:
		if output == 'screen': print(f'Table saved in: {save_path}')
		if output == 'screen': print(f'Tex file name: {plot_title} with {save_extension} extension PRINTED')
		if plot_model_params:
			model_est_params = PlaneModelND().get_general_params(model_est_params)
			A = "%.4f" % model_est_params[0]
			B = "%+.4f" % model_est_params[1]
			C = "%+.4f" % model_est_params[2]
			params = f'{A}x{B}y{C}=0'
			plt.text(-27.5*cm, -37.5*cm, params , fontsize=14, color = 'black')
		if print_info == True and estimator != 'default':
			if estimator == 'RANSAC': 
				info = f'$C$: {np.sum(outliers)}'
			elif estimator == 'MSAC': 
				score = "%.2f" % np.sum(compatibilities)
				info = f'$C$: {score}'
			elif estimator[3] != 4: #FMR4
				score = "%.2f" % np.sum(compatibilities)
				info = f'$\\varphi$: {score}'
			else:
				score = "%.2f" % np.sum(compatibilities)
				info = f'$\varphi$: {score}'
		
			plt.text(-6.5*cm, -37.5*cm, info, fontsize=14)
		else: 
			plt.text(-6.5*cm, -37.5*cm, " ", fontsize=14, color = 'white')
		plt.title(f'Iteración $t={test_id+1}$', fontsize = 14, color = 'black')
		plt.savefig(f'{save_path}/refinement/{plot_title}{save_extension}', bbox_inches='tight', pad_inches=0.01, dpi=300)
	else:
		plt.show()	