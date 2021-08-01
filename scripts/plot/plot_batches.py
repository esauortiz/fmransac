from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits import mplot3d
from scipy import stats

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import sys
import pylab

def compare(improv_worse, datasets, columns):
	""" Compares how much improvement or
		how worse is columns[0] compared to columns[1]
		
		Can handle multiple tables
		e.g. Compare how is the improvement for 2D and 3D
		hyperplanes when varying, for instance, the outlier
		ratio
	"""
	results = []
	for data in datasets:
		if improv_worse == 'improv':
			result = 1 - data.T[columns[1]]/data.T[columns[0]]
		elif improv_worse == 'worse':
			result = data.T[columns[0]]/data.T[columns[1]]
		results = [*results, result * 100]
	return results

#######################################
## USAGE: python plot_mesh.py model_class metrica Groups variable_name
## USAGE e.g.: python plot_batches.py PlaneModelND theta 52,53,54 sigma

model_class = str(sys.argv[1])
metric = str(sys.argv[2])
Groups = list(sys.argv[3].split(','))
var_name = str(sys.argv[4])

#######################################
## Some fig variables

cm = 1/2.54  # centimeters in inches
#myfigsize=(3.75*1.35*cm,1*8*cm) #vertical
#myfigsize=(1.35*5.21*cm,3.75*1.25*cm) #horizontal
myfigsize = (5.75*cm,5.75*cm) #squared medium
#myfigsize=(7.5*cm,7.5*cm) #squared big
#save_path_header = "/home/esau/tfm/memoria/CAP_5/FIG/"
save_path_header = '/home/esau/Documents/00_publicaciones/template_paper/FIG/PGF/'
plot_title = f'{model_class[0]}M_sim{var_name}_worse'

legfontsize = 7.5
save_figure = True
print_legend = True
print_legend_fig = False
figure_extension = 'pgf'

if save_figure == True:
	matplotlib.rcParams['text.usetex'] = True

	plt.rcParams.update({
	    "text.usetex": True,
	    "font.family": "serif"
	})

if var_name in ['sigma', 'kappa']:
	print_legend = False

# instance figure
fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(myfigsize))

#######################################
## Load data
estimators = ['ransac', 'msac', 'FMR2fun1', 'FMR2fun2', 'FMR2fun3', 'FMR2fun4', 'FMR4fun1', 'FMR4fun2', 'FMR4fun3', 'FMR4fun4']
estimators_label = {
	'ransac' : 'RANSAC',
	'msac' : 'MSAC',
	'FMR2fun1' : 'fmr21',
	'FMR2fun2' : 'fmr22',
	'FMR2fun3' : 'fmr23',
	'FMR2fun4' : 'fmr24',
	'FMR4fun1' : 'fmr41',
	'FMR4fun2' : 'fmr42',
	'FMR4fun3' : 'fmr43',
	'FMR4fun4' : 'fmr44'
}

estimators = ['ransac', 'msac', 'FMR2fun1', 'FMR2fun2', 'FMR2fun3', 'FMR2fun4']
estimators_label = {
	'ransac' : 'RANSAC',
	'msac' : 'MSAC',
	'FMR2fun1' : 'fmr11',
	'FMR2fun2' : 'fmr12',
	'FMR2fun3' : 'fmr21',
	'FMR2fun4' : 'fmr22'
}

data_path_header = f'/home/esau/tfm/codigo_fuente/python/MISC/plot_tables/txt_tables/{model_class}'
datasets = []
for G in Groups:
	dataset = np.loadtxt(f'{data_path_header}/G{G}_{metric}.txt')
	datasets = [*datasets, dataset]

tab20 = plt.get_cmap('Paired')(np.linspace(0,1,12))
FMR2colors = [tab20[1], tab20[3], tab20[5], tab20[7]]
FMR4colors = [tab20[0], tab20[2], tab20[4], tab20[6]]

#FMR2colors = plt.get_cmap('seismic')(np.repeat(0.25,4))
#FMR4colors = plt.get_cmap('seismic')(np.repeat(0.6,4))
"""
mycolors = ['black', 'grey', FMR2colors[0], FMR2colors[1], 
			FMR2colors[2], FMR2colors[3], FMR2colors[4], 
			FMR2colors[5], FMR2colors[6], FMR2colors[7], 
			FMR2colors[8], FMR2colors[9]]
"""
#mycolors = plt.get_cmap('hsv')(np.linspace(0,1,10))

#######################################
## Generating plot data
x_data = {
	'or' : [0.6,0.5,0.4,0.2],
	'sigma' : [2.00,1.00,0.50,0.25],
	'kappa' : [4,3,2.5,2,1]
}
x_titles = {
	'or' : 'bmomega$',
	'sigma' : 'bmsigma$',
	'kappa' : 'bmkappa$'
}
x = x_data[var_name]
x_title = x_titles[var_name]

y_titles = {
	'PlaneModelND' : 'avgtabtitledegree',
	'EllipseModel' : 'avgtabtitlerpercentage',
	'HomographyModel' : 'avgtabtitlermse'
}
y_title = y_titles[model_class]
#if var_name in ['sigma', 'or']: # kappa is placed in the right side with ylabel
#	y_title = ' '

metric_units_dic = {
	'theta' : '$^\\circ$',
	'RMSE' : '\\textbf{px}',
	'linfinity' : '\\textbf{\\%}'
}

linestyles = {
	'1' : ':',
	'2' : '-.',
	'3' : '--',
	'4' : '-'
}

# get comparison of how worse is column 1 and column 5 across all datasets
data = compare('worse', datasets, [1,5])

mycmap = plt.get_cmap('Set1')(np.linspace(0,1,9))

mycolors = mycmap[0:3]
labels = ['10D Hyperplanes', '3D Hyperplanes', '2D Hyperplanes']
for i, line, label, color in zip(range(3) ,data, labels, mycolors):
	print(line)
	ax.set_ylabel(y_title)
	ax.set_xlabel(x_title)
	ax.plot(x, line, marker="", label = label, color = color, zorder = 4-i)
	ax.yaxis.set_label_position("right")
	ax.yaxis.set_label_coords(1.3,0.50)
	ax.yaxis.tick_right()
	#ax.xaxis.set_label_position("top")
	#ax.xaxis.tick_top()

# hline at 100 in y-axis
ax.plot(x, [100,100,100,100], marker="", alpha = 0.8, color = [0.65,0.65,0.65,1], zorder = -1, linestyle = ':')

plt.xticks(x)
plt.xlim([x[-1],x[0]])
y_lim_top = np.ceil(np.max(data)/float(2.5))*float(2.5)
plt.ylim([0, 700])
#plt.xlim([min_x_value, max_x_value])

if print_legend_fig:
	figleg, axleg = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(myfigsize))
	handles,labels = ax.get_legend_handles_labels()
	figleg_leg = figleg.legend(handles, labels, borderaxespad=0, framealpha=1, fancybox = False)
	figleg_leg.get_frame().set_edgecolor('black')
	figleg_leg.get_frame().set_linewidth(0.5)
	axleg.remove()

if print_legend == True:
	#leg = plt.legend(bbox_to_anchor=(-0.06,1), loc="upper right", borderaxespad=0, framealpha=1, fancybox = False)
	leg = plt.legend(framealpha=1, fancybox = False, loc = "best", fontsize = legfontsize, handlelength = 1.5)
	leg.get_frame().set_edgecolor('black')
	leg.get_frame().set_linewidth(0.5)

if save_figure == True:
	# saving
	print(f'Saving fig in directory: {save_path_header}')
	print(f'fig_name: {plot_title}')
	print(f'extension: {figure_extension}')
	plt.grid(axis='x', color='0.85', zorder = -1)
	plt.grid(axis='y', color='0.85', zorder = -1)
	fig.savefig(f'{save_path_header}{plot_title}.{figure_extension}', bbox_inches='tight', pad_inches=0.01, dpi=300)
	#figleg.savefig(f'{save_path_header}{plot_title}_leg.{figure_extension}', bbox_inches='tight', pad_inches=0.01, dpi=300)
else:
	plt.show()

"""
python plot_batches.py PlaneModelND theta 71,74,80 or
python plot_batches.py PlaneModelND theta 72,75,81 sigma
python plot_batches.py PlaneModelND theta 73,76,82 kappa

"""