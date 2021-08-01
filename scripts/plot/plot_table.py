from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits import mplot3d
from scipy import stats
from data.myfit import PlaneModelND

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import sys
import pylab

#######################################
## USAGE: python plot_mesh.py model_class metrica Group variable_name
## USAGE e.g.: python plot_table.py PlaneModelND theta 52 sigma

model_class = str(sys.argv[1])
metric = str(sys.argv[2])
G = int(sys.argv[3])
var_name = str(sys.argv[4])
dims = str(sys.argv[5])

#######################################
## Some fig variables

cm = 1/2.54  # centimeters in inches
#myfigsize=(3.75*1.35*cm,1*8*cm) #vertical
#myfigsize=(1.35*5.21*cm,3.75*1.25*cm) #horizontal
myfigsize = (5.75*cm,5.75*cm) #squared medium
#myfigsize=(7.5*cm,7.5*cm) #squared big
save_path_header = "/home/esau/tfm/memoria/CAP_5/FIG/"
save_path_header = '/home/esau/Documents/00_publicaciones/template_paper/FIG/PGF/'
plot_title = f'{model_class[0]}M{dims}_sim{var_name}'

legfontsize = 7.5
save_figure = False
print_legend = True
print_legend_fig = False #una figura unicamente con leyenda, sin "axes"
figure_extension = 'pgf'

if save_figure == True:
	matplotlib.rcParams['text.usetex'] = True

	plt.rcParams.update({
	    "text.usetex": True,
	    "font.family": "serif"
	})

if var_name in ['or', 'kappa']:
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
	'FMR2fun1' : '$M^{1,1}$',
	'FMR2fun2' : '$M^{1,2}$',
	'FMR2fun3' : '$M^{2,1}$',
	'FMR2fun4' : '$M^{2,2}$'
}

data_path_header = f'/home/esau/tfm/codigo_fuente/python/MISC/plot_tables/txt_tables/{model_class}'
dataset = np.loadtxt(f'{data_path_header}/G{G}_{metric}.txt')
tab20 = plt.get_cmap('Paired')(np.linspace(0,1,12))
FMR2colors = [tab20[1], tab20[3], tab20[5], tab20[7]]
FMR4colors = [tab20[0], tab20[2], tab20[4], tab20[6]]

FMR2colors = [tab20[1], tab20[7], tab20[0], tab20[6]]
FMR4colors = [tab20[0], tab20[2], tab20[4], tab20[6]]

#FMR2colors = plt.get_cmap('seismic')(np.repeat(0.25,4))
#FMR4colors = plt.get_cmap('seismic')(np.repeat(0.6,4))
"""
mycolors = ['black', 'grey', FMR2colors[0], FMR2colors[1], 
			FMR2colors[2], FMR2colors[3], FMR2colors[4], 
			FMR2colors[5], FMR2colors[6], FMR2colors[7], 
			FMR2colors[8], FMR2colors[9]]
"""
mycolors = ['black', [0.65,0.65,0.65,1], *FMR2colors, *FMR4colors]
#mycolors = plt.get_cmap('hsv')(np.linspace(0,1,10))

#######################################
## Generating plot data
x_data = {
	'or' : [0.6,0.5,0.4,0.2],
	'sigma' : [2.00,1.00,0.50,0.25],
	'kappa' : [4,3,2.5,2,1]
}
x_titles = {
	'or' : 'bmomega',
	'sigma' : 'bmsigma',
	'kappa' : 'bmkappa'
}
x = x_data[var_name]
x_title = x_titles[var_name]

y_titles = {
	'PlaneModelND' : 'avgtabtitledegree',
	'EllipseModel' : 'avgtabtitlerpercentage',
	'HomographyModel' : 'avgtabtitlermse'
}
y_title = y_titles[model_class]
if var_name in ['sigma', 'or']: # kappa is placed in the right side with ylabel
	y_title = ' '

metric_units_dic = {
	'theta' : '$^\\circ$',
	'RMSE' : '\\textbf{px}',
	'linfinity' : '\\textbf{\\%}'
}

y_lim_top_dic = {
	'or' : 13 ,
	'sigma' : 27.5,
	'kappa' : 13,
}

for c, estimator, color in zip(dataset.T, estimators, mycolors):
	ax.set_ylabel(y_title)
	ax.set_xlabel(x_title)
	#plt.title('CDF using sorting the data')
	if estimator in ['ransac', 'msac']:
		#linestyle = linestyles[estimator[7]]
		linestyle = '-'
		zorder = 0
	else:
		linestyle = '-'
		zorder = 1
		if estimator[3] == '4' or estimator in ['FMR2fun3', 'FMR2fun4']:
			linestyle = '--'
			zorder = 2
	ax.plot(x, c, marker="", label = estimators_label[estimator], linestyle = linestyle, color = color, zorder = zorder)
	ax.yaxis.set_label_position("right")
	ax.yaxis.set_label_coords(1.3,0.50)
	ax.yaxis.tick_right()
	#ax.xaxis.set_label_position("top")
	#ax.xaxis.tick_top()

plt.xticks(x)
plt.xlim([x[-1],x[0]])
plt.ylim([0, y_lim_top_dic[var_name]])
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
	leg = plt.legend(framealpha=1, fancybox = False, loc = "upper left", fontsize = legfontsize, handlelength = 1.5)
	leg.get_frame().set_edgecolor('black')
	leg.get_frame().set_linewidth(0.5)

if save_figure == True:
	# saving
	print(f'Saving fig in directory: {save_path_header}')
	print(f'fig_name: {plot_title}')
	print(f'extension: {figure_extension}')
	fig.savefig(f'{save_path_header}{plot_title}.{figure_extension}', bbox_inches='tight', pad_inches=0.01, dpi=300)
	#figleg.savefig(f'{save_path_header}{plot_title}_leg.{figure_extension}', bbox_inches='tight', pad_inches=0.01, dpi=300)
else:
	plt.show()

"""
python plot_table.py PlaneModelND theta 71 or 2d
python plot_table.py PlaneModelND theta 72 sigma 2d
python plot_table.py PlaneModelND theta 73 kappa 2d
python plot_table.py PlaneModelND theta 74 or 3d
python plot_table.py PlaneModelND theta 75 sigma 3d
python plot_table.py PlaneModelND theta 76 kappa 3d
python plot_table.py PlaneModelND theta 80 or 10d
python plot_table.py PlaneModelND theta 81 sigma 10d
python plot_table.py PlaneModelND theta 82 kappa 10d
"""
