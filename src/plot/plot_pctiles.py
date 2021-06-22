from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits import mplot3d
from scipy import stats

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import sys

def get_percentileofscores(data, scores):
	y = []
	for score in scores:
		percentile = stats.percentileofscore(data, score)
		y = [*y, percentile]
	return np.asarray(y)

def get_dataset(data_path_header, start_batch, estimators, metric):
	dataset = []
	i = 0
	for estimator in estimators:
		data = np.loadtxt(f'{data_path_header}/batch_{start_batch + i}/results/{estimator}/00_{metric}_values.txt')
		#print(f'{data_path_header}/batch_{start_batch + i}/results/{estimator}/00_{metric}_values.txt')
		#i += 1
		dataset = [*dataset, data]
	return np.array(dataset)

#######################################
## USAGE: python plot_mesh.py model_class metrica start_batch1 fmransac_version
## USAGE e.g.: python plot_pctiles.py PlaneModelND theta 18 2

#######################################
## Some fig variables

cm = 1/2.54  # centimeters in inches
#myfigsize=(3.75*1.35*cm,1*8*cm) #vertical
#myfigsize=(1.35*5.21*cm,3.75*1.25*cm) #horizontal
myfigsize = (5.75*cm,5.75*cm) #squared medium
#save_path_header = "/home/esau/tfm/memoria/CAP_5/FIG/"
save_path_header = '/home/esau/Documents/00_publicaciones/template_paper/FIG/PGF/'
dims = str(sys.argv[5])
plot_title = f'PM{dims}_s1_or40_pctiles'
legfontsize = 7.5

save_figure = True
figure_extension = 'pgf'

if save_figure == True:
	matplotlib.rcParams['text.usetex'] = True

	plt.rcParams.update({
	    "text.usetex": True,
	    "font.family": "serif"
	})

# instance figure
fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(myfigsize))

#######################################
## Load data
estimators = ['ransac', 'msac', 'fun1', 'fun2', 'fun3', 'fun4']
estimators_label = {
	'ransac' : 'RANSAC',
	'msac' : 'MSAC',
	'fun1' : '$M^{1,1}$',
	'fun2' : '$M^{1,2}$',
	'fun3' : '$M^{2,1}$',
	'fun4' : '$M^{2,2}$'
}

estimators_to_plot = ['ransac', 'msac', 'fun4']

model_class = str(sys.argv[1])
metric = str(sys.argv[2])
start_batch_1 = int(sys.argv[3])
rv = int(sys.argv[4])

data_path_header = f'/home/esau/tfm/Tests/{model_class}'
dataset = get_dataset(data_path_header, start_batch_1, estimators, metric)

tab20 = plt.get_cmap('Paired')(np.linspace(0,1,12))
FMR2colors = [tab20[1], tab20[3], tab20[5], tab20[7]]
FMR4colors = [tab20[0], tab20[2], tab20[4], tab20[6]]

FMR2colors = [tab20[1], tab20[7], tab20[0], tab20[6]]
FMR4colors = [tab20[0], tab20[2], tab20[4], tab20[6]]

if rv == 2:
	colors = ['black', [0.65,0.65,0.65,1], *FMR2colors]
if rv == 4:
	colors = ['black', [0.65,0.65,0.65,1], *FMR4colors]

#######################################
## Generating plot data
markers = ['X', '*', 's', 'o', '^', '+']
markers = ['', '', '', '', '', '']

min_x_value = 0.5
max_x_value = 0.95
x = np.linspace(min_x_value,max_x_value,10)
x_ticks = np.linspace(0.5,0.9,5)

metric_units_dic = {
	'theta' : '$^\\circ$',
	'RMSE' : '\\textbf{px}',
	'linfinity' : ''
}

for data, estimator, marker, color in zip(dataset, estimators, markers, colors):
	if estimator not in estimators_to_plot:
		continue
	if estimator in ['ransac', 'msac']:
		linestyle = '-'
	else:
		linestyle = '--'
	n_data = data.shape[0]
	y = np.percentile(data, x*100) # y \in [0, 1] as probability
	ax.set_ylabel('\\textbf{Error (}' + metric_units_dic[metric] + '\\textbf{)}')
	ax.set_xlabel('\\textbf{Probability}')
	#plt.title('CDF using sorting the data')
	ax.plot(x, y, marker=marker, label = estimators_label[estimator], color = color, linestyle = linestyle)
	ax.yaxis.set_label_position("right")
	ax.yaxis.set_label_coords(1.3,0.50)
	ax.yaxis.tick_right()
	#ax.xaxis.set_label_position("top")
	#ax.xaxis.tick_top()

plt.xticks(x_ticks)
plt.xlim([min_x_value, max_x_value])
plt.ylim((0,13))

#leg = plt.legend(framealpha=1, fancybox = False, loc = "upper left", fontsize = legfontsize)
#leg = plt.legend(bbox_to_anchor=(-0.06,1), loc="upper right", borderaxespad=0, framealpha=1, fancybox = False)
#leg.get_frame().set_edgecolor('black')
#leg.get_frame().set_linewidth(0.5)

if save_figure == True:
	# saving
	print(f'Saving fig in directory: {save_path_header}')
	print(f'fig_name: {plot_title}')
	print(f'extension: {figure_extension}')
	plt.grid(axis='x', color='0.85', zorder = -1)
	plt.grid(axis='y', color='0.85', zorder = -1)

	plt.savefig(f'{save_path_header}{plot_title}.{figure_extension}', bbox_inches='tight', pad_inches=0.01, dpi=300)
else:
	plt.show()


"""
python plot_pctiles.py PlaneModelND theta 401 2 2d
python plot_pctiles.py PlaneModelND theta 414 2 3d
python plot_pctiles.py PlaneModelND theta 440 2 10d
"""