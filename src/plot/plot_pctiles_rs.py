from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits import mplot3d
from scipy import stats

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import sys

def get_percentileofscores(data, scores):
	y = []
	for score in scores:
		percentile = stats.percentileofscore(data, score)
		y = [*y, percentile]
	return np.asarray(y)

def get_dataset(rs_model, distance, n_datasets, estimator):
	dataset = []
	for dataset_id in range(n_datasets):
		data = np.loadtxt(f'/home/esau/rs_pointclouds/rs_{rs_model}/{distance}/{estimator}/test_{dataset_id + 1}_residuals.txt')
		dataset = [*dataset, data]
	return np.array(dataset)

#######################################
## USAGE e.g.: python plot_pctiles.py rs_model distance n_datasets estimator

rs_model = str(sys.argv[1])
distance = str(sys.argv[2])
n_datasets = int(sys.argv[3])
estimator = str(sys.argv[4])

#######################################
## Some fig variables

cm = 1/2.54  # centimeters in inches
myfigsize = (7.75*cm,7.75*cm) #squared medium
save_path_header = '/home/esau/Documents/00_publicaciones/template_paper/FIG/rs_pointcloud/'
plot_title = f'rs_{rs_model}_{distance}_{estimator}'
legfontsize = 7.5

save_figure = True
figure_extension = 'pdf'

if save_figure == True:
	matplotlib.rcParams['text.usetex'] = True

	plt.rcParams.update({
	    "text.usetex": True,
	    "font.family": "serif"
	})

# instance figure
fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(myfigsize))

# read data
dataset = get_dataset(rs_model, distance, n_datasets, estimator)

min_x_value = 0.0
max_x_value = 1.0
x = np.linspace(-5,5,100)
x_ticks = x

title_dic = {
	"from_3_0m" : "at 3.0 m",
	"from_1_5m" : "at 1.5 m"
}

np.random.seed(10)
for data in dataset:
	index = np.random.choice(data.shape[0], 4000, replace=False)  
	data = data[index]*100
	ax.set_title(f'\\textbf{{{rs_model.capitalize()} {title_dic[distance]}}}')
	ax.set_ylabel('Density')
	ax.set_xlabel('Residuals (cm)')
	#plt.title('CDF using sorting the data')
	#sns.kdeplot(data, common_norm = True, color = [0,0,1,0.5])
	sigma = np.std(data)
	ax.vlines(2*sigma, ymin = 0, ymax = 1.0, color = [1,0.2,0.2,1], zorder =100)
	ax.vlines(-2*sigma, ymin = 0, ymax = 1.0, color = [1,0.2,0.2,1], zorder =100)
	s = pd.Series(data)
	ax = s.plot.kde(color = 'blue', alpha = .5)

	ax.yaxis.set_label_position("right")
	ax.yaxis.tick_right()
	#ax.xaxis.set_label_position("top")
	#ax.xaxis.tick_top()

#plt.xticks(x_ticks)
if distance == "from_3_0m":
	plt.ylim([0, 0.25])
	plt.xlim((-30,30))
else:
	plt.ylim([0, 1.0])
	plt.xlim((-5,5))

#leg = plt.legend(framealpha=1, fancybox = False, loc = "best", fontsize = legfontsize)
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
python plot_pctiles_rs.py d455 from_1_5m 50 fun1
python plot_pctiles_rs.py l515 from_1_5m 50 fun1
python plot_pctiles_rs.py d435i from_1_5m 50 fun1
python plot_pctiles_rs.py d455 from_3_0m 50 fun2
python plot_pctiles_rs.py l515 from_3_0m 50 fun2
python plot_pctiles_rs.py d435i from_3_0m 50 fun2
"""

