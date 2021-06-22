import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib
import numpy as np
import pandas as pd
import sys

# Usage
# sys argv -> 1: puntuacion \\ 0: angulo entre vectores original (inicio etapa iterativa) y estimado (final etapa iterativa) 
# e.g. pyhton plt_convergence.py 1 

"""
shell 

python plt_convergence.py 0 1 0 1
python plt_convergence.py 0 1 0 2
python plt_convergence.py 0 1 0 3
python plt_convergence.py 0 1 0 4

python plt_convergence.py 0 1 1 1
python plt_convergence.py 0 1 1 2
python plt_convergence.py 0 1 1 3
python plt_convergence.py 0 1 1 4

python plt_convergence.py 0 1 2 1
python plt_convergence.py 0 1 2 2
python plt_convergence.py 0 1 2 3
python plt_convergence.py 0 1 2 4
"""
################################################################
## Using latex text format
################################################################
matplotlib.rcParams['text.usetex'] = True

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"

})

################################################################
## PLOT SETTINGS
################################################################
cm = 1/2.54  # centimeters in inches
myfigsize=(4.5*cm,4.5*cm)
file_path = '/home/esau/tfm/Tests/convergence_results/estylf_s2_or40/v'

#plt.style.use('grayscale')
model_estimators = {"fun1", "fun2", "fun3", "fun4"}
FRANSAC_versions = [int(sys.argv[3])]
phi_idx = int(sys.argv[4])
title = f'v{str(FRANSAC_versions[0] + 2)}_phi{str(phi_idx)}_s2_or40'

#title = "epsilon_convergence"
FRANSAC_versions = [0]
# cmapping
cmap = plt.get_cmap('Greys')
mycolors = cmap(np.linspace(0.1, 1.5, 40))

#fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=myfigsize)
#axes[0].set_ylabel('$\\Delta\\varepsilon(^\\circ)$')

tests_num = 250
myprops = [None] * tests_num

for v in FRANSAC_versions:
	df0 = pd.DataFrame({'$\\phi_1$' : []})
	for model_e in model_estimators:
		diff = []
		for i in range(tests_num):
			#reading data
			file = file_path + str(v + 2) + '/' + model_e + '_score/txt_' + str(i + 1) + '.txt'
			data = np.loadtxt(file)

			#selecting theta
			diff_i = data[0]
			diff = np.append(diff, diff_i)
			myprops[i] = mycolors[int(data[2])]
		# first column in df, instantiate df
		if model_e == "fun1":
			df0['$\\phi_1$'] = diff
		# rest of the columns of df 
		elif model_e == "fun2":
			df0['$\\phi_2$'] = diff
		elif model_e == "fun3":
			df0['$\\phi_3$'] = diff
		elif model_e == "fun4":
			df0['$\\phi_4$'] = diff

	# sort columns by name and plot
	#df0 = df0.reindex(sorted(df0.columns), axis=1)
	#df0.boxplot(ax = axes[v], grid=False, sym='.', medianprops=dict(color=mycolors[8]), boxprops=dict(color=mycolors[39]), whiskerprops=dict(color=mycolors[39]))
	#axes[v].set_ylim((5,-20))
	#axes[v].set_title(f'F-RANSAC{v+2}')


for v in FRANSAC_versions:
	df1 = pd.DataFrame({'$\\phi_1$' : []})
	for model_e in model_estimators:
		diff = []
		for i in range(tests_num):
			#reading data
			file = file_path + str(v + 2) + '/' + model_e + '_score/txt_' + str(i + 1) + '.txt'
			data = np.loadtxt(file)
			
			#selecting score
			diff_i = data[1]
			diff = np.append(diff, diff_i)
		# first column in df, instantiate df
		if model_e == "fun1":
			df1['$\\phi_1$'] = diff
		# rest of the columns of df 
		elif model_e == "fun2":
			df1['$\\phi_2$'] = diff
		elif model_e == "fun3":
			df1['$\\phi_3$'] = diff
		elif model_e == "fun4":
			df1['$\\phi_4$'] = diff

	# sort columns by name and plot
	#df1 = df1.reindex(sorted(df1.columns), axis=1)
	#df1.boxplot(ax = axes[v], grid=False, sym='.', medianprops=dict(color=mycolors[8]), boxprops=dict(color=mycolors[39]), whiskerprops=dict(color=mycolors[39]))
	#axes[v].set_ylim((-2.5,10))
	#axes[v].set_title(f'F-RANSAC{v+2}')

"""
if int(sys.argv[1]) == 1:
	title = "convergence_results_score"
else: 
	title = "convergence_results_theta"
"""

#plt.savefig(f'/home/esau/tfm/memoria/CAP_4/FIG/{title}.pgf', bbox_inches='tight', pad_inches=0.01, dpi=300)
print(f'Saving convergence plot in: /home/esau/tfm/memoria/CAP_4/FIG/{title} .pgf')
#plt.show()

df = pd.DataFrame()
df['theta'] = df0[f'$\\phi_{str(phi_idx)}$']
df['score'] = df1[f'$\\phi_{str(phi_idx)}$']

######################################################################################
## grafico xy + boxplot de cada variante con cada \phi
######################################################################################

left = 0.1
bottom = 0.1
top = 0.8
right = 0.8
plt.figure(figsize=myfigsize)
main_ax = plt.axes([left,bottom,right-left,top-bottom])
# create axes to the top and right of the main axes and hide them
top_ax = plt.axes([left,top,right - left,1-top])
plt.axis('off')
right_ax = plt.axes([right,bottom,1-right,top-bottom])
plt.axis('off')
main_ax.scatter(df['score'],  df['theta'], 2, myprops, alpha=1)
# Save the default tick positions, so we can reset them..

tcksx = main_ax.get_xticks()
tcksy = main_ax.get_yticks()

right_ax.boxplot(df['theta'], positions=[0], sym='.', notch=False, widths=1.)
top_ax.boxplot(df['score'], positions=[0],sym='.', vert=False, notch=False, widths=1.)

main_ax.set_yticks(tcksy) # pos = tcksy
main_ax.set_xticks(tcksx) # pos = tcksx
main_ax.set_yticklabels([int(j) for j in tcksy])
main_ax.set_xticklabels([int(j) for j in tcksx])
main_ax.set_ylim([min(tcksy-1),max(tcksy)])
main_ax.set_xlim([min(tcksx-1),max(tcksx)])

main_ax.set_xlabel('$\\Delta \\Phi$')
main_ax.set_ylabel('$\\Delta \\varepsilon (^\\circ)$')

# set the limits to the box axes
top_ax.set_xlim(main_ax.get_xlim())
top_ax.set_ylim(-1,1)
right_ax.set_ylim(main_ax.get_ylim())
right_ax.set_xlim(-1,1)


#print(df['score'].idxmax())
#print(df.loc[df['score'].idxmax(), 'score'])
plt.savefig(f'/home/esau/tfm/memoria/CAP_1/figure_format_figs/{title}.pgf', bbox_inches='tight', pad_inches=0.01, dpi=300)
print(title)
#plt.show()
