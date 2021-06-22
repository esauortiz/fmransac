from myransac import fun1_score, fun2_score, fun3_score, fun4_score, sac_score
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


def ransac_score(_, threshold, residuals):
	scores = (abs(residuals) < threshold).astype(int)
	return -scores

def msac_score(_, threshold, residuals):
	size = np.size(residuals)
	scores = np.zeros(size) + 1
	for i, residual in zip(range(size), residuals):
		if residual > threshold:
			continue
		scores[i] = threshold**2*1.25- residual**2
	scores /= (threshold**2)*1.25
	return - scores

def mlesac_cost(inlier_ratio, residual, v):
	sigma = 0.6
	cost = inlier_ratio*(1/np.sqrt(2.0*np.pi*sigma**2.0))*np.exp(-residual**2.0/(2*sigma**2.0)) + (1-inlier_ratio)/v
	return cost

def mlesac_score(_, threshold, residuals):
	size = np.size(residuals)

	inliers_num = np.sum(residuals < threshold)
	inlier_ratio = inliers_num/size
	v = np.max(residuals) - np.min(residuals)

	scores = np.zeros(size)
	for i, residual in zip(range(size), residuals):
		scores[i] = mlesac_cost(inlier_ratio, residual, v)

	#scores /= np.max(scores)
	#scores *= 0.9
	scores = np.log(scores)
	scores = scores/(np.max(scores))
	return (scores/np.max(scores) - 1.025) * 0.95

def LS_score(_, threshold, residuals):
	scores= threshold**2*1.25 - residuals**2
	return -scores/((threshold**2)*1.25)

def plot_fun(ax, label, fun, n_list, theta_list, residuals):
	n_list = np.round_(n_list, 2)
	for n, theta in zip(n_list, theta_list):
		if n == 1:
			color = 'Black'
			zorder = 1
		else:
			color = [0.75,0.75,0.75,1]
			zorder = 0
		#ax.set_title(label)
		scores = fun(n, theta, abs(residuals))
		# cutting score function
		#inliers = abs(residuals) < theta
		#inliers = scores > 0.5
		#outliers = inliers == False
		#scores[outliers] = 0
	
		if fun.__name__ in {'msac_score', 'ransac_score'}:
			ax.plot(residuals, scores, color = 'black')
		else:
			ax.plot(residuals, scores, label = f'n={n}', color = color, zorder = zorder)
			#ax.legend(fancybox = False, handlelength = 1)
	return True

def get_scores(fun, n_list, theta_list, residuals):
	n_list = np.round_(n_list, 3)
	scores = []
	for n, theta in zip(n_list, theta_list):
		fuzzy_score = sac_score(fun, n, theta, abs(residuals))
		fuzzy_score = round(fuzzy_score, 3)
		scores.append(fuzzy_score)
	return scores


################################################################
## Score fun list
################################################################

score_fun_list = {
	"ransac" : ransac_score,
	"msac" : msac_score,
	"mlesac" : mlesac_score,
	"ls" : LS_score,
	"fun1" : fun1_score,
	"fun2" : fun2_score,
	"fun3" : fun3_score,
	"fun4" : fun4_score
}

################################################################
## Plot params
################################################################
cm = 1/2.54  # centimeters in inches

#small size
myfigsize=(4*cm,4*cm)
#medium size
#myfigsize=(5.75*cm,5.75*cm)

save_figure = True
file_path = "/home/esau/tfm/memoria/CAP_4A/FIG/"
plot_title = "fun4"

################################################################
## Using latex text format
################################################################
if save_figure == True:
	matplotlib.rcParams['text.usetex'] = True

	plt.rcParams.update({
	    "text.usetex": True,
	    "font.family": "serif"

	})

for n_value in [2]:
	#n_list = [4,4,4,4]
	#n_list = [0.5,0.5,0.5,0.5]
	#n_list = [0.5,2,4]
	n_list = np.array([1,2])
	#n_list = [0.5,1,2,4]
	#n_list = np.repeat(n_value,6)
	sigma = 1
	residual_threshold = 2
	#theta = np.repeat(residual_threshold,4)
	theta = [2,2]
	
	#residuals = np.linspace(-10,20,100)
	mu, sigma = 0, 1
	np.random.seed(5)
	residuals = np.sort(np.linspace(-5, 5, 10000))
	#residuals = np.loadtxt('/home/esau/tfm/Tests/HomographyModel/batch_0/data/test_1_residuals.txt')
	residuals = np.sort(residuals)

	#Plot score function shape given a function score (fun1, fun2, ..., ransac)
	fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(myfigsize))

	plot_fun(ax ,plot_title, score_fun_list[plot_title], n_list, theta, residuals)
	#plot_fun(axs ,"MSAC", score_fun_list["msac"], n_list, theta, residuals)
	#plot_fun(axs ,"MLESAC", score_fun_list["mlesac"], n_list, theta, residuals)
	#plot_fun(axs ,"LS", score_fun_list["ls"], n_list, theta, residuals)
	#plot_fun(axs ,"fun3", score_fun_list["fun3"], n_list, theta, residuals)
	
	plt.ylabel("Score")
	plt.xlabel("Error")
	plt.xlim((-4,4))  
	plt.ylim((-0.05,1.05))  

	# ransac and msac remains unvariable changing "n". Also theta is interpreted as threshold
	#plot_fun("ransac",ransac_score, [1], theta, residuals)
	#plot_fun("msac",msac_score, [1], theta, residuals)
	#plt.legend(loc='best')

	if save_figure == True:
		print(f'Table saved in: {file_path}')
		print(f'Tex file name: {plot_title} with .pgf extension PRINTED')
		plt.savefig(f'{file_path}{plot_title}.pgf', bbox_inches='tight', pad_inches=0.01, dpi=300)
	else:
		plt.show()


