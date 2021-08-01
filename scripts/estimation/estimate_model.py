from myransac import ransac, ransac_score, msac_score, fun1_score, fun2_score, fun3_score, fun4_score, loransac_score
from myfit import LineModelND, EllipseModel, CircleModel, BaseModel, PlaneModelND, HomographyModel
from data.gen_data import line_nd_data, ellipse_data, circle_data, plane_nd_data

from multiprocessing import Pool
from pprint import pprint

import numpy as np
import sys
import yaml
import io


# Read test params
test_num = int(sys.argv[1])
save_path = sys.argv[2]
yaml_file_path = save_path + '/yaml/test_'
data_file_path = save_path + '/data/test_'

# run estimation
model_estimators = list(sys.argv[3].split(','))

# ransac version score function dict
score_fun_dict = {
	"ransac" : ransac_score,
	"msac" : msac_score,
	"loransac" : loransac_score,
	"fun1" : fun1_score,
	"fun2" : fun2_score,
	"fun3" : fun3_score,
	"fun4" : fun4_score,
}

# Update save_path, now it is results save path
save_path = save_path + '/results/'

#for test_id in range(1, test_num + 1):
def run_test(test_id):

	sys.stdout.write("\rEstimating params for test %i" % test_id)
	sys.stdout.flush()
	test_id = str(test_id)
	yaml_file = yaml_file_path + test_id + '.yaml'

	# Read test params
	with open(yaml_file, 'r') as stream:
		test_params = yaml.safe_load(stream)
	ransac_params = test_params['ransac_params']

	# Read model_class
	model_class = test_params['data_params']['model']

	# Other relevant variables
	gen_data_function = None
	gen_data_length = None

	if model_class == 'LineModelND':
		# set model_class
		model_class = LineModelND
		# set data generation function
		gen_data_function = line_nd_data
		# set generated data length
		margin = 10 # extend the line for more visibility
		axis_range = [np.min(data, axis = 0)[0] - margin, np.max(data, axis = 0)[0] + margin]
		gen_data_length = [axis_range, 0]

	elif model_class == 'CircleModel':
		# set model_class
		model_class = CircleModel
		# set data generation function
		gen_data_function = circle_data
		# set generated data length
		gen_data_length = [30]

	elif model_class == 'EllipseModel':
		# set model_class
		model_class = EllipseModel
		# set data generation function
		gen_data_function = ellipse_data
		# set generated data length
		gen_data_length = [30]
	elif model_class == 'PlaneModelND':
		# set model_class
		model_class = PlaneModelND
		# set data generation function
		gen_data_function = plane_nd_data
		# set generated data length
		gen_data_length = [*test_params['data_params']['data_len'], test_id]
	elif model_class == 'HomographyModel':
		# set model_class
		model_class = HomographyModel

	# Read noisy data
	if model_class.__name__ == 'HomographyModel':
		try:
			#correspondences = np.loadtxt(data_file_path + test_id + '_correspondences.txt')
			data1 = np.loadtxt(data_file_path + test_id + '_proj1.txt')
			data2 = np.loadtxt(data_file_path + test_id + '_proj2.txt')
			# data1 and data2 are suposed to be ordered. Index N should be data2[N]=H·data1[N]
			data = np.column_stack((data1,data2))
		except IOError:
			print(f'Data for test_{test_id} not found')
			#continue
	else:
		try:
			data = np.loadtxt(data_file_path + test_id + '.txt', delimiter=" ")
			#data = data[~np.isnan(data)]
			#data = data.reshape((-1,3))
		except IOError:
			print(f'Data for test_{test_id} not found')
			return True
			#continue
	
	# normalazing data
	#data, H = musigma_norm(data)

	# default estimation (e.g. Total Least Squares)
	# fit line using all data
	model_d = model_class()
	model_d.estimate(data)
	# generate estimated model data
	#model_data = gen_data_function(*gen_data_length, *model_d.params)
	# save results
	#np.savetxt((save_path + 'default/test_' + test_id + '_data.txt'), model_data)
	# denormalizing params before saving
	np.savetxt((save_path + 'default/test_' + test_id + '_params.txt'), model_d.params)

	for model_e in model_estimators:
		# robustly fit line only using inlier data (robust model)
		model_r, inliers, best_scores, iterations = ransac(data, model_class, *ransac_params, score_fun_dict[model_e], save_path = yaml_file_path)
		# generate data with robust model
		#model_data = gen_data_function(*gen_data_length, *model_r.params)
		#residuals = np.abs(model_r.residuals(data))
		#np.savetxt((save_path + model_e + '/test_' + test_id + '_data.txt'), model_data)
		#np.savetxt((save_path + model_e + '/test_' + test_id + '_residuals.txt'), residuals)
		if model_r is not None:
			np.savetxt((save_path + model_e + '/test_' + test_id + '_inliers.txt'), inliers)
			np.savetxt((save_path + model_e + '/test_' + test_id + '_params.txt'), model_r.params)
			np.savetxt((save_path + model_e + '/test_' + test_id + '_more_info.txt'), np.array([iterations]))
			np.savetxt((save_path + model_e + '/test_' + test_id + '_residuals.txt'), best_scores)
	return True

#run_test(test_num)
with Pool(8) as p:
	p.map(run_test, range(1, test_num + 1))

#for i in range(1, test_num + 1):
#	run_test(i)