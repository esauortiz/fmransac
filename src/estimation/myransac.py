import numpy as np
import yaml
from estimation_results import compute_theta

def _max_trials(n_inliers, n_samples, min_samples, probability, outlier_ratio = None):

	"""Determine number trials such that at least one outlier-free subset is
	sampled for the given inlier/outlier ratio.
	Parameters
	----------
	n_inliers : int
		Number of inliers in the data.
	n_samples : int
		n_total number of samples in the data.
	min_samples : int
		Minimum number of samples chosen randomly from original data.
	probability : float
		Probability (confidence) that one outlier-free sample is generated.
		Desired probability that we get a good sample
	outlier_ratio : float
	Returns
	-------
	trials : int
		Number of trials.
	"""
	if n_inliers == 0:
		return np.inf

	nom = 1 - probability
	if nom == 0:
		return np.inf

	if outlier_ratio is not None:
		inlier_ratio = 1 - outlier_ratio
	else:
		inlier_ratio = n_inliers / float(n_samples)

	denom = 1 - inlier_ratio ** min_samples
	if denom == 0:
		return 1
	elif denom == 1:
		return np.inf

	nom = np.log(nom)
	denom = np.log(denom)
	if denom == 0:
		return 0

	return int(np.ceil(nom / denom))

def ransac_score(threshold, residuals):
	"""Determine base ransac version score function
	Parameters
	----------
	threshold : float
		residual threshold, determines if a point is an outlier
	residuals :	array(dtype = float)
		array of residuals w.r.t. sample model
	"""
	return abs(residuals) > threshold

def loransac_score(threshold, residuals):
	"""Determine base ransac version score function
	Parameters
	----------
	threshold : float
		residual threshold, determines if a point is an outlier
	residuals :	array(dtype = float)
		array of residuals w.r.t. sample model
	"""
	return ransac_score(threshold, residuals)

def msac_score(threshold, residuals):
	"""Determine msac score function
	Parameters
	----------
	threshold : float
		residual threshold, determines if a point is an outlier
	residuals :	array(dtype = float)
		array of residuals w.r.t. sample model
	"""	
	size = np.size(residuals)
	scores = np.ones(size) * (threshold ** 2)
	for i, residual in zip(range(size), residuals):
		if residual > threshold:
			continue
		scores[i] = residual**2
		# highest score \leq 1
	return scores

# Set of fuzzy score functions

def phi1(n, theta, residuals):
	size = np.size(residuals)
	fuzzy_scores = np.zeros(size)
	for i, residual in zip(range(size), residuals):
		if residual > (n * theta):
			continue 
		fuzzy_scores[i] = (1 - (residual / (n * theta))) ** n
	return fuzzy_scores

def phi2(n, theta, residuals):
	size = np.size(residuals)
	fuzzy_scores = np.zeros(size)
	for i, residual in zip(range(size), residuals):
		if residual > (theta):
			continue 
		fuzzy_scores[i] = 1 - ((residual ** n) / (theta ** n))
	return fuzzy_scores

def phi3(n, theta, residuals):
	exponents = (residuals ** n) / (theta ** n) 
	fuzzy_scores = np.exp(-exponents)
	return fuzzy_scores

def phi4(m, n, theta, residuals):
	fuzzy_scores = (theta ** n) / ((theta ** n) + (m * (residuals ** n)))
	return fuzzy_scores

def fun1_score(n, theta, residuals):
	return phi1(n, theta, residuals)

def fun2_score(n, theta, residuals):
	return phi2(n, theta, residuals)

def fun3_score(n, theta, residuals):
	return phi3(n, theta, residuals)

def fun4_score(n, theta, residuals):
	return phi4(1, n, theta, residuals)

def sac_score(fun, n, theta, residual_threshold, sample_model_residuals):
	"""Determine `ransac version` cost based in `fun` score function
	Parameters
	----------
	fun : function
		score function
	n : float
		function parameter
	theta : float
		residual threshold
	sample_model_residuals : array(dtype = float)
	Returns
	-------
	cost : float
		`ransac version` cost
	"""
	if fun.__name__ in ["ransac_score", "msac_score", "loransac_score"]:
		return fun(residual_threshold, sample_model_residuals)
	# Es posible que sea necesario cambiar residual_threshold por sigma
	# en el caso de las opciones fuzzy
	return fun(n, theta, sample_model_residuals)

def get_inliers(data, residuals, threshold, model = None):
	if model is not None: 
		residuals = np.abs(model.residuals(*data))
	inliers_mask = residuals < threshold
	inliers_data = [d[inliers_mask] for d in data]
	return inliers_mask, inliers_data

def local_optimizacion(n_iterations, m_theta, residual_threshold, model_class, best_model, best_residuals, best_model_cost, data, seed):
	# implementation based in Lebeda, Karel & Matas, Jiri & Chum, Ondrej. (2012). Fixing the locally optimized RANSAC. 10.5244/C.26.95
	new_model_found = False
	initial_model = model_class()

	# Mmθ ← model estimated by LSq onfind_inliers(M∗s,mθ·θ)
	_, Ms_inliers_data = get_inliers(data, best_residuals, m_theta * residual_threshold)
	initial_model.estimate(*Ms_inliers_data)

	# Ibase ← find_inliers(Mmθ,θ)
	Ibase_mask, Ibase = get_inliers(data, None, residual_threshold, initial_model)
	n_Ibase = np.sum(Ibase_mask)

	min_samples = int(np.min((14, n_Ibase/2)))
	Mis_min_cost = 0
	lo_random_state = np.random.RandomState(seed)

	# LORANSAC params
	iters = 4 # for iterative LSq
	delta_theta = (m_theta * residual_threshold - residual_threshold) / (iters - 1)

	for i in range(n_iterations):
		# non-minimal subset of base inliers
		spl_idxs = lo_random_state.choice(n_Ibase, min_samples, replace=False)
		samples = [d[spl_idxs] for d in Ibase]
		Mis = model_class()
		success = Mis.estimate(*samples)
		# compute cost
		Mis_residuals = np.abs(Mis.residuals(*data))
		Mis_cost = np.sum(fun1_score(1, residual_threshold, Mis_residuals))
		# verify if is the best local model and the best global model
		if (Mis_cost > Mis_min_cost and Mis_cost > best_model_cost):
			best_model = Mis
			Mis_min_cost = Mis_cost
			new_model_found = True

		# Algorithm 3 in cited paper
		Mr, Mr_cost = iterative_LSq(n_iterations = iters,
									m_theta = m_theta,
									delta_theta = delta_theta, # acording to table 1 in cited paper
									residual_threshold = residual_threshold,
									model_class = model_class,
									Mis = Mis,
									data = data)
		if (Mr_cost > Mis_min_cost and Mr_cost > best_model_cost):
			best_model = Mr
			Mis_min_cost = Mr_cost
			new_model_found = True
		
	return best_model, new_model_found

def iterative_LSq(n_iterations, m_theta, delta_theta, residual_threshold, model_class, Mis, data):
	# M′←model estimated by LSq onfind_inliers(Mis,θ)
	Mr_best = model_class()
	Mr = model_class()
	_, data_inliers  = get_inliers(data, None, residual_threshold, Mis)
	Mr_best.estimate(*data_inliers)
	Mr.estimate(*data_inliers)

	Mr_residuals = np.abs(Mr_best.residuals(*data))
	Mr_scores = fun1_score(1, residual_threshold, Mr_residuals)
	Mr_min_cost = np.sum(Mr_scores)

	residual_threshold_r = m_theta * residual_threshold

	for i in range(n_iterations):
		Mr_inliers_mask,_ = get_inliers(data, None, residual_threshold_r, Mr)
		Mr_residuals = np.abs(Mr.residuals(*data))
		# w′←computed weights ofI′(depend on model). 
		wr = fun1_score(1, residual_threshold_r, Mr_residuals)
		wr /= np.sum(wr)
		# M′←model estimated by LSq onI′weighted byw
		Mr.estimate(*data, wr)
		Mr_residuals = np.abs(Mr.residuals(*data))
		Mr_scores = fun1_score(1, residual_threshold, Mr_residuals)
		Mr_cost = np.sum(Mr_scores)
		if Mr_cost > Mr_min_cost:
			Mr_best = Mr
			Mr_min_cost = Mr_cost 
		residual_threshold_r -= delta_theta
	return Mr_best, Mr_min_cost

def ransac(data, model_class, min_samples, n, 
	residual_threshold, max_trials, 
	stop_probability, outlier_ratio, sigma, seed, score_fun, save_path = None):
	variante_1 = False
	variante_2 = False
	variante_3 = False
	variante_3_cte = 0.5
	variante_4 = True
	converge = True
	convergence_threshold = 0.0005
	improvements = 0

	str_variante = '1'
	if variante_2 == True:
		str_variante = '2'
	if variante_3 == True:
		str_variante = '3'
	if variante_4 == True:
		str_variante = '4'

	is_fuzzy_estimator = score_fun.__name__ in ["fun1_score","fun2_score","fun3_score","fun4_score", "loransac_score"]
	is_loransac = False

	if score_fun.__name__ == "loransac_score":
		is_loransac = True
		lo_random_state = np.random.RandomState(seed)
	best_model = None
	best_inliers = np.zeros((np.shape(data)[0],), dtype = bool)
	best_residuals = np.zeros((np.shape(data)[0],), dtype = float)
	if is_fuzzy_estimator:
		best_scores = np.zeros((np.shape(data)[0],), dtype = float)
	else:
		best_scores = np.ones((np.shape(data)[0],), dtype = float) * np.inf

	if variante_4:
		# all points are considered inliers
		sample_model_inliers = np.ones((np.shape(data)[0],), dtype = bool)

	random_state = np.random.RandomState(seed)
	
	teoric_max_trials = _max_trials(None, None, min_samples, stop_probability, outlier_ratio)
	if max_trials > teoric_max_trials: max_trials = teoric_max_trials

	if not isinstance(data, (tuple, list)):
			data = (data, )

	num_samples = len(data[0])

	if not (0 < min_samples < num_samples):
		raise ValueError("`min_samples` must be in range (0, <number-of-samples>)")

	if residual_threshold < 0:
		raise ValueError("`residual_threshold` must be greater than zero")

	if max_trials < 0:
		raise ValueError("`max_trials` must be greater than zero")

	if not (0 <= stop_probability <= 1):
		raise ValueError("`stop_probability` must be in range [0, 1]")

	#for the first run use initial guess of inliers
	spl_idxs = random_state.choice(num_samples, min_samples, replace=False)

	for num_trials in range(max_trials):
		#do sample selection according data pairs
		samples = [d[spl_idxs] for d in data]
		#for next iteration choose random sample set and be sure that no samples repeat
		spl_idxs = random_state.choice(num_samples, min_samples, replace=False)
		# estimate model for current random sample set
		sample_model = model_class()

		# First estimation i.e. whithout weights
		success = sample_model.estimate(*samples)
		# if the model could not be estimate then continue
		if success is not None and not success:
			continue
		sample_model_residuals = np.abs(sample_model.residuals(*data))
		# compute scores
		sample_model_scores = sac_score(fun = score_fun, 
										n = n, 
										theta = sigma, 
										residual_threshold = residual_threshold, 
										sample_model_residuals = sample_model_residuals)
		# consensus set / inliers
		if is_fuzzy_estimator and (variante_1 or variante_2):
			sample_model_inliers = sample_model_residuals < residual_threshold
			# only selecting inliers scores
			sample_model_scores[sample_model_inliers == False] = 0
			sample_model_score = np.sum(sample_model_scores)
		elif is_fuzzy_estimator and (variante_3):
			sample_model_inliers = sample_model_scores > variante_3_cte
			# only selecting inliers scores
			sample_model_scores[sample_model_inliers == False] = 0
			sample_model_score = np.sum(sample_model_scores)
		elif is_fuzzy_estimator and variante_4:
			sample_model_score = np.sum(sample_model_scores)
		else:
			sample_model_inliers = sample_model_residuals < residual_threshold
			sample_model_score = np.sum(sample_model_scores)

		best_score = np.sum(best_scores)
		if (is_fuzzy_estimator and best_score < sample_model_score) or (not is_fuzzy_estimator and best_score > sample_model_score):
			improvements += 1
			best_model = sample_model
			best_scores = sample_model_scores
			best_inliers = sample_model_inliers
			best_residuals = sample_model_residuals

			#dynamic_max_trials = _max_trials(np.sum(best_inliers), num_samples, min_samples, stop_probability)
			#if num_trials >= dynamic_max_trials:
			#	break
	# failure if inliers_num < min_samples
	if np.sum(best_inliers) < min_samples:
		best_model = None
		best_inliers = None
		best_scores = None
		improvements = None
		return best_model, best_inliers, best_scores, improvements

	# estimate final model using all inliers
	# ransac nor msac wont never execute iterative module
	if best_inliers is not None and (not converge or score_fun.__name__ == 'ransac_score' or score_fun.__name__ == 'msac_score' or score_fun.__name__ == 'loransac_score'):
		# this iterations refers to which are made in iterative module
		iterations  = 0
		#if is_fuzzy_estimator
		if is_fuzzy_estimator and (variante_2 or variante_3 or variante_4):
			# weighted estimation
			if (variante_4):
				best_scores = best_scores / np.sum(best_scores)
				best_model.estimate(*data, best_scores)
			else:
				data_inliers = [d[best_inliers] for d in data]
				scores_inliers = best_scores[best_inliers]
				scores_inliers = scores_inliers / np.sum(scores_inliers)
				best_scores = best_scores / np.sum(best_scores)
				best_model.estimate(*data_inliers, scores_inliers)
		else:
			# select inliers for each data array (variante_1 and ransac)
			data_inliers = [d[best_inliers] for d in data]
			best_model.estimate(*data_inliers)

	elif best_inliers is not None and converge:
		# converge implies variante_2, variante_3 or variante_4

		def estimate_w_convergence(model, model_class, id, data, scores, score_fun, n, sigma, residual_threshold, iterations = None, best_inliers = None, best_model = None, best_scores = None, save_path = None, mytxt = None):

			if iterations is not None:
				iterations += 1

			# debug: change iterations to 1
			if iterations == -1:
				if save_path is not None:
					yaml_file = save_path + str(id) + '.yaml'
					with open(yaml_file, 'r') as stream:
						test_params = yaml.safe_load(stream)
					params_original_i = np.array(test_params['data_params']['model_params'])
					params_original_i = params_original_i.reshape(2, -1)
					theta, _= compute_theta(params_original_i[1], model.params[1])
					#print('(STARTING POINT)')
					#print('(STARTING POINT)')

			# prev_model and new_model
			prev_params = model.params
			new_model = model_class()

			# normalize scores and estimate new_model

			if best_inliers is None:
				scores = scores / np.sum(scores)
				new_model.estimate(*data, scores)
			else:
				data_inliers = [d[best_inliers] for d in data]
				scores_inliers = scores[best_inliers]
				scores_inliers = scores_inliers / np.sum(scores_inliers)
				new_model.estimate(*data_inliers, scores_inliers)

			# residuals and scores
			new_residuals = new_model.residuals(*data)
			new_scores = sac_score(	fun = score_fun, 
									n = n, 
									theta = residual_threshold, 
									residual_threshold = residual_threshold, 
									sample_model_residuals = np.abs(new_residuals))

			# check if theta is less in each iteration. For debug comment save_path line
			save_path = None
			if save_path is not None:
				# reading original params and computing angle theta between normal vectors (original and estimated)
				yaml_file = save_path + str(id) + '.yaml'
				with open(yaml_file, 'r') as stream:
					test_params = yaml.safe_load(stream)
				params_original_i = np.array(test_params['data_params']['model_params'])
				params_original_i = params_original_i.reshape(2, -1)
				theta, _= compute_theta(params_original_i[1], new_model.params[1])
				# create array to save theta and score
				if iterations == 1:
					# last item is 0 iterations
					mytxt = - np.array([theta, np.sum(new_scores[best_inliers]), 0])
					#print('theta: {:.2f} and score: {:.2f}(STARTING POINT)'.format(theta, np.sum(new_scores[best_inliers])))

				else:
					True
					# else stack next iteration results
					#mytxt = np.vstack((mytxt, np.array([theta, np.sum(new_scores[best_inliers])])))
					#print('theta: {:.2f} | new_score: {:.2f}'.format(theta, np.sum(new_scores[best_inliers])))
			def check_convergence(prev_params, new_params):
				L_infinity = abs(np.asarray(prev_params) - np.asarray(new_params))
				if np.max(L_infinity) < convergence_threshold:
					return True
				return False

			if (check_convergence(prev_params, new_model.params) == True) or iterations > 25: # or iterations > 25: For Homography model which does not seem to converge in a reasonable number of iterations
				#mytxt += np.array([theta, np.sum(new_scores[best_inliers]), iterations])
				#np.savetxt(('/home/esau/tfm/Tests/convergence_results/estylf/' + f'v{str_variante}/' + score_fun.__name__ + '/txt_' + str(id) + '.txt'), mytxt)
				return new_model, new_scores, iterations
			else:
				return estimate_w_convergence(new_model, model_class, seed, data, new_scores, score_fun, n, sigma, residual_threshold, best_inliers = best_inliers, iterations = iterations, save_path = save_path, mytxt = mytxt)

		if variante_4:
			best_model, best_scores, iterations = estimate_w_convergence(best_model, model_class, seed, data, best_scores, score_fun, n, sigma, residual_threshold, iterations = 0, save_path = save_path)
		elif variante_3:
			# best_inliers as argument for first iteration
			best_model, best_scores, iterations = estimate_w_convergence(best_model, model_class, seed, data, best_scores, score_fun, n, sigma, residual_threshold, best_inliers = best_inliers, iterations = 0, save_path = save_path)
		elif variante_2:
			# best_inliers as argument for first iteration
			#print(score_fun.__name__)
			best_model, best_scores, iterations = estimate_w_convergence(best_model, model_class, seed, data, best_scores, score_fun, n, sigma, residual_threshold, best_inliers = best_inliers, iterations = 0, save_path = save_path)

	#best_residuals = best_model.residuals(*data)
	return best_model, best_inliers, best_scores, iterations
