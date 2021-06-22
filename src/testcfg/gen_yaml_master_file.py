import sys
import yaml
import io

from scipy.stats.distributions import chi2

def get_batch_list(start_batch, length):
	batch_list = []
	for i in range(length):
		batch_list.append(f'batch_{start_batch+i}')
	return batch_list

####################################
#	data_params for LineModelND
"""
	    'data_params' : {
	        'axis_range' : [15, 16],
	        'axis': 0,
	        'origin': [-10, 10],
	        'direction': [-10, 10],
	        'dim': 2,
	        # if quadrant wont be setted -> 'quadrant': 0
			'quadrant': [45,90]
	    },
"""
####################################
#	data_params for CircleModel
"""
	    'data_params' : {
	    	'samples' : [100],
			'xc' : [5,6],
			'yc' : [5,6],
			'radius' : [10,12]
	    },
"""
####################################
#	data_params for EllipseModel
"""
	    'data_params' : {
	    	'samples' : [1000],
			'xc' : [0,0],
			'yc' : [0,0],
			'a' : [6,7],
			'b' : [4,5],
			# theta in rads will de devided by 10^-3 later
			'theta' : [0,3140],
	    	'ranges' : ranges,
			'bbox_limits' : bbox_limits,
			'bbox_limits_tolerance' : 3
	    },
"""
#####################################
#	data_params for PlaneModelND
"""
	    'data_params' : {
			'ranges' : ranges,
			'dim' : dim,
	    	'samples' : [10000],
			'nrm_vctr_range' : [-100,100],
			'orgn_range' : [0,0],
			'bbox_limits' : bbox_limits,
			'bbox_limits_tolerance' : 2
	    },
"""
#####################################
#	data_params for HomographyModel
"""
	    'data_params' : {
			'dim' : dim,
	    	'samples' : [points_num[i]],
			'ranges' : ranges,
			
			'nrm_vctr_range' : [-100,100],
			'orgn_range' : [0,0],

			# extrinsics of camera
			'tx' : [0,3],
			'ty' : [0,3],
			'tz' : [1,3],
			# rotation range in degrees
			'theta' : [0,phi],

			'bbox_limits' : bbox_limits,
			'bbox_limits_tolerance' : 3
	    },
"""
#####################################

save_path = sys.argv[1]
test_batch = sys.argv[2]
model_class = 'HomographyModel'
min_samples = 3
dim = 2
target_dim = [0,1]
points_num = 100

# total outlier ratio i.e. all the data will contain this outlier_ratio
outlier_ratios = [0.4,0.4,0.4,0.4,0.4]
# outlier_ratio in uniform noise
un_outlier_ratio = 1
stop_prob = 0.99
max_t = 5000
max_trials = [max_t,max_t,max_t,max_t,max_t,max_t]

bbox_mag = 10*1
rng_mag = bbox_mag*1.5
un_level = bbox_mag
n_list = [1,1,1,1,1,1]

sd_list = [1,1,1,1,1,1]

# phi as rotation angle in degrees in HomographyModel data
phi = 45
tau1 = ((sd_list[0]**2)*float(chi2.ppf(0.997, df=2)))**0.5
tau2 = ((sd_list[1]**2)*float(chi2.ppf(0.997, df=2)))**0.5
tau3 = ((sd_list[2]**2)*float(chi2.ppf(0.997, df=2)))**0.5
tau4 = ((sd_list[3]**2)*float(chi2.ppf(0.997, df=2)))**0.5

#tau1, tau2, tau3, tau4 = [1.3*sd_list[0],1.3*sd_list[1],1.3*sd_list[2],1.3*sd_list[3]]
#tau1, tau2, tau3, tau4 = [3*sd_list[0],3*sd_list[1],3*sd_list[2],3*sd_list[3]]
#tau1, tau2, tau3, tau4, tau5, tau6 = [0.5,0.5,1,1.5,2,3]

threshold_list = [tau1,tau2,tau3,tau4]
#threshold_list = [1,2,2.5,3,4]
theta = threshold_list

#gn_num = points_num * (1 - outlier_ratio) - ((points_num * outlier_ratio* (1 - un_outlier_ratio)) / un_outlier_ratio)
#un_num = (points_num * outlier_ratio) / un_outlier_ratio
on_num = 0

inliers_num = [int(points_num*(1-x)) for x in outlier_ratios]
outliers_num = [int(points_num*(x)) for x in outlier_ratios]

#inliers_num = [points_num, points_num, points_num, points_num]
#outliers_num = [int(outlier_ratio*inliers_n/(1-outlier_ratio)) for inliers_n,outlier_ratio in zip(inliers_num, outlier_ratios)]
points_num = [x+y for x,y in zip(inliers_num, outliers_num)]

ranges =[[-rng_mag,rng_mag],
		[-rng_mag,rng_mag]]

bbox_limits = 	[[-bbox_mag,bbox_mag],
				[-bbox_mag,bbox_mag]]

start_batch = 53
batches_num = 4
batch_list = get_batch_list(start_batch, length = batches_num)

payload = dict()

for i in range(batches_num):
	payload[batch_list[i]] = {
		'model_class' : model_class,
	    'data_params' : {
			'dim' : dim,
	    	'samples' : [points_num[i]],
			'ranges' : ranges,
			
			'nrm_vctr_range' : [-100,100],
			'orgn_range' : [0,0],

			# extrinsics of camera
			'tx' : [0,3],
			'ty' : [0,3],
			'tz' : [1,3],
			# rotation range in degrees
			'theta' : [0,phi],

			'bbox_limits' : bbox_limits,
			'bbox_limits_tolerance' : 3
	    },

		'noise_params' : {
			#gn_params  : [residual_thresh,		gn_num,	max_trials,	mu, sigma,  gn_targdim	outlier_ratio]
			'gn_params' : [sd_list[i]*2,	inliers_num[i], inliers_num[i]*1000, 0,  sd_list[i],  target_dim,	0.0],
 
 			#un_params  : [residual_thresh,		un_num,	max_trials, un_noise_level, 		un_targdim	outlier_ratio]
			'un_params' : [sd_list[i]*2,	outliers_num[i], outliers_num[i]*1000, [-un_level, un_level],  target_dim,	1],
			
			#on_params  : [residual_thresh  on_num,	maxtrials,  on_noise_level, cloud_size, on_targdim]
			'on_params' : [10,				on_num, on_num*100, 15,             2,          target_dim]			
		},
		#ransac_params' : [min_samples,	n,	residual_threshold, max_trials,	stop_prob, 	outlier_ratio, 	sigma]       
		'ransac_params' : [min_samples,	n_list[i],	threshold_list[i],				max_trials[i], 		stop_prob, 	outlier_ratios[i],	theta[i]]
	}

# Write YAML file
with io.open(save_path + '/batch_params.yaml', 'w', encoding='utf8') as outfile:
    yaml.dump(payload[test_batch], outfile, default_flow_style=False, allow_unicode=True)

"""
##################
# Homography Model
##################

for i in range(batches_num):
	payload[batch_list[i]] = {
		'model_class' : model_class,
	    'data_params' : {
			'dim' : dim,
	    	'samples' : [points_num[i]],
			'ranges' : ranges,
			
			'nrm_vctr_range' : [-100,100],
			'orgn_range' : [0,0],

			# extrinsics of camera
			'tx' : [0,3],
			'ty' : [0,3],
			'tz' : [1,3],
			# rotation range in degrees
			'theta' : [0,phi],

			'bbox_limits' : bbox_limits,
			'bbox_limits_tolerance' : 3
	    },
		'noise_params' : {
			#gn_params  : [residual_thresh,		gn_num,	max_trials,	mu, sigma,  gn_targdim	outlier_ratio]
			'gn_params' : [sd_list[i]*130,	inliers_num[i], 30000, 0,  sd_list[i],  target_dim,	0.0],
 
 			#un_params  : [residual_thresh,		un_num,	max_trials, un_noise_level, 		un_targdim	outlier_ratio]
			'un_params' : [sd_list[i]*130,	outliers_num[i], 30000, [-un_level, un_level],  target_dim,	0.0],
			
			#on_params  : [residual_thresh  on_num,	maxtrials,  on_noise_level, cloud_size, on_targdim]
			'on_params' : [10,				on_num, on_num*100, 15,             2,          target_dim]			
		},
		#ransac_params' : [min_samples,	n,	residual_threshold, max_trials,	stop_prob, 	outlier_ratio, 	sigma]       
		'ransac_params' : [min_samples,	n_list[i],	threshold_list[i],				max_trials, 		stop_prob, 	outlier_ratios[i],	theta[i]]
	}

# Write YAML file
with io.open(save_path + '/batch_params.yaml', 'w', encoding='utf8') as outfile:
    yaml.dump(payload[test_batch], outfile, default_flow_style=False, allow_unicode=True)


    save_path = sys.argv[1]
test_batch = sys.argv[2]
model_class = 'PlaneModelND'
min_samples = 6
dim = 5
target_dim = [0,1,2,3,4]
points_num = 300

# total outlier ratio i.e. all the data will contain this outlier_ratio
outlier_ratio = 0.4
inlier_ratio = 1 - outlier_ratio
#or_list = [0.2,0.4,0.5,0.6]
or_list = [outlier_ratio,outlier_ratio,outlier_ratio,outlier_ratio]
# outlier_ratio in uniform noise
un_outlier_ratio = 1
stop_prob = 0.99
max_trials = 2000

bbox_mag = 10*1
rng_mag = bbox_mag*1.5
un_level = bbox_mag
#n_list = [4, 4, 4, 4]
#n_list = [0.5, 0.5, 0.5, 0.5]
n_list = [2, 2, 2, 2]
#n_list = [0.5, 1, 2, 4]
#threshold_list = [1, 2, 3, 4]
threshold_list = [1/20,2/20,3/20,4/20]
theta = threshold_list
#theta = [1,2,3,4]

sd_list = [1,1,1,1]

gn_num = points_num * (1 - outlier_ratio) - ((points_num * outlier_ratio* (1 - un_outlier_ratio)) / un_outlier_ratio)
un_num = (points_num * outlier_ratio) / un_outlier_ratio
on_num = 0


ranges =[[-rng_mag,rng_mag],
		[-rng_mag,rng_mag],
		[-rng_mag,rng_mag],
		[-rng_mag,rng_mag],
		[-rng_mag,rng_mag]]

bbox_limits = 	[[-bbox_mag,bbox_mag],
				[-bbox_mag,bbox_mag],
				[-bbox_mag,bbox_mag],
				[-bbox_mag,bbox_mag],
				[-bbox_mag,bbox_mag]]

start_batch = 42
batch_list = get_batch_list(start_batch, length = 4)

payload = {
	batch_list[0] : {
		'model_class' : model_class,
	    'data_params' : {
			'dim' : dim,
	    	'samples' : [10000],
			'nrm_vctr_range' : [-100,100],
			'orgn_range' : [0,0],
			'ranges' : ranges,
			'bbox_limits' : bbox_limits,
			'bbox_limits_tolerance' : 3
	    },
		'noise_params' : {
			#gn_params  : [residual_thresh,		gn_num,	max_trials,	mu, sigma,  gn_targdim	outlier_ratio]
			'gn_params' : [sd_list[0]*2,	(1-or_list[0])*points_num, (1-or_list[0])*points_num*3000, 0,  sd_list[0],  target_dim,	0.0],
 
 			#un_params  : [residual_thresh,		un_num,	max_trials, un_noise_level, 		un_targdim	outlier_ratio]
			'un_params' : [sd_list[0]*2,	or_list[0]*points_num, or_list[0]*points_num*3000, [-un_level, un_level],  target_dim,	un_outlier_ratio],
			
			#on_params  : [residual_thresh  on_num,	maxtrials,  on_noise_level, cloud_size, on_targdim]
			'on_params' : [10,				on_num, on_num*100, 15,             2,          target_dim]			
		},
		#ransac_params' : [min_samples,	n,	residual_threshold, max_trials,	stop_prob, 	outlier_ratio, 	sigma]       
		'ransac_params' : [min_samples,	n_list[0],	threshold_list[0],				max_trials, 		stop_prob, 	or_list[0],	theta[0]]
	},
	batch_list[1] : {
		'model_class' : model_class,
	    'data_params' : {
			'dim' : dim,
	    	'samples' : [10000],
			'nrm_vctr_range' : [-100,100],
			'orgn_range' : [0,0],
			'ranges' : ranges,
			'bbox_limits' : bbox_limits,
			'bbox_limits_tolerance' : 3
	    },
		'noise_params' : {
			#gn_params  : [residual_thresh,		gn_num,	max_trials,	mu, sigma,  gn_targdim	outlier_ratio]
			'gn_params' : [sd_list[1]*2,	(1-or_list[1])*points_num, (1-or_list[1])*points_num*3000, 0,  sd_list[1],  target_dim,	0.0],
 
 			#un_params  : [residual_thresh,		un_num,	max_trials, un_noise_level, 		un_targdim	outlier_ratio]
			'un_params' : [sd_list[1]*2,	or_list[1]*points_num, or_list[1]*points_num*3000, [-un_level, un_level],  target_dim,	un_outlier_ratio],
			
			#on_params  : [residual_thresh  on_num,	maxtrials,  on_noise_level, cloud_size, on_targdim]
			'on_params' : [10,				on_num, on_num*100, 15,             2,          target_dim]			
		},
		#ransac_params' : [min_samples,	n,	residual_threshold, max_trials,	stop_prob, 	outlier_ratio, 	sigma]       
		'ransac_params' : [min_samples,	n_list[1],	threshold_list[1],				max_trials, 		stop_prob, 	or_list[1],	theta[1]]
	},
	batch_list[2] : {
		'model_class' : model_class,
	    'data_params' : {
			'dim' : dim,
	    	'samples' : [10000],
			'nrm_vctr_range' : [-100,100],
			'orgn_range' : [0,0],
			'ranges' : ranges,
			'bbox_limits' : bbox_limits,
			'bbox_limits_tolerance' : 3
	    },
		'noise_params' : {
			#gn_params  : [residual_thresh,		gn_num,	max_trials,	mu, sigma,  gn_targdim	outlier_ratio]
			'gn_params' : [sd_list[2]*2,	(1-or_list[2])*points_num, (1-or_list[2])*points_num*3000, 0,  sd_list[2],  target_dim,	0.0],
 
 			#un_params  : [residual_thresh,		un_num,	max_trials, un_noise_level, 		un_targdim	outlier_ratio]
			'un_params' : [sd_list[2]*2,	or_list[2]*points_num, or_list[2]*points_num*3000, [-un_level, un_level],  target_dim,	un_outlier_ratio],
			
			#on_params  : [residual_thresh  on_num,	maxtrials,  on_noise_level, cloud_size, on_targdim]
			'on_params' : [10,				on_num, on_num*100, 15,             2,          target_dim]			
		},
		#ransac_params' : [min_samples,	n,	residual_threshold, max_trials,	stop_prob, 	outlier_ratio, 	sigma]       
		'ransac_params' : [min_samples,	n_list[2],	threshold_list[2],				max_trials, 		stop_prob, 	or_list[2],	theta[2]]
	},
	batch_list[3] : {
		'model_class' : model_class,
	    'data_params' : {
			'dim' : dim,
	    	'samples' : [10000],
			'nrm_vctr_range' : [-100,100],
			'orgn_range' : [0,0],
			'ranges' : ranges,
			'bbox_limits' : bbox_limits,
			'bbox_limits_tolerance' : 3
	    },
		'noise_params' : {
			#gn_params  : [residual_thresh,		gn_num,	max_trials,	mu, sigma,  gn_targdim	outlier_ratio]
			'gn_params' : [sd_list[3]*2,	(1-or_list[3])*points_num, (1-or_list[3])*points_num*3000, 0,  sd_list[3],  target_dim,	0.0],
 
 			#un_params  : [residual_thresh,		un_num,	max_trials, un_noise_level, 		un_targdim	outlier_ratio]
			'un_params' : [sd_list[3]*2,	or_list[3]*points_num, or_list[3]*points_num*3000, [-un_level, un_level],  target_dim,	un_outlier_ratio],
			
			#on_params  : [residual_thresh  on_num,	maxtrials,  on_noise_level, cloud_size, on_targdim]
			'on_params' : [10,				on_num, on_num*100, 15,             2,          target_dim]			
		},
		#ransac_params' : [min_samples,	n,	residual_threshold, max_trials,	stop_prob, 	outlier_ratio, 	sigma]       
		'ransac_params' : [min_samples,	n_list[3],	threshold_list[3],				max_trials, 		stop_prob, 	or_list[3],	theta[3]]
	}
}

# Write YAML file
with io.open(save_path + '/batch_params.yaml', 'w', encoding='utf8') as outfile:
    yaml.dump(payload[test_batch], outfile, default_flow_style=False, allow_unicode=True)
"""