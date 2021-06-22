#!/bin/bash

# Configurable params
model='HomographyModel' # {LineModelND, CircleModel, EllipseModel, PlaneModelND, HomographyModel}
#save_path_header='C:/Users/esauo/Documents/00_UIB-MEIN/00_TFM/Tests'
#save_path_header='/home/esau/Documents/TFM/Tests'
save_path_header='/home/esau/tfm/Tests'
#tests_num=$2
tests_num=1

# Master batch indicates if all the data of the master batch is
# going to be copied in the other batches of the test
#master_batch='batch_'$1 # Could be a 'batch_NN' or 'None'
master_batch='None' # Could be a 'batch_NN' or 'None'
#master_batch='batch_500' # Could be a 'batch_NN' or 'None'

initial_batch=$1
let batches_num=initial_batch+0

for i in $(seq $initial_batch $batches_num); do

	save_path=${save_path_header}/${model}/batch_$i

	# Create directories
	#rm -r ${save_path}
	mkdir -p ${save_path}
	mkdir -p ${save_path}/data
	mkdir -p ${save_path}/results
	mkdir -p ${save_path}/results/default
	mkdir -p ${save_path}/results/ransac
	mkdir -p ${save_path}/results/msac
	mkdir -p ${save_path}/results/fun1
	mkdir -p ${save_path}/results/fun2
	mkdir -p ${save_path}/results/fun3
	mkdir -p ${save_path}/results/fun4
	mkdir -p ${save_path}/results/loransac
	mkdir -p ${save_path}/yaml
	# Create master yaml: describes the batch params.
	# Also saves the master yaml in each batch directory
	python src/testcfg/gen_yaml_master_file.py ${save_path} batch_$i

	echo 'Running test: 'batch_$i
	echo 'Model class: '$model 
	echo 'Number of tests: '${tests_num}

	# Generate each test's yaml_file
	python src/testcfg/gen_yaml_file.py batch_$i ${tests_num} ${save_path}
	echo 'Yaml files generated'

	# Generating data only if current batch is master_batch or if there is none
	if [ batch_$i == $master_batch ] || [ $master_batch == 'None' ]
	then
		python src/data/gen_data_${model}.py ${tests_num} ${save_path}
		echo 'Data generated'
	# Else copy data from master batch in the current batch folder
	else
		#cp -r ${save_path_header}/${model}/${master_batch}/data ${save_path}
		echo 'Data generated (copied from master batch('$master_batch'))'
	fi

	# Estimate model params, default estimation will always be run
	python src/estimation/estimate_model.py ${tests_num} ${save_path} ransac # {ransac,msac,fun1,fun2,fun3,fun4}
	echo ''
	echo 'Tests finished. Generating results... '

	# Create results files
	python src/estimation/estimation_results.py ${save_path} ${tests_num} ${model} ransac

	echo 'Results generated: 'batch_$i
	echo ' '
done

