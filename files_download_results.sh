#!/bin/bash

path_header='/home/esau/tfm/Tests/PlaneModelND' # HomographyModel LineModelND CircleModel EllipseModel PlaneModelND

model_estimators=(
	'ransac'
	'msac'
	'fun1'
	'fun2'
	'fun3'
	'fun4'
	)

files=(
	'00_more_info.txt'
	'00_abs_errors.txt'
	'00_rel_errors.txt'
	'00_NSE_values.txt'
	'00_theta_values.txt'
	'00_PRI_values.txt'
	'00_linfinity_values.txt'
	'00_RMSE_values.txt'
)

files=(
	'00_abs_errors.txt'
	'00_rel_errors.txt'
	'00_theta_values.txt'
	'00_more_info.txt'
)

initial_batch=$1
let batches_num=initial_batch+4

for i in $(seq $initial_batch $batches_num); do
	save_path=${path_header}/batch_$i
	#rm -r ${save_path}
	echo ${save_path}
	for f in "${files[@]}"; do
		for model_e in "${model_estimators[@]}"; do
			path=${path_header}/batch_${i}/results/${model_e}/
			scp -r -P 2200 esau@130.206.30.27:${path}${f} ${path}
		done
	done
done
