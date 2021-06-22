#!/bin/bash

path_header='/home/esau/tfm/Tests/PlaneModelND' # LineModelND CircleModel EllipseModel PlaneModelND HomographyModel

# usage bash files_download_test.sh batch_idx test_idx model_e
# e.g. bash files_download_test.sh 1 1 msac
test_batches=(
	'batch_'$1
	)


model_estimators=(
	$3
	)

tests=(
	$2
)

for batch in "${test_batches[@]}"; do
	for id in "${tests[@]}";do
		path=${path_header}/${batch}
		echo ${path}
		yaml_file=${path_header}/${batch}/yaml/test_${id}.yaml

		mkdir -p ${path_header}/${batch}/yaml/
		scp -r -P 2200 esau@130.206.30.27:${yaml_file} ${path_header}/${batch}/yaml/

		noisy_data_path=${path_header}/${batch}/data/test_${id}.txt
		mkdir -p ${path_header}/${batch}/data/
		scp -r -P 2200 esau@130.206.30.27:${noisy_data_path} ${path_header}/${batch}/data/

		noisy_data_path=${path_header}/${batch}/data/test_${id}_inliers.txt
		scp -r -P 2200 esau@130.206.30.27:${noisy_data_path} ${path_header}/${batch}/data/

		noisy_data_path=${path_header}/${batch}/data/test_${id}_proj1.txt
		scp -r -P 2200 esau@130.206.30.27:${noisy_data_path} ${path_header}/${batch}/data/

		noisy_data_path=${path_header}/${batch}/data/test_${id}_proj2.txt
		scp -r -P 2200 esau@130.206.30.27:${noisy_data_path} ${path_header}/${batch}/data/

		noisy_data_path=${path_header}/${batch}/data/test_${id}_correspondences.txt
		scp -r -P 2200 esau@130.206.30.27:${noisy_data_path} ${path_header}/${batch}/data/

		for model_e in "${model_estimators[@]}"; do
			file_name=${model_e}/test_${id} 
			inliers=${path_header}/${batch}/results/${file_name}_inliers.txt
			scores=${path_header}/${batch}/results/${file_name}_best_scores.txt
			model_params=${path_header}/${batch}/results/${model_e}/test_${id}_params.txt
			mkdir -p ${path_header}/${batch}/results/${model_e}/
			scp -r -P 2200 esau@130.206.30.27:${inliers} ${path_header}/${batch}/results/${model_e}/
			scp -r -P 2200 esau@130.206.30.27:${model_params} ${path_header}/${batch}/results/${model_e}/
			scp -r -P 2200 esau@130.206.30.27:${scores} ${path_header}/${batch}/results/${model_e}/
		done
	done
done

#scp -r -P 2200 esau@130.206.30.27:/home/esau/tfm/Tests/EllipseModel/batch_00/ ${path}
#scp -r -P 2200 esau@130.206.30.27:/home/esau/tfm/Tests/EllipseModel/batch_01/ ${path}
