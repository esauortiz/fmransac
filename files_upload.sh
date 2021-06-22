#!/bin/bash

#path_header='/home/esau/tfm/codigo_fuente/python'
path_header='/home/esau/tfm/codigo_fuente/python'


files=(
	'myfit.py'
	'runtest.sh'
	'myransac.py'
	'gen_yaml_master_file.py'
	#'gen_data_HomographyModel.py'
	'test_results.py'
	'test_estimate_model.py'
	'gen_yaml_file.py'
	)


for f in "${files[@]}"; do
	
	path=${path_header}/${f}
	scp -P 2200 ${path} esau@130.206.30.27:/home/esau/tfm/codigo_fuente/python

done
