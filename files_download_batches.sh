#!/bin/bash

path_header='/home/esau/tfm/Tests/PlaneModelND' # LineModelND CircleModel EllipseModel PlaneModelND

test_batches=(
	'batch_123'
	'batch_124'
	'batch_125'
	'batch_126'
)

for batch in "${test_batches[@]}"; do
	scp -r -P 2200 esau@fuzzymar:${path_header}/${batch} ${path_header}/
done

