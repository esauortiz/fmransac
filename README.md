# FM-Based RANSAC (FMR)

## Abstract 
Robust model estimation is a recurring problem in application areas such as robotics and computer vision. This package proposes FMR, a model estimation approach that is based on the well known RANSAC algorithm, though modified taking inspiration from a notion of distance that arises in a natural way in fuzzy logic. In more detail, FMR makes use of a Fuzzy Metric (FM) within the main RANSAC loop to encode the compatibility of each sample to the current model/hypothesis. Further, once a number of hypotheses have been explored, we make use of the same fuzzy metric to refine the winning model. The incorporation of this fuzzy metric permits us to express the distance between two points as a kind of degree of nearness measured with respect to a parameter, which is very appropriate in the presence of the vagueness or imprecision inherent to noisy data. 

For a more in-depth description about RANSAC, MSAC and FMR, we refer the reader to the following papers and references therein:

A. Ortiz, E. Ortiz, J. J. Miñana, and Ó. Valero, [“On the Use of Fuzzy Metrics for Robust Model Estimation: A RANSAC-Based Approach,”](https://doi.org/10.1007/978-3-030-85030-2_14) Advances in Computational Intelligence. Springer International Publishing, pp. 165–177, 2021. 

A. Ortiz, E. Ortiz, J. J. Miñana, and Ó. Valero, [“Hypothesis Scoring and Model Refinement Strategies for FM-Based RANSAC,”](https://doi.org/10.1007/978-3-030-85713-4_10) Advances in Artificial Intelligence. Springer International Publishing, pp. 96–105, 2021.

J. J. Miñana, A. Ortiz, Esaú Ortiz, and Ó. Valero, [“On the standard fuzzy metric: generalizations and application to model estimation,”](https://doi.org/10.5281/zenodo.4964783) Jun. 2021.

## Description
This python package includes modules to:

- Generate synthetic data for hyperplane, ellipse and homography (similarity) models
- Estimate the aforementioned models with RANSAC, MSAC and the proposed FMR variants.
- Calculate estimation accuracy
- Plot the estimation result (dataset, original model and estimated model) [In development]

## Installation

This package requires Python 3.6.9+ to run.

Install requirements.
```sh
pip install -r requirements.txt 
```

Or install basic requirements to run a [example](https://github.com/esauortiz/fmransac/tree/master/example).
```sh
pip install -r basic_requirements.txt 
```

## Configure the estimation tests
All configurable estimation test parameters are placed in ```scripts/test_configuration/params``` folder.
- Choose a folder to save the estimation tests ```scripts/test_configuration/params/tests_path.yaml```
- Configure **tests batch group** parameters ```scripts/test_configuration/params/batch_group.yaml```
- Configure **model class** parameters ```scripts/test_configuration/params/model/model_class.yaml```
 
## Run estimation tests
Once **model class** and **tests batch group identifier** have been set up:
```sh
python scripts/run_tests.py model_class group_id
```

## Retrieve and save results
A table of results could be generated with:
```sh
python scripts/estimation/results/tabulate.py model_class group_id rows_labels metric stat_type
```
For example, running:

```sh
python scripts/estimation/results/tabulate.py PlaneModelND G1 outlier_ratio estimation_errors mean
```
will generate a table of results of PlaneModelND estimations which belongs to group G1 using outlier_ratio values as row labels. Table values are the mean of estimation errors (defined as the angle in degrees between ground truth normal vector and estimated normal vector in the case of PlaneModelND). Estimations with each estimator (RANSAC, MSAC or FMR variant) are grouped in columns. 

Not configuring neither **tests batch group** nor **model class**, running estimation tests and running the example script will (1) generate synthetic noisy datasets of straight 2D lines; (2) estimate straight 2D lines placed in the generated datasets; and (3) generate table of results with estimation errors computed as the angle between the estimated line and the ground truth line:

|outlier ratio|RANSAC|MSAC |FMR4_M2 |
|-------------|------|-----|--------|
|0.6          |4.96  |2.57 |1.33    |
|0.5          |2.81  |1.62 |1.01    |
|0.4          |1.87  |1.29 |0.85    |
|0.2          |0.96  |0.86 |0.66    |

showing that FM-based RANSAC is better at straight 2D line estimation in terms of accuracy with any outlier ratio between 0.2 and 0.6.

A visual representation of how our algorithm enhances estimation accuracy compared to RANSAC and MSAC is shown in the following figure, where the best (a) and worst (b) estimations (out of 500 estimations) with FMR4_M2 compared against MSAC are plotted.

![](https://github.com/esauortiz/fmransac/blob/master/doc/fig/ransac_msac_vs_fmr4.png)

Note that:

* The noise magnitude of the inliers (sigma) and the outlier ratio (omega) of the generated datasets are 1.0 and 0.4, respectively.
* Ground truth is plotted as grey lines and estimation as black lines.
* Inliers/Outliers for RANSAC and MSAC are plotted in blue/red.
* The degree of compatibility is shown with a grey scale, the darker the higher the compatibility.
