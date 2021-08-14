# FM-Based RANSAC (FMRANSAC)

## Abstract 
Robust model estimation is a recurring problem in application areas such as robotics and computer vision. This package proposes FMRANSAC, a model estimation approach that is based on the well known RANSAC algorithm, though modified taking inspiration from a notion of distance that arises in a natural way in fuzzy logic. In more detail, FMRANSAC makes use of a Fuzzy Metric (FM) within the main RANSAC loop to encode the compatibility of each sample to the current model/hypothesis. Further, once a number of hypotheses have been explored, we make use of the same fuzzy metric to refine the winning model. The incorporation of this fuzzy metric permits us to express the distance between two points as a kind of degree of nearness measured with respect to a parameter, which is very appropriate in the presence of the vagueness or imprecision inherent to noisy data. 

For a more in-depth description about RANSAC, MSAC and FMRANSAC, we refer the reader to the following papers and references therein:

A. Ortiz, E. Ortiz, J. J. Miñana, and O. Valero, “On the Use of Fuzzy Metrics for Robust Model Estimation: a RANSAC-based Approach ,” in _Proceedings of the International Work conference On Artificial Neural Networks_, Lecture Notes in Computer Science. Springer, 2021,
in press.

A. Ortiz, E. Ortiz, J. J. Miñana, and O. Valero, “Hypothesis Scoring and Model Refinement Strategies for FM-based RANSAC,” in _Proceedings of the Spanish Conference on Fuzzy Logic and Technologies_, Lecture Notes in Artificial Intelligence. Springer, 2021, in press.

J. J. Miñana, A. Ortiz, E. Ortiz, and O. Valero, “On the standard fuzzy metric: generalizations and application to model estimation ,” in _Proceedings of the Spanish Conference on Fuzzy Logic and Technologies_, Lecture Notes in Artificial Intelligence. Springer, 2021, in press.

## Description
This python package includes modules to:

- Generate synthetic data for hyperplane, ellipse and homography (similarity) models
- Estimate the aforementioned models with RANSAC, MSAC and the proposed FMRANSAC variants.
- Calculate estimation accuracy
- Plot the estimation result (dataset, original model and estimated model) [WIP]

## Installation

This package requires Python 3.6.9+ to run.

Install requirements.
```sh
pip install -r requirements.txt 
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