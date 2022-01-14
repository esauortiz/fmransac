## Basic estimation example

This example performs 2D line estimations with TLS, RANSAC and FM-Based RANSAC variant (4th variant with fuzzy metric M2). Results are prompted with ```[origin, normal_vector]``` format. Moreover, estimated 2D lines are plotted alongside the ground truth line and the 2D source points. A successful execution of the ```example.py``` script should generate the following figure (although the result may vary because of different sequence of minimal sample sets which have been selected):

![](https://github.com/esauortiz/fmransac/blob/master/example/fig/example.png)

Basic requirements could be found in [basic_requirements.txt](https://github.com/esauortiz/fmransac/blob/master/basic_requirements.txt) and ```PYTHONPATH``` should be configured to include the path to the ```fmransac/scripts``` folder. Help to configure ```PYTHONPATH``` in Windows could be found in this video https://www.youtube.com/watch?v=A7E18apPQJs.