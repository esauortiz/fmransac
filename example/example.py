from estimation.fit import PlaneModelND
from estimation.estimators import RANSAC, FMR
from estimation.fuzzy_metrics import M2
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    
    # read data
    data = np.loadtxt('2d_line_data.txt')
    ground_truth_params = [[0.0, 0.0], [-0.32665296,  0.94514435]]

    # model objects
    model = PlaneModelND()

    # fit model with all available data
    model.estimate(data)

    # robust estimation with RANSAC and FMR
        # configure estimators
    ransac = RANSAC(min_samples = 2,
                    residual_threshold = 2.0,
                    max_trials = 1000,
                    stop_probability = 0.99,
                    outlier_ratio = None)

    fmr4_m2 = FMR(  min_samples = 2,
                    residual_threshold = 2.0,
                    max_trials = 1000,
                    stop_probability = 0.99,
                    variant = 4,
                    fuzzy_metric = M2(n = 1, m = 1, theta = 2),
                    t_max = 25,
                    outlier_ratio = None)

        # run estimation
    ransac_robust_model, inliers, compatibilities, main_loop_iters, refinement_iters = ransac.run(data, PlaneModelND, seed = None)
    fmr_robust_model, inliers, compatibilities, main_loop_iters, refinement_iters = fmr4_m2.run(data, PlaneModelND, seed = None)

    # print results
    print('Ground truth: ', ground_truth_params)
    print('TLS: ', model.params)
    print('RANSAC: ', ransac_robust_model.params)
    print('FMR4_M2: ', fmr_robust_model.params)

    # plot results
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.axis('equal')
    ax.scatter(data[:,0], data[:,1])
    ax.set_ylim([-10,10])
    ax.set_xlim([-10,10])
    plt.show()