from estimation.fit import PlaneModelND # model class
from estimation.estimators import RANSAC, FMR # estimators
from estimation.fuzzy_metrics import M2 # fuzzy metrics
from dataset_generation.PlaneModelND import _get_model_dataset # to plot results 
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    
    # read data
    data = np.loadtxt('2d_line_data.txt')
    ground_truth_params = [[0.0, 0.0], [-0.32665296,  0.94514435]] # [[origin_x, origin_y], [n_x, n_y]] where `n` is normal vector to the 2D line
    true_outlier_ratio = 0.4

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
    np.set_printoptions(precision=3)
    print('Ground truth: ', ground_truth_params)
    print('TLS: ', model.params)
    print('RANSAC: ', ransac_robust_model.params)
    print('FMR4_M2: ', fmr_robust_model.params)

    gt_line_data = _get_model_dataset(ground_truth_params, model_samples = 10, model_bbox = 25)
    tls_line_data = _get_model_dataset(model.params, model_samples = 10, model_bbox = 25)
    ransac_line_data = _get_model_dataset(ransac_robust_model.params, model_samples = 10, model_bbox = 25)
    fmr_line_data = _get_model_dataset(fmr_robust_model.params, model_samples = 10, model_bbox = 25)

    # plot results
    fig, ax = plt.subplots()
    ax.scatter(data[:,0], data[:,1], s = 4.0, color = 'black')
    ax.plot(gt_line_data[:,0], gt_line_data[:,1], label = 'Original model', linewidth = 2.0)
    ax.plot(tls_line_data[:,0], tls_line_data[:,1], label = 'TLS', linewidth = 2.0)
    ax.plot(ransac_line_data[:,0], ransac_line_data[:,1], label = 'RANSAC', linewidth = 2.0)
    ax.plot(fmr_line_data[:,0], fmr_line_data[:,1], label = 'FMR4_M2', linewidth = 2.0)
    
    leg = plt.legend(loc='best', fontsize = 12, borderaxespad=0, framealpha=1, fancybox = False, handlelength = 1.25)
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(0.5)

    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.show()