from pathlib import Path
from test_configuration.utils import _read_yaml
from dataset_formatting.HomographyModel.utils import convert_pts_to_keypoints, read_validation, show_keypoints
import numpy as np
import cv2 as cv
import sys
  

def residuals(H, src_pts, dst_pts):
    """ Apply homography to all correspondences, 
        return error for each transformed point. """

    def normalize(points):
        """ Normalize a collection of points in 
            homogeneous coordinates so that last row = 1. """

        for row in points:
            row /= points[-1]
        return points

    # transform fp
    src_transformed = np.dot(H, src_pts.T)
    # normalize hom. coordinates
    src_transformed = normalize(src_transformed)
    
    # compute the reprojection error
    residuals = np.sqrt(np.sum((dst_pts-src_transformed.T)**2,axis=1))
    return residuals
    
if __name__ == '__main__':

    # general params
    dataset_label = sys.argv[1]
    group_id = sys.argv[2]

    # directories and constants
    #DATASET_PATH = f'/media/esau/hdd_at_ubuntu/fm_packages/fm_matching/tests/datasets/{dataset_label}'
    DATASET_PATH = f'/home/esau/tfm/datasets/HomographyModel/{dataset_label}'
    GROUP_PATH = f'/home/esau/tfm/tests/HomographyModel/00_batch_groups/{group_id}'
    BATCH_PATH = '/home/esau/tfm/tests/HomographyModel'
    MIN_MATCH_COUNT = 4

    # group params
    group_params = _read_yaml(f'{GROUP_PATH}/batch_group_params.yaml')
    initial_batch_id = group_params['group_params']['initial_batch_id']
    n_batches = group_params['group_params']['n_batches']
    png_figures = {str(f).split('/')[-1][:-5] : 'png' for f in Path(DATASET_PATH).glob('**/*.png')}
    jpg_figures = {str(f).split('/')[-1][:-5] : 'jpg' for f in Path(DATASET_PATH).glob('**/*.jpg')}
    figures_dict = {**png_figures, **jpg_figures}

    # format data for each image pair
    for figure_label, idx in zip(figures_dict, range(n_batches)):
        figure_ext = figures_dict[figure_label]
        """
        # load image pair
        fname1 = f'{DATASET_PATH}/{figure_label}A.{figure_ext}'
        fname2 = f'{DATASET_PATH}/{figure_label}B.{figure_ext}'
        img1 = cv.imread(fname1,0) # trainImage
        img2 = cv.imread(fname2,0) # queryImage
        # Initiate SIFT detector
        sift = cv.SIFT_create(nfeatures = 10000, contrastThreshold = 0.01, edgeThreshold = 40)
        #surf = cv.SURF_create()
        #brief = cv.xfeatures2d.BriefDescriptorExtractor_create()

        # from keypoints to descriptors, but descriptors are conditioned to manually set kp.size
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
    
        # matching w/ brute force
        #bf = cv.BFMatcher_create()
        #matches = bf.match(des1, des2)

        # matching with FLANN
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)

        flann = cv.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)

        good = []
        pts1 = []
        pts2 = []

        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.6*n.distance:
                good.append(m)
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)
        
        matches = good

        if len(matches)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ])
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ])
        else:
            print( "Not enough matches are found - {}/{}".format(len(matches), MIN_MATCH_COUNT) )
        """
        # keypoints and its descriptors from ground truth
        kp1_ori, kp2_ori, src_pts, dst_pts, M_ori = read_validation(DATASET_PATH, figure_label, as_keypoints = False)

        np.savetxt(f'{BATCH_PATH}/batch_{initial_batch_id + idx}/tests_params/original_params.txt', M_ori)
        np.savetxt(f'{BATCH_PATH}/batch_{initial_batch_id + idx}/tests_params/kp1.txt', kp1_ori)
        np.savetxt(f'{BATCH_PATH}/batch_{initial_batch_id + idx}/tests_params/kp2.txt', kp2_ori)
        np.savetxt(f'{BATCH_PATH}/batch_{initial_batch_id + idx}/datasets/src_pts.txt', src_pts)
        np.savetxt(f'{BATCH_PATH}/batch_{initial_batch_id + idx}/datasets/dst_pts.txt', dst_pts)

        #show_keypoints(img1, convert_pts_to_keypoints(kp1_ori))
        #show_keypoints(img2, convert_pts_to_keypoints(kp2_ori))
        
        """
        # check kornia homography estimation
        kp1_ori = kp1_ori[:,:2]
        kp2_ori = kp2_ori[:,:2]
        import kornia
        import torch
        kp1_ori = torch.from_numpy(kp1_ori.reshape(1,*kp1_ori.shape))
        kp2_ori = torch.from_numpy(kp2_ori.reshape(1,*kp2_ori.shape))
        M_est =  kornia.geometry.homography.find_homography_dlt(kp1_ori, kp2_ori) # shape (1, 3, 3)
        M_est = M_est.cpu().detach().numpy()
        M_est = M_est[0]
        print(M_est)
        print(M_ori)
        """

        print(f'Data from `{figure_label}` image pair has been formated')