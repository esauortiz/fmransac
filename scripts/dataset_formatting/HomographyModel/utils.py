from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import scipy.io

def read_validation(img_path_header, figure_label, as_keypoints = False):
    """ Returns validation N points in img1 and img2
        with shape (N, 3) as homogeneous coords
        and ground truth homography M from 
        .mat file (kp1 = M·kp2) 
        used for 'homogr' dataset found in http://cmp.felk.cvut.cz/data/geometry2view/index.xhtml
    """
    mat = scipy.io.loadmat(f'{img_path_header}/{figure_label}_vpts.mat')
    kp1 = mat['validation'][0][0][0][:3].T
    kp2 = mat['validation'][0][0][0][3:].T
    M = mat['validation'][0][0][2]
    if as_keypoints:
        kp1 = convert_pts_to_keypoints(kp1)
        kp2 = convert_pts_to_keypoints(kp2)
    return kp1, kp2, M

def show_matches(img1,kp1,img2,kp2,matches, save_figure = False, figure_label = None):
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                singlePointColor = None,
                flags = 2)
    img = cv.drawMatches(img1,kp1,img2,kp2,matches,None,**draw_params)
    if save_figure == True: plt.imsave(f'{figure_label}.png', img)
    else: plt.imshow(img, 'gray'),plt.show()

def show_keypoints(img,kp, save_figure = False, figure_label = None):
    img = cv.drawKeypoints(img, kp, img, color=(255,0,0))
    if save_figure == True: plt.imsave(f'{figure_label}.png', img)
    else: plt.imshow(img, 'gray'),plt.show()

def convert_pts_to_keypoints(pts, sizes = None): 
    kps = []
    if sizes is None:
       sizes = np.ones((len(pts),)) 
    if pts is not None: 
        kps = [ cv.KeyPoint(p[0], p[1], size=size) for p, size in zip(pts, sizes) ]
    return kps