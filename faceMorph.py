# USAGE
# py faceMorph.py -i images/strange.jpg -j images/tony.jpg

from scipy.spatial import Delaunay
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from tools import *

# from skimage import data
# from skimage.viewer import ImageViewer
# import matplotlib.pyplot as plt


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image1", required=True,
     help="path to input image 1")
ap.add_argument("-j", "--image2", required=True,
     help="path to input image 2")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# load the input images, resize it, and convert it to grayscale
im1 = cv2.imread(args["image1"])
im2 = cv2.imread(args["image2"])


im1 = imutils.resize(im1, width=500)
im2 = imutils.resize(im2, width=500)
if np.shape(im2) != np.shape(im1):
	print("Images not the same size. Potentially incorrect morph")

# get points for each images facial feature 
points1 = get_shape(im1, detector, predictor)
points2 = get_shape(im2, detector, predictor)

# average each corresponding points to get the average points
avgPoints = ((points1 + points2)/2)

# apply Delaunay in order to form triangluar areas of the image for morphing.
# get tri points to build Masks and Affine of the two images
tri = Delaunay(avgPoints)

ori_tri_pts = points1[tri.simplices]
dest_tri_pts = points2[tri.simplices]
mid_tri_pts = avgPoints[tri.simplices]

width, height  = np.shape(im1)[1], np.shape(im2)[0]

mid_tri_pts_masks1, mid_tri_pts_affine1 = getMaskAffines(ori_tri_pts, mid_tri_pts, height, width)
mid_tri_pts_masks2, mid_tri_pts_affine2 = getMaskAffines(dest_tri_pts, mid_tri_pts, height, width)

# using the mask and affine, construct middle image
mid_img1 = getMidImage(im1,mid_tri_pts, mid_tri_pts_masks1, mid_tri_pts_affine1, height, width) 
mid_img2 = getMidImage(im2,mid_tri_pts, mid_tri_pts_masks2, mid_tri_pts_affine2, height, width)
mid_img = mid_img1/2 + mid_img2/2

show(mid_img)

cv2.imwrite("output/final.png", mid_img)