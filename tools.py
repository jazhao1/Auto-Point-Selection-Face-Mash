from imutils import face_utils
import numpy as np
import dlib
import cv2

from skimage.draw import polygon
def write(lines, name):
	outF = open(name + ".txt", "w")
	for line in lines:
		line = np.array_str(line)
		# write line to output file
		outF.write(line)
		outF.write("\n")
	outF.close()
	# np.savetxt(name+'.txt', lines, delimiter=',') 

def show(image, points=[]):
	clone = image.copy()

	if points != []:
		for (x, y) in points:
			cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
	cv2.imshow("Image",clone)  
	cv2.waitKey(0)

def p_show(img):
    plt.imshow(img, cmap="Greys_r")
    plt.show()

def get_shape(image, detector, predictor):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale image
	rects = detector(gray, 1)
	# index out the first face 
	rect = list(rects)[0]


	# determine the facial landmarks for the face region, then
	# convert the landmark (x, y)-coordinates to a NumPy array
	shape = predictor(gray, rect)
	facial_points = face_utils.shape_to_np(shape)

	height, width, __ = np.shape(image) 
	corner_points = np.array([[0,0], [0, height], [width, 0], [width, height]])
	all_points = np.append(facial_points, corner_points, axis=0)
	show(image, all_points)

	return all_points
def createMask(triangle,height, width):
    img = np.zeros((height, width), dtype=np.uint8)
    c = triangle[:,0]
    r = triangle[:,1]
    rr, cc = polygon(r, c)
    img[rr,cc] = 1
    return img

def computeAffine(tri1_pts, tri2_pts):
    ones = np.array([[1,1,1]])
    source = np.transpose(tri1_pts)
    source = np.concatenate((source, ones), axis=0)
    inv_source = np.linalg.inv(source)
    
    dest = np.transpose(tri2_pts)
    dest = np.concatenate((dest, ones), axis=0)
    return np.dot(dest,inv_source) 

def getMaskAffines(tri_pts, mid_tri_pts, height, width):
    mid_tri_pts_masks = []
    mid_tri_pts_affine = []
    ones = np.array([[1,1,1]])

    for i in range(mid_tri_pts.shape[0]):
        curr_tri = tri_pts[i]
        curr_mid_tri = mid_tri_pts[i]

        curr_mask = createMask(curr_mid_tri,height, width)
        curr_affine = computeAffine(curr_tri, curr_mid_tri)
        curr_affine = np.linalg.inv(curr_affine)

        mid_tri_pts_masks.append(curr_mask)
        mid_tri_pts_affine.append(curr_affine)
    return  mid_tri_pts_masks, mid_tri_pts_affine

def getMidImage(img, mid_tri_pts, mid_tri_pts_masks, mid_tri_pts_affine, height, width): 
    mid_img = np.zeros((height,width,3))
    for r in range(height):
        for c in range(width):
            for i in range(mid_tri_pts.shape[0]):
                curr_mask = mid_tri_pts_masks[i]
                curr_affine = mid_tri_pts_affine[i]
                if curr_mask[r,c] == 1:
                    curr_dest = np.transpose(np.array([[c,r,1]]))
                    curr_source = np.dot(curr_affine, curr_dest)
                    ori_r = int(curr_source[1,0])
                    ori_c =  int(curr_source[0,0])
                    ori_pxl = img[ori_r, ori_c]
                    mid_img[r,c] = ori_pxl
                    break
    return mid_img


