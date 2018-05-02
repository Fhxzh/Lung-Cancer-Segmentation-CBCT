import os
import sys
from skimage import io
import time
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage import data
from skimage.feature import match_template

import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure
from scipy.spatial import ConvexHull
import matplotlib.lines as mlines
from scipy.signal import medfilt
from scipy import ndimage

def newline(p1, p2):
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if(p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

    l = mlines.Line2D([xmin,xmax], [ymin,ymax])
    ax.add_line(l)
    return l

def load_scan(filepath):
	img = io.imread(filepath)
	#img = Image.open(filepath)
	img = np.array(img, dtype=np.int16)
	return img

def get_histogram(image):
	hist = [0]*65536
	for j in range(image.shape[0]):
		for k in range(image.shape[1]):
			hist[image[j][k]] += 1
	return hist

def get_points(mask):
	points = []
	for i in range(mask.shape[0]):
		for j in range(mask.shape[1]):
			if mask[i][j] > 0:
				points.append([i,j])
	points = np.array(points)
	return points

def apply_mask(image, mask):
	output = np.zeros_like(image).astype(np.int16)
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			if mask[i][j] > 0:
				output[i][j] = image[i][j]
	return output

def point_in_hull(point, hull, tolerance=1e-12):
    return all((np.dot(eq[:-1], point) + eq[-1] <= tolerance) for eq in hull.equations)

def get_image_from_hull(image, hull):
	out = np.zeros_like(image).astype(np.int16)
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			if point_in_hull((i,j), hull):
				out[i][j] = image[i][j]
	return out

def ROI(mask):
	output = np.zeros_like(mask).astype(np.uint8)

	n = np.max(mask)
	count = []
	for i in range(1,n+1):
		#if(np.sum(mask==i)>250):
		#	print np.sum(mask==i)
		count.append(np.sum(mask==i))

	'''
	indices = np.argsort(count)
	idx1 = indices[-1]
	idx2 = indices[-2]
	'''
	c = 0

	for i in range(mask.shape[0]):
		for j in range(mask.shape[1]):
			#if mask[i][j] == idx1+1 or mask[i][j] == idx2+1:
			if count[mask[i][j]-1] > 250:
				output[i][j] = 255
	return output

def threshold(img, min1, min2):
	mask = np.zeros_like(img, dtype=np.uint8)

	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if img[i][j]>=min1 and img[i][j]<=min2:
				mask[i][j] = 255
	return mask

def process(path, file):
	filepath = path+file 
	print filepath
	
	img = load_scan(filepath)
	mask = np.zeros_like(img, dtype=np.uint8)
	hist = get_histogram(img)

	min1 = 100
	min2 = 700

	mask = threshold(img, min1, min2)

	blobs_labels = measure.label(mask)
	out = ROI(blobs_labels)

	
	
	mask = ndimage.binary_closing(out, structure=np.ones((10,10))).astype(np.uint8)
	mask = ndimage.binary_fill_holes(mask, structure=np.ones((10,10))).astype(np.uint8)

	out2 = measure.label(mask)
	props = measure.regionprops(out2)

	max_x = 0
	min_x = 1000
	for i in range(len(props)):
		x,y = props[i].centroid
		print props[i].bbox
		max_x = max(max_x, y)
		min_x = min(min_x, y)

	line_X = (max_x+min_x)//2

	min_row = 100000
	min_col = 100000
	max_row = 0
	max_col = 0

	max_cluster = 0
	for i in range(len(props)):
		ar = props[i].area
		if ar>max_cluster:
			max_cluster = ar
			_, cy = props[i].centroid
			min_row, min_col, max_row, max_col = props[i].bbox

	print min_row, min_col, max_col, max_row
	lung = np.fliplr(mask[min_row-1:max_row, min_col-1:max_col])
	if cy<line_X:
		mask_ncc = mask[0:mask.shape[1], int(line_X):mask.shape[1]]
	else:
		mask_ncc = mask[0:mask.shape[1], 0:int(line_X)-1]
	#mask_ncc = mask
	result = match_template(mask_ncc, lung)
	ij = np.unravel_index(np.argmax(result), result.shape)
	x, y = ij[::-1]
	
	fig, ax1 = plt.subplots(1)

	ax1.imshow(mask, cmap='gray')
	# highlight matched region
	hlung, wlung = lung.shape
	rect = plt.Rectangle((x+line_X, y), wlung, hlung, edgecolor='y', facecolor='none')
	ax1.add_patch(rect)
	plt.show()

	points = get_points(mask)
	hull = ConvexHull(points)
	out1 = get_image_from_hull(img, hull)

	mask = threshold(out1, 1700, 60000)
	out = apply_mask(img, mask)

	out = Image.fromarray(out)
	mask = Image.fromarray(mask)

	p1 = [(max_x+min_x)/2, 0]
	p2 = [(max_x+min_x)/2, 245]

	plt.imshow(lung, cmap='gray')
	newline(p1, p2)
	plt.show()
	out.save('bin_thres.tif')


def process2(path, file, img):
	
	#img = load_scan(filepath)
	mask = np.zeros_like(img, dtype=np.uint8)
	hist = get_histogram(img)

	min1 = 100
	min2 = 700

	mask = threshold(img, min1, min2)

	blobs_labels = measure.label(mask)
	out = ROI(blobs_labels)
	
	mask = ndimage.binary_closing(out, structure=np.ones((10,10))).astype(np.uint8)
	mask = ndimage.binary_fill_holes(mask, structure=np.ones((10,10))).astype(np.uint8)

	start = time.time()
	out2 = measure.label(mask)
	props = measure.regionprops(out2)
	print time.time()-start

	max_x = 0
	min_x = 1000
	for i in range(len(props)):
		x,y = props[i].centroid
		#print props[i].bbox
		max_x = max(max_x, y)
		min_x = min(min_x, y)

	line_X = (max_x+min_x)//2

	left_mask = mask[:, 0: int(line_X)]
	right_mask = mask[:, int(line_X):]

	out1 = np.zeros_like(img[:, 0:int(line_X)], dtype=np.uint16)
	out2 = np.zeros_like(img[:, int(line_X):], dtype=np.uint16)

	start = time.time()
	points = get_points(left_mask)
	if len(points)>0:
		hull = ConvexHull(points)
		out1 = get_image_from_hull(img[:, 0: int(line_X)], hull)


	points = get_points(right_mask)
	if len(points) > 0:
		hull = ConvexHull(points)
		out2 = get_image_from_hull(img[:, int(line_X):], hull)
	print time.time()-start

	out = np.zeros_like(img, dtype=np.uint16)
	out = np.concatenate((out1, out2), axis=1)
	
	
	#fig, ax1 = plt.subplots(1)
	#ax1.imshow(out, cmap='gray')
	#plt.show()
	
	#out_img = Image.fromarray(out)
	return out

def process3(mask):
	out = np.zeros_like(mask, dtype=np.uint8)
	out2 = measure.label(mask)
	props = measure.regionprops(out2)

	for i in range(out2.shape[0]):
		for j in range(out2.shape[1]):
			for k in range(out2.shape[2]):
				if out2[i][j][k] ==0:
					continue
				if props[out2[i][j][k]-1].area > 100:
					out[i][j][k] = mask[i][j][k]
	return out

def process_all(filepath):

	for path, subdirs, files in os.walk(filepath):
		for filename in files:
			f = str(filename)
			if "mask" not in f:
				out = process2(path,f)
				matplotlib.image.imsave('../LungData/output_processed/'+f, out, cmap='gray')
				#out.save('../LungData/output_processed/'+f)
				#mask.save('masks/'+f)

def localize_all(filepath):

	for path, subdirs, files in os.walk(filepath):
		for filename in files:
			f = str(filename)
			if f.endswith("_Denoised_resized.tiff"):
				start = time.time()
				img = load_scan(os.path.join(path, f))
				out = np.zeros_like(img, dtype = np.uint16)
				print path+f 
				for i in range(img.shape[0]):
					out[i] = process2(path, filename, img[i])
				io.imsave(os.path.join(path, f.replace("_Denoised_resized.tiff", "_loc.tiff")), out, plugin='tifffile', compress = 1)

				print time.time()-start

if __name__ == '__main__':
	#process_all('../LungData/processed/')
	'''
	filepath = 'visualization/'
	filename = 'CBCT1_uc.tiff'
	maskname = 'BIN1.tiff'
	sl = 27
	img = load_scan(filepath+filename)[sl][80:325, 7:]
	mask_gt = load_scan(filepath+maskname)[sl][80:325, 7:]
	mask_pred = load_scan(filepath+'13-4146_CBCT1_output_seg_medfilt.tiff')[sl]

	out = np.zeros_like(img, dtype=np.uint8) 
	stacked_img_gt = np.stack((img,)*3, -1)
	stacked_img_pred = np.stack((img,)*3, -1)
	
	print stacked_img_pred.shape, stacked_img_gt.shape, mask_gt.shape
	for i in range(stacked_img_gt.shape[0]):
		for j in range(stacked_img_gt.shape[1]):
			if img[i][j] > 10 and mask_gt[i][j]>0:
				stacked_img_gt[i][j][0] = 0
				stacked_img_gt[i][j][1] = 255
				stacked_img_gt[i][j][2] = 0
			elif img[i][j] > 10:
				stacked_img_gt[i][j][0] = 255
				stacked_img_gt[i][j][1] = 0
				stacked_img_gt[i][j][2] = 0

	for i in range(stacked_img_pred.shape[0]):
		for j in range(stacked_img_pred.shape[1]):
			if mask_pred[i][j] > 0 and mask_gt[i][j]>0:
				stacked_img_pred[i][j][0] = 0
				stacked_img_pred[i][j][1] = 255
				stacked_img_pred[i][j][2] = 0
			elif mask_pred[i][j] > 0:
				stacked_img_pred[i][j][0] = 255
				stacked_img_pred[i][j][1] = 0
				stacked_img_pred[i][j][2] = 0
	
	#for i in range(len(img)):
	#	out[i] = process3(img[i])
	#out = process3(img)
	#out = ndimage.median_filter(stacked_img_pred, 3)
	#io.imsave('visualization/output_merged.tiff', stacked_img, plugin='tifffile', compress = 1)
	matplotlib.image.imsave(filepath+'img.png', img, cmap='gray')
	matplotlib.image.imsave(filepath+'output_prediction_medfilt.png', stacked_img_pred)
	matplotlib.image.imsave(filepath+'output_groundtruth.png', stacked_img_gt)
	
	'''
	
	filepath = '/home/somnath/Academics/MTP/src/visualization/'
	filename = 'CBCT3_Denoised_resized.tiff'
	img = load_scan(filepath+filename)
	out = np.zeros_like(img, dtype = np.uint16)

	print img.shape[0]
	#for i in range(img.shape[0]):
	#	out[i] = process2(filepath, filename, img[i])
	
	out = ndimage.median_filter(img, 3)
	io.imsave('visualization/13-4146_CBCT1_output_seg_medfilt.tiff', out, plugin='tifffile', compress = 1)
	
	#localize_all('../LungData/13-4146/')