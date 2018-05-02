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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

IMG_DIM = 50

from skimage.transform import resize
def normalize(arr):
    arr_min = np.min(arr)
    return (arr-arr_min)/(np.max(arr)-arr_min)

def explode(data):
    shape_arr = np.array(data.shape)
    size = shape_arr[:3]*2 - 1
    exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
    exploded[::2, ::2, ::2] = data
    return exploded

def expand_coordinates(indices):
    x, y, z = indices
    x[1::2, :, :] += 1
    y[:, 1::2, :] += 1
    z[:, :, 1::2] += 1
    return x, y, z

def plot_cube(cube, angle=320):
    #cube = normalize(cube)
    
    facecolors = cm.viridis(cube)
    facecolors[:,:,:,-1] = cube
    facecolors = explode(facecolors)
    
    filled = facecolors[:,:,:,-1] != 0
    x, y, z = expand_coordinates(np.indices(np.array(filled.shape) + 1))

    fig = plt.figure(figsize=(30/2.54, 30/2.54))
    ax = fig.gca(projection='3d')
    ax.view_init(30, angle)
    ax.set_xlim(right=IMG_DIM*2)
    ax.set_ylim(top=IMG_DIM*2)
    ax.set_zlim(top=IMG_DIM*2)
    
    ax.voxels(x, y, z, filled, facecolors=facecolors)
    plt.show()

def dist(point1, point2, point3): # x3,y3 is the point
    px = point2[0]-point1[0]
    py = point2[1]-point1[1]
    pz = point2[2]-point1[2]

    something = px*px + py*py + pz*pz

    u =  ((point3[0] - point1[0])*px + (point3[1] - point1[1])*py + (point3[2] - point1[2])*pz)/float(something)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x = point1[0] + u*px
    y = point1[1] + u*py
    z = point1[2] + u*pz

    dx = x - point3[0]
    dy = y - point3[1]
    dz = z - point3[2]

    #dist = math.sqrt(dx*dx + dy*dy)
    dist = (dx*dx + dy*dy + dz*dz)
    return dist

def dist_cvx_hull(hull1, hull2):
	
	dist_ov = []
	for p in hull2:
		dists=[]
		for i in range(len(hull1)-1):
			dists.append(dist(hull1[i], hull1[i+1], p))
		dist_ov.append(min(dists))
	return min(dist_ov)

def connectedcomps(mask):

	mask = np.array(mask, dtype=np.uint8)

	label = measure.label(mask)
	props = measure.regionprops(label)

	area = []
	for i in range(len(props)):
		area.append(props[i].area)

	#x = np.sort(x)
	points = [[[0] for _ in range(1)] for _ in range(len(props))]
	cvx_hull_points = [[[0] for _ in range(1)] for _ in range(len(props))]

	for i in range(mask.shape[0]):
		for j in range(mask.shape[1]):
			for k in range(mask.shape[2]): 
				if label[i][j][k] == 0:
					continue
				
				id = label[i][j][k]-1
				if len(points[id][0]) == 1:
					points[id][0] = [i,j,k]
				else:
					points[id].append([i,j,k])

	for i in range(len(points)):
		try:
			hull = ConvexHull(points[i])
			for j in range(len(hull.points)):
				if j==0:
					cvx_hull_points[i][j] = points[i][hull.points[j]]
				else:
					cvx_hull_points[i].append(points[i][hull.points[j]])
		except:
			cvx_hull_points[i] = points[i]

	d = [10000000]*len(cvx_hull_points)
	for i in range(len(cvx_hull_points)):
		if props[i].area > 5000:
			for j in range(len(cvx_hull_points)):
				if props[j].area>5000:
					continue
				if i!=j:
					print 'Started Yo!', i, j
					y = dist_cvx_hull(cvx_hull_points[i], cvx_hull_points[j])
					d[i] = min(d[i], y)
					#Merge 'em (i&j)
					if y < 500 and y == d[i]: 
						for p in points[j]:
							label[p[0]][p[1]][p[2]] = i

	for i in range(mask.shape[0]):
		for j in range(mask.shape[1]):
			for k in range(mask.shape[2]): 
				if label[i][j][k] == 0:
					continue	
				id = label[i][j][k]-1
				if props[id].area<5000:
					mask[i][j][k] = 0
	return mask

def largestcc():

	img = io.imread('/home/somnath/Academics/MTP/src/seg-map/13-3131-CBCT0_loc.tiff')
	mask = io.imread('/home/somnath/Academics/MTP/src/seg-map/13-3131-CBCT0_loc_seg.tiff')

	img = np.array(img, dtype=np.uint16)
	mask = np.array(mask, dtype=np.uint8)
	
	label = measure.label(mask)
	props = measure.regionprops(label)

	for i in range(mask.shape[0]):
		for j in range(mask.shape[1]):
			for k in range(mask.shape[2]): 
				if label[i][j][k] == 0:
					continue	
				id = label[i][j][k]-1
				if props[id].area<1000:
					mask[i][j][k] = 0
	
	io.imsave('visualization/13-3131_CBCT0_output_seg_cc.tiff', mask, plugin='tifffile', compress = 1)

def connectedcomps_util(filepath):
	for path, subdirs, files in os.walk(filepath):
		for filename in files:
			if filename.endswith("_seg.tiff"):
				print filename
				start = time.time()
				mask = io.imread(os.path.join(path, filename))
				mask = connectedcomps(mask)
				io.imsave(os.path.join(path, filename.replace(".tiff", "_cc.tiff")), mask, plugin='tifffile', compress = 1)
				print time.time()-start

if __name__ == '__main__':
	connectedcomps_util('seg-map/')