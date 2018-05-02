import os
import sys
import time
import numpy as np 
import cPickle
from scipy.misc import imsave 

from skimage import io
from PIL import Image 
from skimage.morphology import label
from skimage.measure import regionprops
from skimage.exposure import histogram 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KDTree
from sklearn import metrics

def check(x, y, z):
	if x < 0 or x >= 64:
		return False
	if y < 0 or y >= 255:
		return False 
	if z < 0 or z >= 384:
		return False
	return True

def process(image, mask, path, image_filename, mask_filename):
	try:
		assert image.shape[0] == 64 and image.shape[1] == 245 and image.shape[2] == 384
		assert mask.shape[0] == 64 and mask.shape[1] == 245 and mask.shape[2] == 384
	except AssertionError:
		print 'Image Shape Error.'
		return 

	label_img = label(mask)
	prop = regionprops(label_img)
	max_area = 0
	for i in range(len(prop)):
		if prop[i]['area']>max_area:
			x, y, z = prop[0]['centroid']
			max_area = prop[i]['area']

	c_x, c_y, c_z = int(x), int(y), int(z)

	filename = image_filename.replace("_Denoised_resized.tiff", "")+".txt"
	print path, filename

	file = open(os.path.join(path, filename), "w")
	for i in range(c_x-16, c_x+16):
		for j in range(c_y-32, c_y+32):
			for k in range(c_z-32, c_z+32):
				if not check(i-1, j-1, k-1) and not check(i+1, j+1, k+1):
					continue 

				feat = []
				for a in range(i-1, i+2):
					for b in range(j-1, j+2):
						for c in range(k-1, k+2):
							feat.append(image[a][b][c])

				if len(feat) ==  27:
					#print 'Yo. Done.'
					if mask[i][j][k] > 0:
						file.write("1 ")
					else:
						file.write("0 ")

					for m in range(len(feat)):
						file.write(str(feat[m]) + " ")
					file.write("\n")
	file.close()


def scan_files(filepath):
	count = 0
	for path, subdirs, files in os.walk(filepath):
		for filename in files:
			f = str(filename)
			if f.endswith("resized.tiff") and f.startswith("BIN"):
				image_file = f.replace("BIN", "CBCT")
				image_file = image_file.replace("_resized.tiff", "_Denoised_resized.tiff")
				mask = io.imread(os.path.join(path, f))
				if os.path.exists(os.path.join(path, image_file)):
					img = io.imread(os.path.join(path, image_file))
					process(img, mask, path, image_file, f)

def get_vectors(filepath):
	X = []
	label = []
	for path, subdirs, files in os.walk(filepath):
		for filename in files:
			if filename.startswith("CBCT") and filename.endswith(".txt"):
				file = open(os.path.join(path, filename))
				for line in file:
					line = line.rstrip().split()
					label.append(int(line[0]))
					vec = line[1:]
					vec = [int(x) for x in vec]
					X.append(vec)
				file.close()
	return X, label

def get_accuarcy(X_train, y_train, X_test, y_test):
	
	start = time.time()
	K_value = 5
	tree = KDTree(X_train)
	_, ind = tree.query(X_test, k=K_value)
	y_pred = []

	for i in range(len(ind)):
		y = 0
		for j in range(len(ind[i])):
			y += y_train[ind[i][j]]
		if y>2:
			y=1
		else:
			y = 0
		y_pred.append(y)

	with open('kdtree.pkl', 'wb') as fid:
		cPickle.dump(tree, fid)
	
	print("k-Value: {} Accuracy is: {} F1 Score: {} Precision: {} Recall: {} Time Taken: {}".format(K_value, metrics.accuracy_score(y_test, y_pred)*100,
	 metrics.f1_score(y_test, y_pred,  average='macro'), metrics.precision_score(y_test, y_pred,  average='macro'), metrics.recall_score(y_test, y_pred,  average='macro'), time.time()-start))
	

def deploy_model(filename):
	with open(filename+'.pkl', 'rb') as f:
		tree = cPickle.load(f)
	return tree

def train():
	start = time.time()
	X, label = get_vectors('/usr2/Medical/LungData')
	print len(X)
	print('Total time taken: {}'.format(time.time()-start))
	X_train, y_train = X[: 500000], label[:500000]
	#X_train, y_train = X[: 5], label[:5]
	X_test, y_test = X[ int(0.7*len(X))+1:], label[int(0.7*len(label))+1:]
	get_accuarcy(X_train, y_train, X_test, y_test)

def test(path, filename, model):
	test_image = io.imread(os.path.join(path, filename))
	print test_image.shape
	out = np.zeros_like(test_image, dtype=np.uint8)

	for i in range(2, test_image.shape[0]-2):
		print i
		for j in range(2, test_image.shape[1]-2):
			for k in range(2, test_image.shape[2]-2):
				if not check(i-2, j-2, k-2) and not check(i+2, j+2, k+2):
					continue

				feat_all = []
				for a in range(i-2, i+3):
					for b in range(j-2, j+3):
						for c in range(k-2, k+3):
							feat_all.append(test_image[a][b][c])

				feat = []


				feat.append(test_image[i-2][j-2][k-2])
				feat.append(test_image[i-2][j-2][k+2])
				feat.append(test_image[i+2][j-2][k-2])
				feat.append(test_image[i+2][j-2][k+2])
				feat.append(test_image[i-2][j+2][k-2])
				feat.append(test_image[i-2][j+2][k+2])
				feat.append(test_image[i+2][j+2][k-2])
				feat.append(test_image[i+2][j+2][k+2])
				feat.append(test_image[i+2][j][k])
				feat.append(test_image[i][j+2][k])
				feat.append(test_image[i-2][j][k])
				feat.append(test_image[i][j-2][k])
				feat.append(test_image[i][j][k-2])
				feat.append(test_image[i][j][k+2])

				if len(feat) == 14 and np.max(feat) == 0:
					continue

				#LRT
				feat.append(np.argsort(feat_all)[62])


				if len(feat) == 15:
					y_pred = model.predict([feat])
					out[i][j][k] = y_pred[0]*255
					#print i, j, k, np.min(feat)

	#for i in range(out.shape[0]):
	#	imsave('output/output_' + str(i) + '.tiff', out[i])
	
	io.imsave(path+'/'+filename.replace(".tiff", "_seg.tiff"), out, plugin='tifffile', compress = 1)
	print np.max(out), out.shape


if __name__ == '__main__':


	model = deploy_model('rf_LRT')
	for path, subdirs, files in os.walk('../seg-map/'):
		for filename in files:
			if filename.endswith("_loc.tiff") and not os.path.isfile(os.path.join(path, filename.replace(".tiff", "_seg.tiff"))):
				print filename
				start = time.time()
				test(path, filename, model)
				print time.time()-start