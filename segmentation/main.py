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
from mahotas.features import haralick
from skimage.feature import hog 
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def check(x, y, z):
	if x < 0 or x >= 64:
		return False
	if y < 0 or y >= 255:
		return False 
	if z < 0 or z >= 384:
		return False
	return True


def process_hog(image, mask, path, image_filename, mask_filename):
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

	filename = image_filename.replace("_Denoised_resized.tiff", "")+"_hog.txt"
	print path, filename

	X1 = []
	X0 = []

	file = open(os.path.join(path, filename), "w")
	for i in range(c_x-16, c_x+16):
		for j in range(c_y-32, c_y+32):
			for k in range(c_z-32, c_z+32):
				if not check(i-8, j-8, k) and not check(i+8, j+8, k):
					continue 

				mini_img = image[i-8: i+8, j-8: j+8, k:k+1]
				if mini_img.shape[0] != 16 or mini_img.shape[1] != 16 or mini_img.shape[2] != 1:
					continue

				feat, _ = hog(mini_img.reshape(16, 16), orientations=4, pixels_per_cell=(8, 8),
                    cells_per_block=(1, 1), visualise=True)

				if mask[i][j][k] > 0:
					X1.append(feat)
				else:
					X0.append(feat)
	n = min(len(X1), len(X0), 5000)

	for i in range(n):
		if len(X1[i]) == 16:
			file.write("1 ")
			for m in range(len(X1[i])):
				file.write(str(X1[i][m]) + " ")
			file.write("\n")

		if len(X0[i]) == 16:
			file.write("0 ")
			for m in range(len(X0[i])):
				file.write(str(X0[i][m]) + " ")
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
					process_hog(img, mask, path, image_file, f)

def get_vectors(filepath):
	X = []
	label = []
	for path, subdirs, files in os.walk(filepath):
		for filename in files:
			#if filename.startswith("CBCT") and filename.endswith("_hog.txt"):
			if filename.startswith("CBCT") and filename.endswith(".txt") and "_" not in filename:
				file = open(os.path.join(path, filename))
				print filename
				for line in file:
					line = line.rstrip().split()
					label.append(int(line[0]))
					vec = line[1:]
					vec = [float(x) for x in vec]
					X.append(vec)
				file.close()
	return X, label

def get_accuarcy(X_train, y_train, X_test, y_test):
	
	start = time.time()
	K_value = 5
	knn = KNeighborsClassifier(n_neighbors=K_value)
	knn.fit(X_train, y_train)
	y_pred = knn.predict(X_test)
	'''
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
	'''
	with open('knn_hog.pkl', 'wb') as fid:
		cPickle.dump(knn, fid)
	
	print("k-Value: {} Accuracy is: {} F1 Score: {} Precision: {} Recall: {} Time Taken: {}".format(K_value, metrics.accuracy_score(y_test, y_pred)*100,
	 metrics.f1_score(y_test, y_pred,  average='macro'), metrics.precision_score(y_test, y_pred,  average='macro'), metrics.recall_score(y_test, y_pred,  average='macro'), time.time()-start))
	

def get_accuracy_svm(X_train, y_train, X_test, y_test):

	start = time.time()
	model = svm.SVC()
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)

	print("SVM: Accuracy is: {} F1 Score: {} Precision: {} Recall: {} Time Taken: {}".format(metrics.accuracy_score(y_test, y_pred)*100,
	 metrics.f1_score(y_test, y_pred,  average='macro'), metrics.precision_score(y_test, y_pred,  average='macro'), metrics.recall_score(y_test, y_pred,  average='macro'), time.time()-start))
	
def get_accuracy_rf(X_train, y_train, X_test, y_test):

	start = time.time()
	model = RandomForestClassifier()
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	
	with open('rf_LRT.pkl', 'wb') as fid:
		cPickle.dump(model, fid)
	
	print("RF: Accuracy is: {} F1 Score: {} Precision: {} Recall: {} Time Taken: {}".format(metrics.accuracy_score(y_test, y_pred)*100,
	 metrics.f1_score(y_test, y_pred,  average='macro'), metrics.precision_score(y_test, y_pred,  average='macro'), metrics.recall_score(y_test, y_pred,  average='macro'), time.time()-start))

def get_accuracy_mlp(X_train, y_train, X_test, y_test):

	start = time.time()
	model = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(8, 3), random_state=1)
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	
	with open('mlp_LRT.pkl', 'wb') as fid:
		cPickle.dump(model, fid)
	
	print("MLP: Accuracy is: {} F1 Score: {} Precision: {} Recall: {} Time Taken: {}".format(metrics.accuracy_score(y_test, y_pred)*100,
	 metrics.f1_score(y_test, y_pred,  average='macro'), metrics.precision_score(y_test, y_pred,  average='macro'), metrics.recall_score(y_test, y_pred,  average='macro'), time.time()-start))
	

if __name__ == '__main__':
	#scan_files('/home/somnath/Academics/MTP/LungData')
	start = time.time()
	X, label = get_vectors('/home/somnath/Academics/MTP/LungData')
	print('Time Taken {}'.format(time.time()-start))
	print('Number of items: {}'.format(len(X)))
	X_train, X_test, y_train, y_test = X[: int(0.8*len(X))], X[int(0.8*len(X)):], label[: int(0.8*len(X))], label[int(0.8*len(X)):]
	print len(X_train)
	get_accuracy_mlp(X_train, y_train, X_test, y_test)