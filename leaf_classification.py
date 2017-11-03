import sys
sys.path.append("./lib")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from HSCube import *
from sklearn.decomposition import PCA
from scipy.stats import zscore
from sklearn import svm
from sklearn.model_selection import KFold,StratifiedKFold,cross_val_score
from sklearn.metrics import jaccard_similarity_score
from sklearn.externals import joblib
from scipy import stats
import os

labels = {
	"Sana" : 0,
	"Est1" : 1,
	"Est2" : 2,
	"Est3" : 3,
	"Est4" : 4,
	"Est5" : 5,
	"Est6" : 6,
	"Dano" : 7,
}

def change_label(row):
	label = row['stage']
	if label in labels.keys():
		return int(labels[label])
	return label

def evaluate_wavelength(index,bin_spectral=2):
	return 0.000022*index*index*bin_spectral + 0.586*index*bin_spectral + 386.829

def pri(row):
	spectrum = row[0:520].values
	r_570 = np.mean(spectrum[147:160])
	r_531 = np.mean(spectrum[117:139])#525,550
	index = 1.0*(r_531-r_570)/(r_531+r_570)
	return index

def rgri(spectrum):
	spectrum = spectrum.values
	index = np.sum(spectrum[181:265])/np.sum(spectrum[97:181])
	return index

def draw_pixel(row,img_predicted,img_labeled):
	x = int(row['x'])
	y = int(row['y'])
	
	stage_predicted = row['stage_predicted']
	stage = row['stage']

	if stage_predicted == 1 :#enferma
		img_predicted[x,y] = (10,10,255)#amarillo
	elif stage_predicted == 0 :#sana
		img_predicted[x,y] = (255,10,10)#azul
	
	if stage == 1 :
		img_labeled[x,y] = (10,10,255)
	elif stage == 0:
		img_labeled[x,y] = (255,10,10)

def draw_leaf(group):
	img_predicted = np.zeros((210,210,3))
	img_labeled = np.zeros((210,210,3))
	
	group.apply(lambda x:draw_pixel(x,img_predicted,img_labeled),axis=1)

	fig, ax = plt.subplots(1,2,figsize=(10, 6))
	ax[0].set_title(group['idExperiment'].values[0] + " GT")
	ax[0].imshow(img_labeled,interpolation="nearest")
	ax[0].set_axis_off()
	ax[1].set_title(group['idExperiment'].values[0] + " Prediction")
	ax[1].imshow(img_predicted,interpolation="nearest")
	ax[1].set_axis_off()
	plt.tight_layout()
	plt.show()

def training(df,side,n_features,view_testing):
	df_ft = df[df.columns[n_features:]]
	df_label = df['stage']
	C = 1.0
	kf = StratifiedKFold(n_splits=5)
	clfs = []
	scores = []
	y_predicted = []
	test_indexes = []
	print "5-Fold cross-validation"
	for train_index, test_index in kf.split(df_ft,df_label):
		X_train, X_test = df_ft.iloc[train_index], df_ft.iloc[test_index]
		Y_train, Y_test = df_label.iloc[train_index], df_label.iloc[test_index]
		clf = svm.SVC(kernel='rbf', C=C)
		clf.fit(X_train,Y_train)
		Y_predicted = clf.predict(X_test)
		score = jaccard_similarity_score(Y_test,Y_predicted)
		scores.append(score)
		clfs.append(clf)
		y_predicted.append(Y_predicted)
		test_indexes.append(test_index)
		print "Supervised Classification: Side : %s, Accuracy: %.2f" %(side,score)
	#clf = svm.SVC(kernel='rbf', C=C)
	#results = cross_val_score(clf,df_ft,df_label,cv=30)
	#print "%.2f" % np.mean(results)
	best_accuracy = np.argsort(scores)[::-1][0]

	clf = clfs[best_accuracy]
	best_y_predicted = y_predicted[best_accuracy]
	test_index = test_indexes[best_accuracy]
	if view_testing:
		print "Yelow: regions with first stage"
		print "Cyan: regions healthy"
		df_test = df.iloc[test_index]
		print best_y_predicted
		df_test['stage_predicted'] = best_y_predicted
		df_test.groupby('idExperiment').apply(draw_leaf)
	return clf
	

def testing(dir_input,clfs):
	clf_right = clfs[0]
	clf_left = clfs[1]
	hype_cube = HSCube(dir_input=dir_input,type_cube = 'polder')
	mask = hype_cube.get_mask()
	rows = hype_cube.rows
	cols = hype_cube.cols
	rgri = stats.zscore(hype_cube.rgri())
	pri = stats.zscore(hype_cube.pri())
	data = np.column_stack((pri,rgri))

	labels_right = clf_right.predict(data)
	labels_left = clf_left.predict(data)
	
	labels_right = np.reshape(labels_right,(rows,cols))
	labels_left = np.reshape(labels_left,(rows,cols))
	
	rgb = hype_cube.get_rgb()

	img_predicted = np.zeros((rows,cols,3))
	for row,col in itertools.product(range(rows),range(cols)):
		background = mask[row,col]
		side = hype_cube.get_side_of_leaf(col,row,1,1)
		label = int(labels_left[row,col])
		if side == 'derecha':
			label = int(labels_right[row,col])
		else:
			label = int(labels_left[row,col])

		if label == 1:#estadio 1
			img_predicted[row,col] = (10,10,255)#amarillo
		if label == 0:#sana
			img_predicted[row,col] = [255,10,10]#azul
		if label == 7:
			img_predicted[row,col] = [10,255,10]#azul

	fig, ax = plt.subplots(1,2,figsize=(10, 6))
	ax[0].imshow(rgb,interpolation="nearest")
	ax[0].set_axis_off()
	ax[1].imshow(img_predicted,cmap='gray')
	ax[1].set_axis_off()
	plt.tight_layout()
	plt.show()


def training_by_pca(df):
	pca = PCA(n_components=10)
	data_r = pca.fit_transform(df[df.columns[0:520]])

def classify_with_svi(df,side,view_testing):
	df['pri'] = df.apply(pri,axis=1)
	df['rgri'] = df.apply(rgri,axis=1)
	df['pri'] = stats.zscore(df['pri'])
	df['rgri'] =  stats.zscore(df['rgri'])
	clf = training(df,side,-2,view_testing)#take the two last features
	#joblib.dump(clf, './outputs/svis_svm_' + side +'.pkl')
	return clf

def classify_with_full_spectra(df,side,view_testing):
	wv = [ i for i in range(520)]
	clf = training(df,side,wv,view_testing)
	return clf

def classify_with_pca(df,side,view_testing):
	spectrums = df[df.columns[0:520]]
	n_components = 2
	pca = PCA(n_components=n_components)
	data_r = pca.fit_transform(spectrums)
	for i in range(0,n_components):
		i_component = str(i + 1)
		df['pc' + i_component] = data_r[:,i]
	clf = training(df,side,-n_components,view_testing)#take ten last features
	return clf

def classification_process(df,option=1,view_results=True):
	#do classification process only to classes zero (health) and one (first stage)
	i = 0#health leaf
	j = 1#stage one
	k = 7#others

	sides = ['Derecho','Izquierdo']#sides of leafs: Derecho: Right, Izquierdo: Leaf
	clfs = []

	classify_functions = {
		1: classify_with_svi,
		2: classify_with_pca,
	}

	for side in sides:#execute classification for specified feature selection
		df_side = df.loc[df['side'] == side]
		df_side = df_side.loc[ (df_side['stage'] == i) | (df_side['stage'] == j)]
		clf = classify_functions[option](df_side,side,view_results)
		clfs.append(clf)

	return clfs

def main():

	dir_train = sys.argv[1]
	dir_train_health = sys.argv[2]

	sample_train = np.load(dir_train)
	sample_health = np.load(dir_train_health)

	x_train = np.vstack((sample_train,sample_health))

	wv = [ i for i in range(520)]
	features = wv + ['stage', 'side', 'x','y', 'idRegion', 'idExperiment']
	df = pd.DataFrame(data=x_train,columns = features)
	df['stage'] = df.apply(change_label,axis=1)
	df[df.columns[0:520]] = df[df.columns[0:520]].astype(float)
	#df.to_csv("dataset_all_leafs.csv")
	#df = pd.read_csv("data/dataset_all_leafs.csv",index_col=0)
	try:
		print "Training"
		print "1. Classify with SVIs"
		print "2. Classify with PCA (ten components)"
		option = input("Select one choice\n")
		option = int(option)

		print "Testing"
		print "1. Leaf into same dataset (KFold cross-validator) "
		print "2. Other leafs"
		test_option = input("Select one choice\n")
		test_option = int(test_option)
		
		if test_option == 1:
			clfs = classification_process(df,option,True)
		elif test_option == 2:
			dir_input = raw_input("Write path of experiment\n")
			if os.path.isdir(dir_input.rstrip('\n')):
				clfs = classification_process(df,option,False)
				testing(dir_input,clfs)
			else:
				print("Experiment does not exist")
	except Exception as e:
		print "Option incorrected ",e

#python leaf_classification.py ./data/dataset_diseased_leafs.npy ./data/dataset_health_leafs.npy
if __name__ == '__main__':
	main()