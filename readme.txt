data: 
	files .npy with hyperspectral data
	dataset_diseased_leafs.npy : hyperspectral data of 19 leafs
	dataset_diseased_three_leafs.npy :  hyperspectral data only 3 leafs
	dataset_health_leafs.npy : hyperspectra data of health regions

cubes: experiment of leafs
	Experimento-2017-03-15 17-13-02: experiment to test

lib: library to process hyperspectral data

external dependencies:
	scikit-learn: http://scikit-learn.org/stable/install.html
	scikit-image: http://scikit-image.org/
	numpy and scipy: https://scipy.org/install.html
	pandas: https://pandas.pydata.org/pandas-docs/stable/install.html
	tiffile: https://pypi.python.org/pypi/tifffile
	matplotlib: https://matplotlib.org/users/installing.html

run command:
	python leaf_classification.py ./data/dataset_diseased_leafs.npy ./data/dataset_health_leafs.npy

when you run demo, you are going to see the next menu:
Training
1. Classify with SVIs
2. Classify with PCA (ten components)
Select one choice

You have to choose a specific feature selection strategy.

Next, you have to choose how to see testing results:
1. KFold cross-validator
2. Other leafs

Training:
1. Spectral Vegetative Indexes are calculated for each pixel spectrum
2. it takes all pixel's spectrum and it applies PCA transform, it takes ten principal component
For each option, RBF Support Vector Machine is applied.

Testing:
1. it applies 5-fold cross-validation
2. you have to enter a path of specific cube

