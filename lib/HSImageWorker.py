import os
import tifffile as tiff
import numpy as np
import re

class HSImageWorker(object):
	#Hiperspectral Image Worker: lee directorio y carga imagenes en estructura de datos

	def __init__(self, dir_input=None,spectral_range='all'):
		super(HSImageWorker, self).__init__()
		self.dir_input = dir_input
		self.set_spectral_range(spectral_range)

	def set_spectral_range(self,spectral_range_flag):
		if spectral_range_flag == 'all':
			self.spectral_range = np.array([ i for i in range(520)])
		elif spectral_range_flag == 'nir':
			self.spectral_range = np.array([ i for i in range(350,520)])
		else:
			self.spectral_range = np.array([ i for i in range(520)])

	def set_input_directory(self,dir_input):
		self.dir_input = dir_input

	def get_input_directory(self):
		return self.dir_input

	def read_image(self,dir):
		return tiff.imread(dir)

	def get_file_names(self):
		return self.files_names

	def get_file_names(self):#lee los nombres de los archivos del directorio cubo* 
		onlyfiles = [ (float(f.split('longitud')[1].split('.tif')[0]), os.path.join(self.dir_input,f)) for f in os.listdir(self.dir_input) if os.path.isfile(os.path.join(self.dir_input, f))]
		onlyfiles = np.array(sorted(onlyfiles,key=lambda tup: tup[0]))
		onlyfiles = onlyfiles[self.spectral_range]
		return np.array([ file for w,file in onlyfiles])

	def load(self):#lee las imagenes desde la camara thor
		if os.path.isdir(self.dir_input) and re.search("cubo",self.dir_input)!=None:
			images = np.array([])
			self.files_names = self.get_file_names()
			for file_name in self.files_names:
				img = np.array(self.read_image(file_name))
				images = np.dstack((images, img)) if images.size else img
			return images
		print "it can't find directory"
		return None

	def write_images(self,imgs,dir):
		freeimg.write_multipage(imgs,'myimages.tiff')