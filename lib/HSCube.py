import os
import numpy as np
from HSImageWorker import *
from ImageMask import *
import cv2

class HSCube(object):
	"""docstring for HSCube"""
	def __init__(self,dir_input=None,type_cube='normalizado',spectral_range_flag='all'):
		super(HSCube, self).__init__()
		self.type_cube = type_cube
		self.dir_root = dir_input
		self.leaf = None
		self.background = None
		self.mask = None
		if dir_input == None:
			return			
		if self.type_cube == 'raw':
			self.dir_input = os.path.join(dir_input,'cubo')
		if self.type_cube == 'normalizado':
			self.dir_input = os.path.join(dir_input,'cuboNormalizado')
		if self.type_cube == 'polder':
			self.dir_input = os.path.join(dir_input,'cuboNormalizadoPolder')
		if self.type_cube == 'snv':
			self.dir_input = os.path.join(dir_input,'cuboNormalizadoSNV')

		self.load_data(spectral_range_flag)
		
	def load_data(self,spectral_range_flag):
		self.image_worker = HSImageWorker(self.dir_input,spectral_range=spectral_range_flag)
		self.data = self.image_worker.load()
		if np.any(self.data):
			rows,cols,wv = self.data.shape
			self.rows = rows
			self.cols = cols
			self.num_pixels = self.rows * self.cols

	def get_wavelength(self,index,bin_spectral=2):
		return 0.000022*index*index*bin_spectral + 0.586*index*bin_spectral + 386.829

	def get_wavelengths(self):
		return [ self.get_wavelength(i) for i in range(520)]

	def get_spectrum_by_pixel(self,px,py,filter=False,window_size=11,order=3):
		spectral_range = self.data[py,px,:]
		if filter:
			return signal.savgol_filter(spectral_range,window_size,order)
		return spectral_range

	def get_spectrum_by_region(self,px,py,rx,ry,filter=False,window_size=11,order=3):
		region = self.data[py:py+ry,px:px+rx,:]
		rows,cols,n_wv = region.shape
		shape = (rows*cols,n_wv)#matrix size : num of spectral ranges vs num wavelengths
		matrix = region.reshape(shape)
		spectral_range = np.mean(matrix, axis=0)
		if filter:
			return signal.savgol_filter(spectral_range,window_size,order)
		return spectral_range

	def get_red(self):
		red = np.mean(self.data[:,:,182:307],axis=2)
		return red

	def get_ir(self):
		ir = np.mean(self.data[:,:,315:],axis=2)
		return ir

	def get_blue(self):
		blue = np.mean(self.data[:,:,12:97],axis=2)
		return blue

	def get_green(self):
		green = np.mean(self.data[:,:,98:181],axis=2)
		return green

	def get_rgb(self):
		util = ImageUtils()
		try:
			bgr_img = self.image_worker.read_image(join(join(self.dir_root,'fotoThor'),'output.tif'))
		except Exception as e:
			bgr_img = None
			print "Rgb Image cannot find"
		
		if bgr_img!= None:
			return bgr_img
		
		green = util.scale(self.get_green())
		red = util.scale(self.get_red())
		blue = util.scale(self.get_blue())
		bgr = [red, green, blue]

		bgr_img = cv2.merge(bgr)
		return bgr_img

	def get_background(self,binary_not=True):
		ir = self.get_ir()
		red = self.get_red()
		ndvi = 1.0*(ir-red)/(red + red)
		base_mask = BaseMask(ndvi)
		thresholding = Thresholding(base_mask)
		morph = Closing(thresholding,size_selem=6)
		filling = Filling(morph)
		erosion = Erosion(filling)
		background = erosion.transform()
		if binary_not:
			background = np.uint8(255*np.logical_not(background))
		
		return background

	def get_leaf(self,binary_not=False):
		ir = self.get_ir()
		base_mask = BaseMask(ir)
		hessian = PCBR(base_mask,scale=1.5)
		thresholding = Thresholding(hessian)
		morph = Closing(thresholding,size_selem=6)
		dilation = Dilation(morph)
		skeleton = Skeletonization(dilation)
		mask = skeleton.transform()
		if binary_not:
			background = np.uint8(255*np.logical_not(background))
		return mask

	def get_mask(self):
		self.leaf = self.get_leaf()
		self.background = self.get_background()

		self.mask =  np.uint8(self.background + self.leaf)/self.background.max()
		return self.background

	def get_side_of_leaf(self,px,py,rx,ry):
		if not(self.background.any()):
			self.background = self.get_background()
		regions = measure.regionprops(self.background)
		region = regions[0]
		c_py,c_px = region.centroid
		cols = range(px,px+rx)
		rows = range(py,py+ry)
		
		dist_col = []
		
		for row,col in itertools.product(rows,cols):
			dist_col.append(c_px - col)
		
		dist_col = np.mean(dist_col,axis=0)

		if dist_col < 0 :
			return 'derecha'
		else:
			return 'izquierda'

	def rgri(self):
		index =np.sum(self.data[:,:,181:265],axis=2)/np.sum(self.data[:,:,97:181],axis=2)
		index = index[~np.isnan(index)]
		return index[0:self.num_pixels]
	
	def pri(self):
		r_570 = np.mean(self.data[:,:,147:160],axis=2)
		r_531 = np.mean(self.data[:,:,117:139],axis=2)#525,550
		index = 1.0*(r_531-r_570)/(r_531+r_570)
		index = index[~np.isnan(index)]
		return index[0:self.num_pixels]

	def wi(self):
		r_900 = np.mean(self.data[:,:,398:439],axis=2)
		r_970 = np.mean(self.data[:,:,485:493],axis=2)
		index = 1.0*r_900/r_970
		index = index[~np.isnan(index)]

		return index[0:self.num_pixels]

class HSSyntheticCube(HSCube):
	"""docstring for HSSyntheticCube"""
	def __init__(self,raw_data):
		HSCube.__init__(self,dir_input=None)
		self.build(raw_data)

	def build(self,raw_data):
		print raw_data.shape

		real_num_pixels,wavelenghts = raw_data.shape
		mod = real_num_pixels%10
		num_pixels,wavelenghts = raw_data.shape
		self.num_pixels = num_pixels
		print mod

		if mod != 0:
			new_data = np.array([ 1 for i in range(520)])
			for i in range(10-mod):
				self.data = np.vstack((raw_data,new_data))

		num_pixels,wavelenghts = raw_data.shape
		rows = num_pixels/10
		cols = 10
		self.data = np.reshape(raw_data,(rows,cols,wavelenghts))
		self.rows = rows
		self.cols = cols
		print self.data