from skimage import morphology
from skimage import feature
from skimage import filters
from skimage import exposure
from skimage import measure
from Hessian import *
from ImageUtils import *
import tifffile as tiff
from scipy import ndimage

class Mask(object):
	"""docstring for Mask"""
	def __init__(self,target):
		super(Mask, self).__init__()
		self.target = target

	def transform(self):
		pass

class BaseMask(Mask):
	"""docstring for Binary"""
	def __init__(self,target):
		Mask.__init__(self,target)
		self.target = target

	def transform(self):
		return self.target

class Decorator(Mask):
	"""docstring for Decorator"""
	def __init__(self,component):
		Mask.__init__(self,component.target)
		self.component = component

	def transform(self):
		pass

class PCBR(Decorator):
	"""docstring for Principal curvature-based region detector"""
	def __init__(self,component,scale=2.5):
		Decorator.__init__(self,component)
		self.scale = scale

	def transform(self):
		self.target = self.component.transform()
		hessian = Hessian()
		img_max_eigenvalues,img_min_eigenvalues = hessian.calculate(self.target,self.scale)
		utils = ImageUtils()
		return utils.max(img_max_eigenvalues,img_min_eigenvalues)

class Thresholding(Decorator):
	"""docstring for Binary"""
	def __init__(self,component,thresholding_method = filters.threshold_otsu ):
		Decorator.__init__(self,component)
		self.thresholding_method = thresholding_method
	
	def transform(self):
		self.target = self.component.transform()
		threshold_global_otsu = self.thresholding_method(self.target)
		img_thresholding = self.target > threshold_global_otsu
		return np.float32(img_thresholding*1.0)


class Skeletonization(Decorator):
	"""docstring for Skeletonization"""
	def __init__(self,component):
		Decorator.__init__(self,component)

	def transform(self):
		self.target = self.component.transform()
		self.target = self.target/self.target.max()
		skeleton = morphology.skeletonize(self.target)
		return np.uint8(255*skeleton)

class Closing(Decorator):
	"""docstring for Closing"""
	def __init__(self,component,size_selem=3,n_operations=2):
		Decorator.__init__(self,component)
		self.size_selem = size_selem
		self.n_operations = n_operations

	def transform(self):
		self.target = self.component.transform()
		selem = morphology.square(self.size_selem)
		for i in range(self.n_operations):
			self.target = morphology.closing(self.target, selem)
		return np.uint8(255*self.target)

class Openning(Decorator):
	"""docstring for Closing"""
	def __init__(self,component,size_selem=3,n_operations=2):
		Decorator.__init__(self,component)
		self.size_selem = size_selem
		self.n_operations = n_operations

	def transform(self):
		self.target = self.component.transform()
		selem = morphology.square(self.size_selem)
		for i in range(self.n_operations):
			self.target = morphology.openning(self.target, selem)
		return np.uint8(255*self.target)

class Dilation(Decorator):
	"""docstring for Dilation"""
	def __init__(self,component,size_selem=3,n_operations=2):
		Decorator.__init__(self,component)
		self.size_selem = size_selem
		self.n_operations = n_operations

	def transform(self):
		self.target = self.component.transform()
		selem = morphology.square(self.size_selem)
		for i in range(self.n_operations):
			self.target = morphology.dilation(self.target, selem)
		return np.uint8(255*self.target)

class Erosion(Decorator):
	"""docstring for Erosion"""
	def __init__(self,component,size_selem=3,n_operations=2):
		Decorator.__init__(self,component)
		self.size_selem = size_selem
		self.n_operations = n_operations

	def transform(self):
		self.target = self.component.transform()
		selem = morphology.square(self.size_selem)
		for i in range(self.n_operations):
			self.target = morphology.erosion(self.target, selem)
		return np.uint8(255*self.target)
			

class Filling(Decorator):
	"""docstring for Filling"""
	def __init__(self,component,min_size = 2,connectivity=100):
		Decorator.__init__(self,component)
		self.connectivity = connectivity
		self.min_size = min_size

	def transform(self):
		self.target = self.component.transform()
		selem = morphology.square(3)
		label_img, num_labels = ndimage.label(self.target)
		self.target = morphology.remove_small_objects(label_img,self.min_size,connectivity=self.connectivity,in_place=True)
		self.target = np.uint8(255*self.target)
		self.target[np.where((self.target > 0) & (self.target < 254 ))] = 255
		return np.uint8(self.target)

class FinderRectangle(object):
	"""docstring for FinderRectangle"""
	def __init__(self, target):
		super(FinderRectangle, self).__init__()
		self.target = target

	def get_max_rectangle(self,scale,size_selem):
		self.base_img = filters.gaussian(self.target,sigma=10.0) 
		img_binary = self.binary(self.base_img,scale,size_selem,False)

		rows,cols = img_binary.shape
		areas = []
		rectangles = []

		label_image = measure.label(img_binary)

		for region in measure.regionprops(label_image):
			rectangles.append(region.bbox)
			areas.append(region.area)
		
		rectangles = np.array(rectangles)
		index_sorted = np.argsort(np.array(areas))
		index_max = index_sorted[len(index_sorted)-1]
		max_rectangle = rectangles[index_max]