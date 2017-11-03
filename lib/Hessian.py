from scipy.ndimage.filters import gaussian_filter1d
from skimage.feature import hessian_matrix_eigvals
import numpy as np

class Hessian(object):
	"""docstring for Hessian"""
	def __init__(self):
		super(Hessian, self).__init__()

	def estimate_derivatives(self,img,sigma):
		img = np.float32(img)
		rx = gaussian_filter1d(img,sigma=sigma,order=1,axis=1)
		ry = gaussian_filter1d(img,sigma=sigma,order=1,axis=0)

		rxx = gaussian_filter1d(img,sigma=sigma,order=2,axis=1)
		rxx = gaussian_filter1d(rxx,sigma=sigma,order=0,axis=0)

		ryy = gaussian_filter1d(img,sigma=sigma,order=2,axis=0)
		ryy = gaussian_filter1d(ryy,sigma=sigma,order=0,axis=1)

		rxy = gaussian_filter1d(img,sigma=sigma,order=1,axis=0)
		rxy = gaussian_filter1d(rxy,sigma=sigma,order=1,axis=1)

		rxx = np.float32(rxx)
		ryy = np.float32(ryy)
		rxy = np.float32(rxy)

		return rx,ry,rxx,ryy,rxy

	def calculate(self,img,sigma=2.0):
		img = np.float32(img)
		rx,ry,rxx,ryy,rxy = self.estimate_derivatives(img,sigma)
		img_max_eigen_values,img_min_eigen_values = hessian_matrix_eigvals(rxx,rxy,ryy)
		return img_max_eigen_values,img_min_eigen_values