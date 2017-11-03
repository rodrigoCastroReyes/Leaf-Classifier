import numpy as np
import matplotlib.pyplot as plt
from math import floor,ceil,degrees,acos,asin,cos,sin,atan,sqrt,tan
import pandas as pd
from os.path import isfile, join
import itertools



class ImageUtils(object):
	"""docstring for ImageUtils"""
	def __init__(self):
		super(ImageUtils, self).__init__()

	def max(self,img_one,img_two):
		rows,cols = img_one.shape
		result = np.zeros((rows,cols),np.float32)

		for row,col in itertools.product(range(rows),range(cols)):
			result[row,col] = max(img_one[row,col],img_two[row,col])
				
		return result

	def show_img(self,img,label,vmin=0,vmax=0,mask=None):
		fig, ax = plt.subplots()#RdYlGn_r
		if mask != None:
			rows,cols = img.shape
			for pos in product(range(rows),range(cols)):
				if not(mask[pos]):
					img[pos] = 0.0
		if vmin == 0 and vmax == 0:
			p = ax.imshow(img,interpolation='nearest',cmap=plt.cm.spectral)
		else:
			p = ax.imshow(img,interpolation='nearest',cmap=plt.cm.spectral,vmin = vmin, vmax = vmax)
		cb = plt.colorbar(p,shrink=0.5)
		cb.set_label(label)
		plt.show()

	def scale(self,img):
		scale_img = np.zeros(img.shape)
		rows , cols = img.shape
		maxVal = img.max()
		minVal = img.min()
		alpha = 255.0/(maxVal - minVal)

		beta = (-minVal * 255.0)/(maxVal - minVal)

		for i,j in itertools.product(range(rows),range(cols)):
			value =  img[i,j]
			value = floor(1.0*(alpha*value + beta))
			scale_img[i,j] = value
			
		return np.uint8(scale_img)


	def save_match_points(self,dir_ouput,kp1,kp2,matches_points):
		data = []
		for i,j in matches_points:
			pt_lr =  np.array(kp1[i])
			pt_hr =  np.array(kp2[j])
			py_src = pt_hr[0]#fila
			px_src = pt_hr[1]#columna
			py_dst = pt_lr[0]
			px_dst = pt_lr[1]
			data.append([px_src,py_src,px_dst,py_dst])
		
		df = pd.DataFrame(data=data,columns=['px_src','py_src','px_dst','py_dst'])
		df.to_csv(dir_ouput)