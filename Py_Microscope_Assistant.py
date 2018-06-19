import time
import numpy as np
from numpy import linalg
import cv2
from matplotlib import pyplot as plt
from scipy import stats
from PyQt5 import QtCore

try:
	test = cv2.xfeatures2d
except:
	raise ImportError( "Need to run: python -m pip install opencv-python opencv-contrib-python" )

reduce = 1
x_pixel = 4224
y_pixel = 3156

# Adapted from http://answers.opencv.org/question/5395/how-to-calculate-blurriness-and-sharpness-of-a-given-image/
def calcBlurriness( src_image ):
    #Mat Gx, Gy;
    Gx = cv2.Sobel( src_image, cv2.CV_32F, 1, 0 );
    Gy = cv2.Sobel( src_image, cv2.CV_32F, 0, 1 );
    normGx = cv2.norm( Gx );
    normGy = cv2.norm( Gy );
    sumSq = normGx * normGx + normGy * normGy;
    return 1. / ( sumSq / (src_image.shape[0] * src_image.shape[1]) + 1e-6 );

def Overlay_Image( bottom_image, top_image, offset ):
	end_spot = offset + np.array( top_image.shape[0:2] )
	bottom_image[int(offset[0]):int(end_spot[0]),int(offset[1]):int(end_spot[1]),:] = top_image[:,:,:]

def Combine_Images( img1, img2, offset, overall_offset ):
	offset = np.array( [round( offset[0]), round(offset[1])] ).astype(int)
	overall_offset = np.array( overall_offset ).astype(int)
	 
	relative_to_origin = overall_offset + offset
	test1 = [min( 0, relative_to_origin[ 0 ] ), min( 0, relative_to_origin[ 1 ] )]
	new_shift_in_origin = np.array( [min( 0, relative_to_origin[ 0 ] ), min( 0, relative_to_origin[ 1 ] )] ).astype(int);

	width = int(  max(img1.shape[1], relative_to_origin[1] + img2.shape[1]) ) - min( 0, relative_to_origin[ 1 ] )
	height = int(  max(img1.shape[0], relative_to_origin[0] + img2.shape[0]) ) - min( 0, relative_to_origin[ 0 ] )
	if width > img1.shape[ 1 ] or height > img1.shape[ 0 ]:
	#if new_shift_in_origin[0] < 0 or new_shift_in_origin[1] < 0:
		blank_image = np.zeros( (height,width,img1.shape[2]), img1.dtype )
		new_combined_image = blank_image
		Overlay_Image( new_combined_image, img1, -new_shift_in_origin )
	else:
		new_combined_image = img1.copy()

	location_in_new_image = relative_to_origin - new_shift_in_origin
	Overlay_Image( new_combined_image, img2, location_in_new_image )

	return new_combined_image, overall_offset - new_shift_in_origin


class Camera_Reader_Thread( QtCore.QThread ):

	"""Thread for the left camera frame. """

	cameraFrameReady_signal = QtCore.pyqtSignal(np.ndarray)
	blurriness_signal = QtCore.pyqtSignal(float)

	def __init__(self, parent=None):
		super(Camera_Reader_Thread, self).__init__(parent)

		self.w = 0
		self.h = 0
		self.resolution_x = 0
		self.resolution_y = 0
		self.active_image = None

	def Initialize( self ):
		self.cap = cv2.VideoCapture(0)
		self.w = self.cap.get(3)
		self.h = self.cap.get(4)
		print("Camera resolution: {}x{}".format(self.w, self.h))

		# self.resolution_x = self.w / reduce
		# self.resolution_y = self.h / reduce
		self.resolution_x = x_pixel
		self.resolution_y = y_pixel
		self.cap.set(3,self.resolution_x) # X Resolution
		self.cap.set(4,self.resolution_y) # Y Resolution
		self.cap.set( cv2.CAP_PROP_EXPOSURE, -3.0 ) # Exposure
		self.cap.set( cv2.CAP_PROP_AUTO_EXPOSURE, True )


	def Change_Exposure( self, exposure ):
		self.cap.set( cv2.CAP_PROP_EXPOSURE, exposure / 10.0 )

	def run(self):
		self.Initialize()
		counter = 0
		while(True):
			# Capture frame-by-frame
			ret, self.active_image = self.cap.read()
			if self.active_image is None:
				continue

			self.cameraFrameReady_signal.emit(self.active_image)
			if counter == 30:
				blurrieness = calcBlurriness(self.active_image)
				self.blurriness_signal.emit(blurrieness)
				counter = 0
			else:
				counter += 1
			# QtCore.QCoreApplication.processEvents()
			continue
		# When everything done, release the capture
		self.cap.release()
		cv2.destroyAllWindows()

class Camera_Macro_Thread( QtCore.QThread ):

	"""Thread for the right camera frame. """

	macroFrameReady_signal = QtCore.pyqtSignal(np.ndarray)

	def __init__(self, parent=None):
		super(Camera_Macro_Thread, self).__init__(parent)

		self.first_image = None
		self.active_image = None
		self.w = 0
		self.h = 0
		self.resolution_x = 0
		self.resolution_y = 0

		self._isrunning = 1

	def run(self):
		# Initiate ORB detector
		orb = cv2.xfeatures2d.SIFT_create()
		#orb = cv2.ORB_create()
		#orb = cv2.xfeatures2D.SIFT_create()
		overall_offset = np.array([0,0])
		#reference_keypoints, reference_descriptors = orb.detectAndCompute(combined_img,None)
		# create BFMatcher object
		#bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
		FLANN_INDEX_KDTREE = 1
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		search_params = dict(checks = 50)
		flann = cv2.FlannBasedMatcher(index_params, search_params)

		while( True ):
			if self.first_image is None:
				continue
			combined_img_gray = cv2.cvtColor( self.first_image, cv2.COLOR_BGR2GRAY )
			reference_keypoints, reference_descriptors = orb.detectAndCompute(combined_img_gray,None)
			if len( reference_keypoints ) > 0:
				break

		self.macroFrameReady_signal.emit( self.first_image )

		while(True):
			if self._isrunning == 0:
				break
			if self.active_image is None:
				continue

			# find the keypoints and descriptors with SIFT
			active_image_gray = cv2.cvtColor( self.active_image, cv2.COLOR_BGR2GRAY )
			active_image_keypoints, active_image_descriptors = orb.detectAndCompute(active_image_gray,None)
			#active_image_keypoints, active_image_descriptors = orb.detectAndCompute(active_image,None)

			image_for_display = self.active_image.copy()
			# font = cv2.FONT_HERSHEY_SIMPLEX
			#cv2.putText(image_for_display, 'Blurriness: ' + str(blurrieness), (10, 40), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

			if len(active_image_keypoints) <= 1:
				continue
			#if blurrieness > 1.9e-4:
			#	continue

			# Match descriptors.
			#matches = bf.match(reference_descriptors, active_image_descriptors)
			matches = flann.knnMatch( reference_descriptors, active_image_descriptors, 2 )
			#img3 = cv2.drawMatches(combined_img,reference_keypoints,active_image,active_image_keypoints,matches[:10], None, flags=2)
			#cv2.imshow('match',img3)
			#if cv2.waitKey(1) & 0xFF == ord('s'):
			#	break

			# store all the good matches as per Lowe's ratio test.
			good = []
			for m,n in matches:
				if m.distance < 0.7*n.distance:
					good.append(m)			# Sort them in the order of their distance.
			draw_params = dict(matchColor = (0,255,0), # draw matches in green color
							   singlePointColor = None,
							   flags = 2)
			#img3 = cv2.drawMatches(combined_img,reference_keypoints,active_image,active_image_keypoints,good,None,**draw_params)
			#cv2.imshow('match',img3)
			#if cv2.waitKey(1) & 0xFF == ord('s'):
			#	break
			matches = good
			if len( good ) < 3:
				continue
			#matches = sorted(matches, key = lambda x:x.distance)
			x_delta =[ active_image_keypoints[match.trainIdx].pt[ 0 ] - reference_keypoints[match.queryIdx].pt[ 0 ] for match in matches[:20] ]
			y_delta = [ active_image_keypoints[match.trainIdx].pt[ 1 ] - reference_keypoints[match.queryIdx].pt[ 1 ] for match in matches[:20] ]
			#offset = (-stats.mode(y_delta, axis=0)[0][0], -stats.mode(x_delta, axis=0)[0][0])
			offset = ( -np.median(y_delta), -np.median(x_delta) )
			#offset = (-y_delta[0], -x_delta[0])
			test_combined_img, test_overall_offset = Combine_Images( self.first_image, self.active_image, offset, overall_offset )
			#cv2.putText(test_combined_img, 'offset: ' + str(offset[0]) + ' ' + str(offset[1]), (10, 40), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

			#for i in range( min(len(x_delta), 10) ):
			#	cv2.putText(test_combined_img, 'offset: ' + str(x_delta[i]) + ' ' + str(y_delta[i]), (10, (1+i)*40), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

			self.macroFrameReady_signal.emit( test_combined_img )
			time.sleep(0.5)		# Since the macro thread is no longer slowing down camera thread, maybe we don't need this one.

			QtCore.QCoreApplication.processEvents()

