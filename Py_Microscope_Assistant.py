import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import stats
from PyQt5 import QtCore

try:
	test = cv2.xfeatures2d
except:
	raise ImportError( "Need to run: python -m pip install opencv-python opencv-contrib-python" )


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
	offset = np.array( offset ).astype(int)
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

if __name__ == "__main__":
	cap = cv2.VideoCapture(0)
	#cap.set(3,1280) # X Resolution
	#cap.set(4,720) # Y Resolution
	cap.set(3,640) # X Resolution
	cap.set(4,480) # Y Resolution
	cap.set( cv2.CAP_PROP_EXPOSURE, -8.0) # Exposure
	cap.set( cv2.CAP_PROP_AUTO_EXPOSURE, True )

	img1 = None
	img3 = None
	while(True):
		# Capture frame-by-frame
		ret, img1 = cap.read()
		if img1 is None:
			continue

		## Our operations on the frame come here
		#img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

		# Display the resulting frame
		cv2.imshow('frame',img1)
		if cv2.waitKey(1) & 0xFF == ord('s'):
			break

	#img1 = cv2.imread('box.png',0)          # queryImage
	#img2 = cv2.imread('box_in_scene.png',0) # trainImage
	img3 = np.zeros( (img1.shape[0],img1.shape[1] * 2,img1.shape[2]), img1.dtype )

	# Initiate SIFT detector
	orb = cv2.ORB_create()
	overall_offset = np.array([0,0])

	kp1, des1 = orb.detectAndCompute(img1,None)
	combined_img = img1
	while(True):
		# Capture frame-by-frame
		ret, img2 = cap.read()
		if img2 is None:
			continue

		# find the keypoints and descriptors with SIFT
		kp2, des2 = orb.detectAndCompute(img2,None)

		if len(kp2) == 0:
			blurrieness = calcBlurriness( img2 )
			img3[:,img1.shape[1]:,:] = img2[:,:,:]
			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.putText(img3, 'Blurriness: ' + str(blurrieness), (500, 400), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

			cv2.imshow('match',img3)
			if cv2.waitKey(1) & 0xFF == ord('s'):
				break
			continue
		# create BFMatcher object
		bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

		# Match descriptors.
		matches = bf.match(des1,des2)

		# Sort them in the order of their distance.
		matches = sorted(matches, key = lambda x:x.distance)
		#test1 = des1[matches[0].queryIdx][ 0 ]
		#test2 = des1[matches[0].queryIdx][ 1 ]
		#test3 = des1[matches[0].queryIdx][ 2 ]
		x_delta =[ kp2[match.trainIdx].pt[ 0 ] - kp1[match.queryIdx].pt[ 0 ] for match in matches[:10] ]
		y_delta = [ kp2[match.trainIdx].pt[ 1 ] - kp1[match.queryIdx].pt[ 1 ] for match in matches[:10] ]
		offset = (-stats.mode(y_delta, axis=0)[0][0], -stats.mode(x_delta, axis=0)[0][0])
		combined_img, overall_offset = Combine_Images( combined_img, img2, offset, overall_offset )
		cv2.imshow('Combined',combined_img)
		img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], None, flags=2)

		font = cv2.FONT_HERSHEY_SIMPLEX
		blurrieness = calcBlurriness( img2 )
		#if blurrieness > 1.9e-4:
		#	continue
		cv2.putText(img3, 'Blurriness: ' + str(blurrieness), (500, 400), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

		#cv2.imshow('match',img3)
		plt.imshow(img3),plt.show()
		if cv2.waitKey(1) & 0xFF == ord('s'):
			break

	# When everything done, release the capture
	self.cap.release()
	cv2.destroyAllWindows()


class Camera_Reader_Thread( QtCore.QThread ):
	cameraFrameReady_signal = QtCore.pyqtSignal(np.ndarray)
	macroFrameReady_signal = QtCore.pyqtSignal(np.ndarray)

	def __init__(self, parent=None):
		super(Camera_Reader_Thread, self).__init__(parent)

	def Initialize( self ):
		self.cap = cv2.VideoCapture(0)
		self.cap.set(3,640) # X Resolution
		self.cap.set(4,480) # Y Resolution
		self.cap.set( cv2.CAP_PROP_EXPOSURE, -3.0 ) # Exposure
		self.cap.set( cv2.CAP_PROP_AUTO_EXPOSURE, True )


	def Change_Exposure( self, exposure ):
		debug = exposure / 10.0
		self.cap.set( cv2.CAP_PROP_EXPOSURE, exposure / 10.0 )

	def run(self):
		self.Initialize()
		#while True:
			#QThread.msleep(100)
			# Capture frame-by-frame
		ret, combined_img = self.cap.read()
		if combined_img is not None:
			self.cameraFrameReady_signal.emit( combined_img )
			self.macroFrameReady_signal.emit( combined_img )


		# Initiate ORB detector
		orb = cv2.xfeatures2d.SIFT_create()
		#orb = cv2.ORB_create()
		#orb = cv2.xfeatures2D.SIFT_create()
		overall_offset = np.array([0,0])

		reference_keypoints, reference_descriptors = orb.detectAndCompute(combined_img,None)
		# create BFMatcher object
		bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

		while(True):
			# Capture frame-by-frame
			ret, active_image = self.cap.read()
			if active_image is None:
				continue

			# find the keypoints and descriptors with SIFT
			active_image_keypoints, active_image_descriptors = orb.detectAndCompute(active_image,None)

			blurrieness = calcBlurriness( active_image )
			image_for_display = active_image.copy()
			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.putText(image_for_display, 'Blurriness: ' + str(blurrieness), (10, 40), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
			self.cameraFrameReady_signal.emit( image_for_display )

			if len(active_image_keypoints) == 0:
				continue
			#if blurrieness > 1.9e-4:
			#	continue

			# Match descriptors.
			matches = bf.match(reference_descriptors, active_image_descriptors)

			img3 = cv2.drawMatches(combined_img,reference_keypoints,active_image,active_image_keypoints,matches[:10], None, flags=2)
			cv2.imshow('match',img3)
			if cv2.waitKey(1) & 0xFF == ord('s'):
				break

			# Sort them in the order of their distance.
			matches = sorted(matches, key = lambda x:x.distance)
			x_delta =[ active_image_keypoints[match.trainIdx].pt[ 0 ] - reference_keypoints[match.queryIdx].pt[ 0 ] for match in matches[:20] ]
			y_delta = [ active_image_keypoints[match.trainIdx].pt[ 1 ] - reference_keypoints[match.queryIdx].pt[ 1 ] for match in matches[:20] ]
			#offset = (-stats.mode(y_delta, axis=0)[0][0], -stats.mode(x_delta, axis=0)[0][0])
			offset = ( -np.median(y_delta), -np.median(x_delta) )
			#offset = (-y_delta[0], -x_delta[0])
			test_combined_img, test_overall_offset = Combine_Images( combined_img, active_image, offset, overall_offset )
			#cv2.putText(test_combined_img, 'offset: ' + str(offset[0]) + ' ' + str(offset[1]), (10, 40), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

			for i in range( min(len(x_delta), 10) ):
				cv2.putText(test_combined_img, 'offset: ' + str(x_delta[i]) + ' ' + str(y_delta[i]), (10, (1+i)*40), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

			self.macroFrameReady_signal.emit( test_combined_img )

			QtCore.QCoreApplication.processEvents()

		# When everything done, release the capture
		self.cap.release()
		cv2.destroyAllWindows()


def Newtons_Method( y_value ):
	x_initial_guess = 1.0
	target_resolution = 1e-2
	i = 0
	while( True ):
		i += 1
		x_i_old = x_i
		x_i = x_i - (Function_of_x( x_i ) - y_value) / Derivative_of_Function_of_x( x_i );
		if( abs( x_i - x_i_old ) < target_resolution ):
			return x_i

def Binary_Search( y_value ):
	left = -270.0
	right = 270
	target_resolution = 1e-2
	while( right - left > target_resolution ):
		center = (right + left) / 2
		if( y_value < Function_of_x( center ) ):
			right = center
		else:
			left = center
	center = (right + left) / 2

	return center
