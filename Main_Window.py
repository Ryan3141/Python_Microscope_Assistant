import sys
from PyQt5 import QtNetwork, QtCore, QtGui, uic, QtWidgets
import configparser
from Py_Microscope_Assistant import Camera_Reader_Thread

qtCreatorFile = "Main_Window.ui" # GUI layout file.

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)


def Resize_On_First_Call( image, image_object, signal ):
	signal.disconnect()
	image_object.setImage( image )
	image_object.fitImageInView()
	signal.connect( image_object.setImage )

class MyApp(QtWidgets.QWidget, Ui_MainWindow):

	#Set_New_Temperature_K = QtCore.pyqtSignal(float)
	#Turn_Off_Temperature_Control = QtCore.pyqtSignal(float)
	def __init__(self):
		QtWidgets.QWidget.__init__(self)
		Ui_MainWindow.__init__(self)
		self.setupUi(self)

		self.Init_Subsystems()
		#self.Connect_Control_Logic()
		
		#cap = cv2.VideoCapture(0)
		#ret, img1 = cap.read()
		#if img1 is not None:
		#	self.left_graphicsView.setImage( img1 )
		#self.timer = QtCore.QTimer( self )
		#self.timer.timeout.connect( Read_Camera_Frame )
		#self.timer.start( 1000 ) # Look for a frame

	def Init_Subsystems( self ):
		config = configparser.ConfigParser()
		config.read('configuration.ini')

		self.camera_thread = Camera_Reader_Thread()
		self.camera_thread.cameraFrameReady_signal.connect( lambda image, image_draw_func=self.left_graphicsView, signal=self.camera_thread.cameraFrameReady_signal : Resize_On_First_Call(image, image_draw_func, signal) )
		self.camera_thread.macroFrameReady_signal.connect( lambda image, image_draw_func=self.right_graphicsView, signal=self.camera_thread.macroFrameReady_signal : Resize_On_First_Call(image, image_draw_func, signal) )
		self.camera_thread.start()

		self.exposure_verticalSlider.valueChanged.connect( self.camera_thread.Change_Exposure )

if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	window = MyApp()
	window.show()
	sys.exit(app.exec_())

