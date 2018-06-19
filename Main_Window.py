if __name__ == "__main__":  # This allows running this module by running this script
    import sys

    sys.path.insert(0, "..")

import os
import sys
import time
import numpy as np
from PyQt5 import QtNetwork, QtCore, QtGui, uic, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import configparser
from .Image_Viewer import Image_Viewer
from .Py_Microscope_Assistant import Camera_Reader_Thread
from .Py_Microscope_Assistant import Camera_Macro_Thread
import cv2

__version__ = '0.2'


def resource_path(relative_path):  # Define function to import external files when using PyInstaller.

    """ Get absolute path to resource, works for dev and for PyInstaller """

    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


qtMicroscopeFile = resource_path(os.path.join("Microscope_Assistant", "Main_Window.ui"))  # GUI layout file.
Ui_Microscope, QtBaseClass = uic.loadUiType(qtMicroscopeFile)


class defect_count_thread(QtCore.QThread):

    """Thread for defect count"""

    defect_count_signal = QtCore.pyqtSignal(list)

    def __init__(self, root, parent=None):
        super(defect_count_thread, self).__init__(parent)

        self.root = root
        self.x_pixel = root.x_pixel
        self.y_pixel = root.y_pixel
        self.l_pixel = root.l_pixel
        self.comboBox_mag = root.comboBox_mag
        self.comboBox_type = root.comboBox_type
        self.area_mag = root.area_mag

    def run(self):
        while True:
            time.sleep(1)
            self.count_defects(self.root.liveimage)

    def func_count(self, image, DF, mag, bigger_or_smaller, targetsize):
        if DF == 0:
            image = cv2.bitwise_not(image)  # Invert color for bright field images
        image_data = image  # Only array like image can be taken. Here assume the image is array.
        grayimage = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale.

        hist = cv2.calcHist([grayimage], [0], None, [256], [0, 256])  # Get a 256 * 1 array of histogram
        hist_list = []
        for i in range(0, len(hist)):
            hist_list.append(hist[i][0])
        index_max = hist_list.index(max(hist_list))  # Find the peak on the histogram
        if DF == 0:
            ret, image = cv2.threshold(grayimage, index_max + 40, 255, cv2.THRESH_BINARY)
        else:
            ret, image = cv2.threshold(grayimage, index_max + 40, 255, cv2.THRESH_BINARY)
        image, contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # find par.

        totallength = self.l_pixel / mag
        size = (targetsize / totallength * self.x_pixel) * (targetsize / totallength * self.x_pixel)

        # Calculate the area of each particle
        areas = []
        for i in range(0, len(contours)):
            thearea = cv2.contourArea(contours[i])
            if thearea <= self.x_pixel * self.x_pixel * 0.1:  # Exclude unreasonable big size defects.
                areas.append(thearea)

        # Filter the particles with size
        areas_2 = []
        if bigger_or_smaller == 0:
            for i in range(0, len(areas)):
                if areas[i] < size:
                    areas_2.append(areas[i])
        elif bigger_or_smaller == 1:
            for i in range(0, len(areas)):
                if areas[i] > size:
                    areas_2.append(areas[i])

        # Calculate average size
        if len(areas_2) != 0:
            averagearea = sum(areas_2) / len(areas_2) / self.x_pixel / self.x_pixel * totallength * totallength
        else:
            averagearea = 0

        # Calculate defect density based on calibration data.
        density = len(areas_2) / totallength / (totallength / self.x_pixel * self.y_pixel) / 0.00000001

        return density

    def count_defects(self, image):
        counts = []
        if self.comboBox_type.currentIndex() == 0:
            result = self.func_count(image, 0, self.area_mag[self.comboBox_mag.currentIndex()], 1, 10)
            counts.append(result)
            result = self.func_count(image, 0, self.area_mag[self.comboBox_mag.currentIndex()], 1, 2)
            counts.append(result)
            counts.append('')
        elif self.comboBox_type.currentIndex() == 1:
            counts.append('')
            counts.append('')
            counts.append('')
        elif self.comboBox_type.currentIndex() == 2:
            counts.append('')
            counts.append('')
            result = self.func_count(image, 1, self.area_mag[self.comboBox_mag.currentIndex()], 0, 2)
            counts.append(result)
        self.defect_count_signal.emit(counts)


class Microscope_Assistant_GUI(QWidget, Ui_Microscope):

    """Main Microscope Assistant window."""

    def __init__(self, root, masterroot):
        QWidget.__init__(self)
        Ui_Microscope.__init__(self)
        self.setupUi(self)
        self.root = root  # The mdi area to pack the QWidget
        self.masterroot = masterroot  # The mainwindow
        self.listbox = self.masterroot.listbox
        self.statusbar = self.masterroot.statusbar
        self.status1 = self.masterroot.status1
        self.status2 = self.masterroot.status2
        self.progressbar = self.masterroot.progressbar
        self.addlog = self.masterroot.addlog
        self.addlog_with_button = self.masterroot.addlog_with_button
        self.warningcolor1 = 'red'
        self.warningcolor2 = 'orange'
        self.warningcolor3 = 'royalblue'

        # Camera calibration data
        self.x_pixel = 4224  # X resolution
        self.y_pixel = 3156  # Y resolution
        self.l_pixel = 1160  # Distance of real x length in micron

        # Create left (and right) camera window using reimplemented QgraphicViews
        self.left_graphicsView = Image_Viewer(self.leftframe)
        self.leftlayout.addWidget(self.left_graphicsView)
        self.splitter_2.setSizes([2000, 0])

        # Initial parameters
        self.choice_mag = ["50x", "100x", "200x", "500x", "1000x"]  # Choice of lens with different magnification
        self.area_mag = [1, 2, 4, 10, 20]   # Magnification compared to 50x lens
        self.choice_type = ["Bright Field", "Nomaski", "Dark Field"]
        self.choice_type_short = ["", "_No", "_DF"] # This is used to append to the end of file names
        self.comboBox_mag.addItems(self.choice_mag)
        self.comboBox_mag.setCurrentIndex(0)
        self.comboBox_type.addItems(self.choice_type)
        self.comboBox_type.setCurrentIndex(1)
        self.filename = ''
        self.liveimage = None

        # Signal/slot stuff
        self.buttonclear.clicked.connect(self.clearalldata)
        self.buttonsave.clicked.connect(self.savepicture)
        self.buttonfolder.clicked.connect(self.choosefolder)
        self.shortcut0 = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+S"), self)
        self.shortcut0.activated.connect(self.savepicture)
        self.checkBox_macro.stateChanged.connect(self.show_hide_macro)

        self.parameters = ["exposure", "brightness", "gain", "ISO", "contrast"]
        self.validators = [[-13, 0, 1], [-100, 100, 1], [-100, 100, 1], [-100, 100, 1], [-100, 100, 1]]
        self.func_cv2 = [cv2.CAP_PROP_EXPOSURE, cv2.CAP_PROP_BRIGHTNESS, cv2.CAP_PROP_GAIN, cv2.CAP_PROP_ISO_SPEED,
                         cv2.CAP_PROP_CONTRAST]

        getattr(self, "{}_verticalSlider".format(self.parameters[0])).valueChanged.connect(
            lambda: self.para_valuechange(self.parameters[0]))
        getattr(self, "{}_verticalSlider".format(self.parameters[1])).valueChanged.connect(
            lambda: self.para_valuechange(self.parameters[1]))
        getattr(self, "{}_verticalSlider".format(self.parameters[2])).valueChanged.connect(
            lambda: self.para_valuechange(self.parameters[2]))
        getattr(self, "{}_verticalSlider".format(self.parameters[3])).valueChanged.connect(
            lambda: self.para_valuechange(self.parameters[3]))
        getattr(self, "{}_verticalSlider".format(self.parameters[4])).valueChanged.connect(
            lambda: self.para_valuechange(self.parameters[4]))
        for i in range(0, len(self.parameters)):
            # getattr(self, "{}_verticalSlider".format(self.parameters[i])).valueChanged.connect(lambda index=i: self.para_valuechange(self.parameters[index]))
            getattr(self, "entry_{}".format(self.parameters[i])).returnPressed.connect(lambda index=i: self.set_para(self.parameters[index]))
            getattr(self, "entry_{}".format(self.parameters[i])).setValidator(QDoubleValidator(self.validators[i][0], self.validators[i][1], self.validators[i][2]))

        getattr(self, "checkBox_{}".format(self.parameters[0])).stateChanged.connect(lambda: self.auto_para(self.parameters[0]))

        # Defect count thread
        self.defect_thread = defect_count_thread(self)
        self.defect_thread.defect_count_signal.connect(lambda counts: self.show_defect_count(counts))
        self.defect_thread.start()

        # Camera thread
        self.camera_thread = Camera_Reader_Thread()
        self.camera_thread.cameraFrameReady_signal.connect(self.left_graphicsView.setImage)
        self.camera_thread.cameraFrameReady_signal.connect(self.refresh_Image)
        self.camera_thread.blurriness_signal.connect(self.set_Blurriness)
        self.camera_thread.start()

        # Macro thread: Initialize
        self.macro_thread = None

    def refresh_Image(self, img):

        """Keep a copy of the live image in the main thread and pass the image to macro thread. """

        self.liveimage = img
        if self.macro_thread is not None:
            self.macro_thread.active_image = img

    def set_Blurriness(self, blur):

        """Slot for camera count thread. """

        self.lb_blur.setText("{:.8f}".format(blur))

    def show_defect_count(self, counts):

        """Slot for defect count thread. Show the defect count result in the GUI. """

        for i in range(0, 3):
            if counts[i] != '':
                getattr(self, "lb_defect{}".format(i + 1)).setText("{:.2e}".format(counts[i]))
            else:
                getattr(self, "lb_defect{}".format(i + 1)).setText('')

    def show_hide_macro(self):

        """Show/hide macro camera viewport. Hide the macro camera viewport can improve FPS significantly. """

        if self.checkBox_macro.isChecked() is True:
            self.splitter_2.setSizes([2000, 2000])
            self.right_graphicsView = Image_Viewer(self.rightframe)
            self.rightlayout.addWidget(self.right_graphicsView)
            self.macro_thread = Camera_Macro_Thread()
            self.macro_thread.first_image = self.liveimage
            self.macro_thread.macroFrameReady_signal.connect(self.right_graphicsView.setImage)
            self.macro_thread.start()
        else:
            self.macro_thread._isrunning = 0
            self.macro_thread = None
            self.right_graphicsView.setParent(None)
            self.rightlayout.removeWidget(self.right_graphicsView)
            self.splitter_2.setSizes([2000, 0])

    def choosefolder(self):

        """Choose which folder to save the images."""

        file = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        if file is not "":
            self.entry_folder.setText(file)

    def savepicture(self):

        """Save the image to file based on several entries. """

        def createornotfunc():
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)

            msg.setText("The folder doesn not exists. Do you want to create it?")
            msg.setWindowTitle("Warning")
            msg.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)

            result = msg.exec_()
            if result == QMessageBox.Ok:
                return 1
            else:
                return 0

        def replaceornotfunc():
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)

            msg.setText("The file already exists. Do you want to replace it?")
            msg.setWindowTitle("Warning")
            msg.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)

            result = msg.exec_()
            if result == QMessageBox.Ok:
                return 1
            else:
                return 0

        filename = ''
        if self.entry_folder.text() == '':
            self.addlog("Please choose a folder first", self.warningcolor2)
            return

        if not os.path.isdir(self.entry_folder.text()):
            createornot = createornotfunc()
            if createornot == 1:
                os.makedirs(self.entry_folder.text())
            else:
                return

        filename += self.entry_folder.text() + os.sep

        if self.entry_sample.text() != '':
            filename += self.entry_sample.text() + '_'
        if self.entry_x.text() != '' and self.entry_y.text() != '':
            filename += '(' + self.entry_x.text() + ',' + self.entry_y.text() + ')_'
        filename += self.choice_mag[self.comboBox_mag.currentIndex()]
        filename += self.choice_type_short[self.comboBox_type.currentIndex()] + '.png'

        if filename == self.filename:
            replaceornot = replaceornotfunc()
            if replaceornot == 1:
                self.filename = filename
            else:
                pass
        else:
            self.filename = filename

        cv2.imwrite(self.filename, self.liveimage)
        self.addlog("Image saved to {}.".format(self.filename))

    def para_valuechange(self, para):

        """Behaviors when the parameter sliders are used. """

        value = getattr(self, "{}_verticalSlider".format(para)).value()
        getattr(self, "entry_{}".format(para)).setText("{:.1f}".format(float(value)))

        self.camera_thread.cap.set(self.func_cv2[self.parameters.index(para)], value)
        print("{}: {}".format(para, str(self.camera_thread.cap.get(self.func_cv2[self.parameters.index(para)]))))
        if para == "exposure":
            getattr(self, "checkBox_{}".format(para)).setChecked(False)

    def set_para(self, para):

        """Behaviors when the parameter is manually set to a number.  """

        getattr(self, "{}_verticalSlider".format(para)).setValue(float(getattr(self, "entry_{}".format(para)).text()))
        self.para_valuechange(para)

    def auto_para(self, para):

        """Enable auto parameter. """

        if getattr(self, "checkBox_{}".format(para)).isChecked() is True:
            self.camera_thread.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, True)

    def clearalldata(self):

        """Clear everything. """

        def clearornotfunc():
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)

            msg.setText("Clear and refresh everything?")
            msg.setWindowTitle("Warning")
            msg.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)

            result = msg.exec_()
            if result == QMessageBox.Ok:
                return 1
            else:
                return 0

        clearornot = clearornotfunc()
        if clearornot == 1:
            for i in reversed(range(self.maingrid.count())):
                self.maingrid.itemAt(i).widget().setParent(None)
            # self.__init__(self.root, self.masterroot)
            obj = Microscope_Assistant_GUI(self.root, self.masterroot)
            self.root.setWidget(obj)
            self.root.showMaximized()
            self.root.show()
            self.addlog('-' * 160, "blue")
        else:
            pass


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Microscope_Assistant_GUI(app, app)
    window.show()
    sys.exit(app.exec_())
