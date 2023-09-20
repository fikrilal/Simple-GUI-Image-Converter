from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap
import matplotlib.pyplot as plt
import os

import numpy as np

class Ui_MainWindow(object):

    def openImage(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(None, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)", options=options)
        if file_name:
            try:
                image = QtGui.QPixmap(file_name)
                self.label.setPixmap(image)
            except Exception as e:
             QtWidgets.QMessageBox.critical(None, "Error", f"Error opening image: {str(e)}")


    def saveAsImage(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getSaveFileName(None, "Save Image As", 
                        "", "Images (*.png *.jpg *.jpeg *.bmp)", options=options)
        if file_name:
            pixmap = self.label_2.pixmap()
            pixmap.save(file_name)

    def exitApplication(self):
        QtWidgets.QApplication.quit()

    def convertToGreyscale(self, method):
        original_pixmap = self.label.pixmap()
        if original_pixmap:
            image = original_pixmap.toImage()

            for y in range(image.height()):
                for x in range(image.width()):
                    pixel = QtGui.qGray(image.pixel(x, y))

                    if method == "average":
                        pixel_color = QtGui.qRgb(pixel, pixel, pixel)
                    elif method == "lightness":
                        pixel_color = self.lightnessGreyscale(pixel)
                    elif method == "luminosity":
                        pixel_color = self.luminosityGreyscale(image.pixel(x, y))
                    elif method == "inverse":
                        pixel_color = self.inverseGreyscale(pixel)

                    image.setPixel(x, y, pixel_color)

            greyscale_pixmap = QPixmap.fromImage(image)
            self.label_2.setPixmap(greyscale_pixmap)

    def averageGreyscale(self, pixel):
        greyscale_value = int((QtGui.qRed(pixel) + QtGui.qGreen(pixel) + QtGui.qBlue(pixel)) / 3)
        return QtGui.qRgb(greyscale_value, greyscale_value, greyscale_value)

    def lightnessGreyscale(self, pixel):
        r = QtGui.qRed(pixel)
        g = QtGui.qGreen(pixel)
        b = QtGui.qBlue(pixel)
        lightness = int((max(r, g, b) + min(r, g, b)) / 2)
        return QtGui.qRgb(lightness, lightness, lightness)

    def luminosityGreyscale(self, rgb_pixel):
        r = QtGui.qRed(rgb_pixel)
        g = QtGui.qGreen(rgb_pixel)
        b = QtGui.qBlue(rgb_pixel)
        luminosity = int(0.21 * r + 0.72 * g + 0.07 * b)
        return QtGui.qRgb(luminosity, luminosity, luminosity)
    
    def inverseGreyscale(self, pixel):
        return QtGui.qRgb(255 - QtGui.qRed(pixel), 255 - QtGui.qGreen(pixel), 255 - QtGui.qBlue(pixel))
    
    def flipHorizontal(self):
        original_pixmap = self.label.pixmap()
        if original_pixmap:
            flipped_pixmap = original_pixmap.transformed(QtGui.QTransform().scale(-1, 1))
            self.label_2.setPixmap(flipped_pixmap)

    def flipVertical(self):
        original_pixmap = self.label.pixmap()
        if original_pixmap:
            flipped_pixmap = original_pixmap.transformed(QtGui.QTransform().scale(1, -1))
            self.label_2.setPixmap(flipped_pixmap)

    def rotateClockwise(self):
        original_pixmap = self.label.pixmap()
        if original_pixmap:
            rotated_pixmap = original_pixmap.transformed(QtGui.QTransform().rotate(90))
            self.label_2.setPixmap(rotated_pixmap)

    def histogramEqualization(self):
        original_pixmap = self.label.pixmap()
        if original_pixmap:
            # Konversi pixmap ke QImage
            image = original_pixmap.toImage()
            width = image.width()
            height = image.height()

            grayscale_image = np.zeros((height, width), dtype=np.uint8)

            # Menghitung histogram
            histogram = [0] * 256
            for y in range(height):
                for x in range(width):
                    r, g, b, _ = QtGui.QColor(image.pixel(x, y)).getRgb()
                    gray_value = int((r + g + b) / 3)
                    grayscale_image[y][x] = gray_value
                    histogram[gray_value] += 1

            # Menghitung cumulative histogram
            cumulative_histogram = [sum(histogram[:i+1]) for i in range(256)]

            # Normalisasi cumulative histogram
            max_pixel_value = width * height
            normalized_cumulative_histogram = [(cumulative_histogram[i] / max_pixel_value) * 255 for i in range(256)]

            # Menerapkan equalization pada citra
            equalized_image = np.zeros((height, width), dtype=np.uint8)
            for y in range(height):
                for x in range(width):
                    equalized_image[y][x] = int(normalized_cumulative_histogram[grayscale_image[y][x]])

            equalized_qimage = QtGui.QImage(equalized_image.data, width, height, width, QtGui.QImage.Format_Grayscale8)
            equalized_pixmap = QtGui.QPixmap.fromImage(equalized_qimage)
            self.label_2.setPixmap(equalized_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)

            # Buat histogram sebelum equalization
            plt.figure(figsize=(12, 6))
            plt.subplot(121)
            plt.hist(np.array(grayscale_image).ravel(), bins=256, range=(0, 256), density=True, color='b', alpha=0.6)
            plt.title('Histogram Sebelum Equalization')
            plt.xlabel('Nilai Pixel')
            plt.ylabel('Frekuensi Relatif')

            # Buat histogram sesudah equalization
            plt.subplot(122)
            equalized_image_flat = np.array(equalized_image).ravel()
            plt.hist(equalized_image_flat, bins=256, range=(0, 256), density=True, color='r', alpha=0.6)
            plt.title('Histogram Sesudah Equalization')
            plt.xlabel('Nilai Pixel')
            plt.ylabel('Frekuensi Relatif')

            plt.tight_layout()
            plt.show()


    def fuzzyHERGB(self):
        # Tambahkan logika pemrosesan fuzzy HE RGB di sini
        pass

    def fuzzyGreyscale(self):
        # Tambahkan logika pemrosesan fuzzy greyscale di sini
        pass

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1215, 702)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(7, 8, 591, 631))
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setText("")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(610, 10, 591, 631))
        self.label_2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1215, 31))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuImage_Processing = QtWidgets.QMenu(self.menubar)
        self.menuImage_Processing.setObjectName("menuImage_Processing")
        self.menuRGB_to_Greyscale = QtWidgets.QMenu(self.menuImage_Processing)
        self.menuRGB_to_Greyscale.setObjectName("menuRGB_to_Greyscale")
        self.menuImage_Geometri = QtWidgets.QMenu(self.menubar)
        self.menuImage_Geometri.setObjectName("menuImage_Geometri")
        self.menuAritmatics_Operation = QtWidgets.QMenu(self.menubar)
        self.menuAritmatics_Operation.setObjectName("menuAritmatics_Operation")
        self.menuHistogram_Processing = QtWidgets.QMenu(self.menubar)
        self.menuHistogram_Processing.setObjectName("menuHistogram_Processing")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen = QtWidgets.QAction(MainWindow) 
        self.actionOpen.setObjectName("actionOpen")
        self.actionOpen.triggered.connect(self.openImage) 
        self.actionNew_File = QtWidgets.QAction(MainWindow)
        self.actionNew_File.setObjectName("actionNew_File")
        self.actionSave_As = QtWidgets.QAction(MainWindow)
        self.actionSave_As.setObjectName("actionSave_As")
        self.actionSave_As.triggered.connect(self.saveAsImage)
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.actionExit.triggered.connect(self.exitApplication)
        self.actionInformation = QtWidgets.QAction(MainWindow)
        self.actionInformation.setObjectName("actionInformation")
        self.actionAverage = QtWidgets.QAction(MainWindow)
        self.actionAverage.setObjectName("actionAverage")
        self.actionAverage.triggered.connect(lambda: self.convertToGreyscale("average"))
        self.actionLightness = QtWidgets.QAction(MainWindow)
        self.actionLightness.setObjectName("actionLightness")
        self.actionLightness.triggered.connect(lambda: self.convertToGreyscale("lightness"))
        self.actionLuminosity = QtWidgets.QAction(MainWindow)
        self.actionLuminosity.setObjectName("actionLuminosity")
        self.actionLuminosity.triggered.connect(lambda: self.convertToGreyscale("luminosity"))
        self.actionInverse = QtWidgets.QAction(MainWindow)
        self.actionInverse.setObjectName("actionInverse")
        self.actionInverse.triggered.connect(lambda: self.convertToGreyscale("inverse"))
        self.actionFlipHorizontal = QtWidgets.QAction(MainWindow)
        self.actionFlipHorizontal.setObjectName("actionFlipHorizontal")
        self.actionFlipHorizontal.triggered.connect(self.flipHorizontal)
        self.actionFlipVertical = QtWidgets.QAction(MainWindow)
        self.actionFlipVertical.setObjectName("actionFlipVertical")
        self.actionFlipVertical.triggered.connect(self.flipVertical)
        self.actionRotateClockwise = QtWidgets.QAction(MainWindow)
        self.actionRotateClockwise.setObjectName("actionRotateClockwise")
        self.actionRotateClockwise.triggered.connect(self.rotateClockwise)
        self.actionHistogram_Equalization = QtWidgets.QAction(MainWindow)
        self.actionHistogram_Equalization.setObjectName("actionHistogram_Equalization")
        self.actionHistogram_Equalization.triggered.connect(self.histogramEqualization)
        self.actionFuzzy_HE_RGB = QtWidgets.QAction(MainWindow)
        self.actionFuzzy_HE_RGB.setObjectName("actionFuzzy_HE_RGB")
        self.actionFuzzy_HE_RGB.triggered.connect(self.fuzzyHERGB)
        self.actionFuzzy_Greyscale = QtWidgets.QAction(MainWindow)
        self.actionFuzzy_Greyscale.setObjectName("actionFuzzy_Greyscale")
        self.actionFuzzy_Greyscale.triggered.connect(self.fuzzyGreyscale)
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionNew_File)
        self.menuFile.addAction(self.actionSave_As)
        self.menuFile.addAction(self.actionExit)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionInformation)
        self.menuRGB_to_Greyscale.addAction(self.actionAverage)
        self.menuRGB_to_Greyscale.addAction(self.actionLightness)
        self.menuRGB_to_Greyscale.addAction(self.actionLuminosity)
        self.menuImage_Processing.addAction(self.menuRGB_to_Greyscale.menuAction())
        self.menuImage_Processing.addAction(self.actionInverse)
        self.menuImage_Geometri.addAction(self.actionFlipHorizontal)
        self.menuImage_Geometri.addAction(self.actionFlipVertical)  # Tambahkan item-menu flip vertikal
        self.menuImage_Geometri.addAction(self.actionRotateClockwise)
        self.menuHistogram_Processing.addAction(self.actionHistogram_Equalization)
        self.menuHistogram_Processing.addAction(self.actionFuzzy_HE_RGB)
        self.menuHistogram_Processing.addAction(self.actionFuzzy_Greyscale)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuImage_Processing.menuAction())
        self.menubar.addAction(self.menuImage_Geometri.menuAction()) 
        self.menubar.addAction(self.menuHistogram_Processing.menuAction())
        self.menubar.addAction(self.menuAritmatics_Operation.menuAction())
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuImage_Processing.setTitle(_translate("MainWindow", "Image Processing"))
        self.menuImage_Geometri.setTitle(_translate("MainWindow", "Geometri"))
        self.menuRGB_to_Greyscale.setTitle(_translate("MainWindow", "RGB to Greyscale"))
        self.menuAritmatics_Operation.setTitle(_translate("MainWindow", "Aritmatics Operation"))
        self.menuHistogram_Processing.setTitle(_translate("MainWindow", "Histogram Processing"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionNew_File.setText(_translate("MainWindow", "New File"))
        self.actionSave_As.setText(_translate("MainWindow", "Save As"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        self.actionInformation.setText(_translate("MainWindow", "Information"))
        self.actionAverage.setText(_translate("MainWindow", "Average"))
        self.actionLightness.setText(_translate("MainWindow", "Lightness"))
        self.actionLuminosity.setText(_translate("MainWindow", "Luminosity"))
        self.actionInverse.setText(_translate("MainWindow", "Inverse"))
        self.actionFlipHorizontal.setText(_translate("MainWindow", "Flip Horizontal"))
        self.actionFlipVertical.setText(_translate("MainWindow", "Flip Vertical"))
        self.actionRotateClockwise.setText(_translate("MainWindow", "Rotate 90°"))
        self.actionHistogram_Equalization.setText(_translate("MainWindow", "Histogram Equalization"))
        self.actionFuzzy_HE_RGB.setText(_translate("MainWindow", "Fuzzy HE RGB"))
        self.actionFuzzy_Greyscale.setText(_translate("MainWindow", "Fuzzy Greyscale"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())