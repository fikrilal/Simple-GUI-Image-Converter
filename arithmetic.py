from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap
import numpy as np

class Ui_MainWindow(object):
    def openImage(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(None, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)", options=options)
        if file_name:
            try:
                image = QtGui.QPixmap(file_name)
                label_width = self.current_label.width()
                label_height = self.current_label.height()
                scaled_image = image.scaled(label_width, label_height, QtCore.Qt.KeepAspectRatio)
                self.current_label.setPixmap(scaled_image)
                self.current_label.setAlignment(QtCore.Qt.AlignCenter)

                # Ganti current_label agar digunakan pada label selanjutnya
                if self.current_label == self.label:
                    self.current_label = self.label_2
                else:
                    self.current_label = self.label_3

            except Exception as e:
                QtWidgets.QMessageBox.critical(None, "Error", f"Error opening image: {str(e)}")

    def operasiPenjumlahan(self):
        pixmap1 = self.label.pixmap()
        pixmap2 = self.label_2.pixmap()

        if pixmap1 and pixmap2:
            image1 = pixmap1.toImage()
            image2 = pixmap2.toImage()

            if image1.size() == image2.size():
                result_image = QtGui.QImage(image1.size(), QtGui.QImage.Format_ARGB32)

                for x in range(image1.width()):
                    for y in range(image1.height()):
                        color1 = image1.pixel(x, y)
                        color2 = image2.pixel(x, y)

                        r1, g1, b1, a1 = QtGui.qRed(color1), QtGui.qGreen(color1), QtGui.qBlue(color1), QtGui.qAlpha(color1)
                        r2, g2, b2, a2 = QtGui.qRed(color2), QtGui.qGreen(color2), QtGui.qBlue(color2), QtGui.qAlpha(color2)

                        r = min(r1 + r2, 255)
                        g = min(g1 + g2, 255)
                        b = min(b1 + b2, 255)
                        a = min(a1 + a2, 255)

                        result_image.setPixel(x, y, QtGui.qRgba(r, g, b, a))

                result_pixmap = QtGui.QPixmap.fromImage(result_image)
                label_width = self.label_3.width()
                label_height = self.label_3.height()
                scaled_image = result_pixmap.scaled(label_width, label_height, QtCore.Qt.KeepAspectRatio)
                self.label_3.setPixmap(scaled_image)
                self.label_3.setAlignment(QtCore.Qt.AlignCenter)
            else:
                QtWidgets.QMessageBox.warning(None, "Warning", "Image sizes are not the same.")
        else:
            QtWidgets.QMessageBox.warning(None, "Warning", "Both Label 1 and Label 2 must have images.")

    def operasiPengurangan(self):
        pixmap1 = self.label.pixmap()
        pixmap2 = self.label_2.pixmap()

        if pixmap1 and pixmap2:
            image1 = pixmap1.toImage()
            image2 = pixmap2.toImage()

            if image1.size() == image2.size():
                result_image = QtGui.QImage(image1.size(), QtGui.QImage.Format_ARGB32)

                for x in range(image1.width()):
                    for y in range(image1.height()):
                        color1 = image1.pixel(x, y)
                        color2 = image2.pixel(x, y)

                        r1, g1, b1, a1 = QtGui.qRed(color1), QtGui.qGreen(color1), QtGui.qBlue(color1), QtGui.qAlpha(color1)
                        r2, g2, b2, a2 = QtGui.qRed(color2), QtGui.qGreen(color2), QtGui.qBlue(color2), QtGui.qAlpha(color2)

                        r = max(r1 - r2, 0)
                        g = max(g1 - g2, 0)
                        b = max(b1 - b2, 0)
                        a = max(a1 - a2, 0)

                        result_image.setPixel(x, y, QtGui.qRgba(r, g, b, a))

                result_pixmap = QtGui.QPixmap.fromImage(result_image)
                label_width = self.label_3.width()
                label_height = self.label_3.height()
                scaled_image = result_pixmap.scaled(label_width, label_height, QtCore.Qt.KeepAspectRatio)
                self.label_3.setPixmap(scaled_image)
                self.label_3.setAlignment(QtCore.Qt.AlignCenter)
            else:
                QtWidgets.QMessageBox.warning(None, "Warning", "Image sizes are not the same.")
        else:
            QtWidgets.QMessageBox.warning(None, "Warning", "Both Label 1 and Label 2 must have images.")

    def operasiPerkalian(self):
        pixmap1 = self.label.pixmap()
        pixmap2 = self.label_2.pixmap()

        if pixmap1 and pixmap2:
            image1 = pixmap1.toImage()
            image2 = pixmap2.toImage()

            if image1.size() == image2.size():
                result_image = QtGui.QImage(image1.size(), QtGui.QImage.Format_ARGB32)

                for x in range(image1.width()):
                    for y in range(image1.height()):
                        color1 = image1.pixel(x, y)
                        color2 = image2.pixel(x, y)

                        r1, g1, b1, a1 = QtGui.qRed(color1), QtGui.qGreen(color1), QtGui.qBlue(color1), QtGui.qAlpha(color1)
                        r2, g2, b2, a2 = QtGui.qRed(color2), QtGui.qGreen(color2), QtGui.qBlue(color2), QtGui.qAlpha(color2)

                        r = (r1 * r2) // 255
                        g = (g1 * g2) // 255
                        b = (b1 * b2) // 255
                        a = (a1 * a2) // 255

                        result_image.setPixel(x, y, QtGui.qRgba(r, g, b, a))

                result_pixmap = QtGui.QPixmap.fromImage(result_image)
                label_width = self.label_3.width()
                label_height = self.label_3.height()
                scaled_image = result_pixmap.scaled(label_width, label_height, QtCore.Qt.KeepAspectRatio)
                self.label_3.setPixmap(scaled_image)
                self.label_3.setAlignment(QtCore.Qt.AlignCenter)
            else:
                QtWidgets.QMessageBox.warning(None, "Warning", "Image sizes are not the same.")
        else:
            QtWidgets.QMessageBox.warning(None, "Warning", "Both Label 1 and Label 2 must have images.")

    def operasiPembagian(self):
        pixmap1 = self.label.pixmap()
        pixmap2 = self.label_2.pixmap()

        if pixmap1 and pixmap2:
            image1 = pixmap1.toImage()
            image2 = pixmap2.toImage()

            if image1.size() == image2.size():
                result_image = QtGui.QImage(image1.size(), QtGui.QImage.Format_ARGB32)

                for x in range(image1.width()):
                    for y in range(image1.height()):
                        color1 = image1.pixel(x, y)
                        color2 = image2.pixel(x, y)

                        r1, g1, b1, a1 = QtGui.qRed(color1), QtGui.qGreen(color1), QtGui.qBlue(color1), QtGui.qAlpha(color1)
                        r2, g2, b2, a2 = QtGui.qRed(color2), QtGui.qGreen(color2), QtGui.qBlue(color2), QtGui.qAlpha(color2)

                        if r2 == 0:
                            r = r1
                        else:
                            r = min(int(r1 / r2), 255)

                        if g2 == 0:
                            g = g1
                        else:
                            g = min(int(g1 / g2), 255)

                        if b2 == 0:
                            b = b1
                        else:
                            b = min(int(b1 / b2), 255)

                        if a2 == 0:
                            a = a1
                        else:
                            a = min(int(a1 / a2), 255)

                        result_image.setPixel(x, y, QtGui.qRgba(r, g, b, a))

                result_pixmap = QtGui.QPixmap.fromImage(result_image)
                label_width = self.label_3.width()
                label_height = self.label_3.height()
                scaled_image = result_pixmap.scaled(label_width, label_height, QtCore.Qt.KeepAspectRatio)
                self.label_3.setPixmap(scaled_image)
                self.label_3.setAlignment(QtCore.Qt.AlignCenter)
            else:
                QtWidgets.QMessageBox.warning(None, "Warning", "Image sizes are not the same.")
        else:
            QtWidgets.QMessageBox.warning(None, "Warning", "Both Label 1 and Label 2 must have images.")

    def operasiAND(self):
        pixmap1 = self.label.pixmap()
        pixmap2 = self.label_2.pixmap()

        if pixmap1 and pixmap2:
            image1 = pixmap1.toImage()
            image2 = pixmap2.toImage()

            if image1.size() == image2.size():
                result_image = QtGui.QImage(image1.size(), QtGui.QImage.Format_ARGB32)

                for x in range(image1.width()):
                    for y in range(image1.height()):
                        color1 = image1.pixel(x, y)
                        color2 = image2.pixel(x, y)

                        r1, g1, b1, a1 = QtGui.qRed(color1), QtGui.qGreen(color1), QtGui.qBlue(color1), QtGui.qAlpha(color1)
                        r2, g2, b2, a2 = QtGui.qRed(color2), QtGui.qGreen(color2), QtGui.qBlue(color2), QtGui.qAlpha(color2)

                        # Operasi AND pada setiap komponen RGBA
                        r = r1 & r2
                        g = g1 & g2
                        b = b1 & b2
                        a = a1 & a2

                        result_image.setPixel(x, y, QtGui.qRgba(r, g, b, a))

                result_pixmap = QtGui.QPixmap.fromImage(result_image)
                label_width = self.label_3.width()
                label_height = self.label_3.height()
                scaled_image = result_pixmap.scaled(label_width, label_height, QtCore.Qt.KeepAspectRatio)
                self.label_3.setPixmap(scaled_image)
                self.label_3.setAlignment(QtCore.Qt.AlignCenter)
            else:
                QtWidgets.QMessageBox.warning(None, "Warning", "Image sizes are not the same.")
        else:
            QtWidgets.QMessageBox.warning(None, "Warning", "Both Label 1 and Label 2 must have images.")

    def operasiXOR(self):
        pixmap1 = self.label.pixmap()
        pixmap2 = self.label_2.pixmap()

        if pixmap1 and pixmap2:
            image1 = pixmap1.toImage()
            image2 = pixmap2.toImage()

            if image1.size() == image2.size():
                result_image = QtGui.QImage(image1.size(), QtGui.QImage.Format_ARGB32)

                for x in range(image1.width()):
                    for y in range(image1.height()):
                        color1 = image1.pixel(x, y)
                        color2 = image2.pixel(x, y)

                        r1, g1, b1, a1 = QtGui.qRed(color1), QtGui.qGreen(color1), QtGui.qBlue(color1), QtGui.qAlpha(color1)
                        r2, g2, b2, a2 = QtGui.qRed(color2), QtGui.qGreen(color2), QtGui.qBlue(color2), QtGui.qAlpha(color2)

                        # Operasi XOR pada setiap komponen RGBA
                        r = r1 ^ r2
                        g = g1 ^ g2
                        b = b1 ^ b2
                        a = a1 ^ a2

                        result_image.setPixel(x, y, QtGui.qRgba(r, g, b, a))

                result_pixmap = QtGui.QPixmap.fromImage(result_image)
                label_width = self.label_3.width()
                label_height = self.label_3.height()
                scaled_image = result_pixmap.scaled(label_width, label_height, QtCore.Qt.KeepAspectRatio)
                self.label_3.setPixmap(scaled_image)
                self.label_3.setAlignment(QtCore.Qt.AlignCenter)
            else:
                QtWidgets.QMessageBox.warning(None, "Warning", "Image sizes are not the same.")
        else:
            QtWidgets.QMessageBox.warning(None, "Warning", "Both Label 1 and Label 2 must have images.")

    def operasiNOT(self):
        pixmap1 = self.label.pixmap()

        if pixmap1:
            image1 = pixmap1.toImage()
            result_image = QtGui.QImage(image1.size(), QtGui.QImage.Format_ARGB32)

            for x in range(image1.width()):
                for y in range(image1.height()):
                    color1 = image1.pixel(x, y)

                    r1, g1, b1, a1 = QtGui.qRed(color1), QtGui.qGreen(color1), QtGui.qBlue(color1), QtGui.qAlpha(color1)

                    # Operasi NOT pada setiap komponen RGBA
                    r = 255 - r1
                    g = 255 - g1
                    b = 255 - b1
                    a = a1

                    result_image.setPixel(x, y, QtGui.qRgba(r, g, b, a))

            result_pixmap = QtGui.QPixmap.fromImage(result_image)
            label_width = self.label_3.width()
            label_height = self.label_3.height()
            scaled_image = result_pixmap.scaled(label_width, label_height, QtCore.Qt.KeepAspectRatio)
            self.label_3.setPixmap(scaled_image)
            self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        else:
            QtWidgets.QMessageBox.warning(None, "Warning", "Label 1 must have an image.")


    def saveAsImage(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getSaveFileName(None, "Save Image As", 
                        "", "Images (*.png *.jpg *.jpeg *.bmp)", options=options)
        if file_name:
            pixmap = self.label_3.pixmap()
            pixmap.save(file_name)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1832, 705)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(7, 8, 591, 631))
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setText("")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(620, 10, 591, 631))
        self.label_2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(1230, 10, 591, 631))
        self.label_3.setFrameShape(QtWidgets.QFrame.Box)
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1832, 31))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuImage_Processing = QtWidgets.QMenu(self.menubar)
        self.menuImage_Processing.setObjectName("menuImage_Processing")
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
        self.actionInformation = QtWidgets.QAction(MainWindow)
        self.actionInformation.setObjectName("actionInformation")
        self.actionAverage = QtWidgets.QAction(MainWindow)
        self.actionAverage.setObjectName("actionAverage")
        self.actionLightness = QtWidgets.QAction(MainWindow)
        self.actionLightness.setObjectName("actionLightness")
        self.actionLuminosity = QtWidgets.QAction(MainWindow)
        self.actionLuminosity.setObjectName("actionLuminosity")
        self.actionInverse = QtWidgets.QAction(MainWindow)
        self.actionInverse.setObjectName("actionInverse")
        self.actionPenjumlahan = QtWidgets.QAction(MainWindow)
        self.actionPenjumlahan.setObjectName("actionPenjumlahan")
        self.actionPengurangan = QtWidgets.QAction(MainWindow)
        self.actionPengurangan.setObjectName("actionPengurangan")
        self.actionPerkalian = QtWidgets.QAction(MainWindow)
        self.actionPerkalian.setObjectName("actionPerkalian")
        self.actionPembagian = QtWidgets.QAction(MainWindow)
        self.actionPembagian.setObjectName("actionPembagian")
        self.actionAND = QtWidgets.QAction(MainWindow)
        self.actionAND.setObjectName("actionAND")
        self.actionXOR = QtWidgets.QAction(MainWindow)
        self.actionXOR.setObjectName("actionXOR")
        self.actionNOT = QtWidgets.QAction(MainWindow)
        self.actionNOT.setObjectName("actionNOT")
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionNew_File)
        self.menuFile.addAction(self.actionSave_As)
        self.menuFile.addAction(self.actionExit)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionInformation)
        self.menuImage_Processing.addAction(self.actionPenjumlahan)
        self.menuImage_Processing.addAction(self.actionPengurangan)
        self.menuImage_Processing.addAction(self.actionPerkalian)
        self.menuImage_Processing.addAction(self.actionPembagian)
        self.menuImage_Processing.addAction(self.actionAND)
        self.menuImage_Processing.addAction(self.actionXOR)
        self.menuImage_Processing.addAction(self.actionNOT)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuImage_Processing.menuAction())

        self.current_label = self.label
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuImage_Processing.setTitle(_translate("MainWindow", "Image Processing"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionNew_File.setText(_translate("MainWindow", "New File"))
        self.actionSave_As.setText(_translate("MainWindow", "Save As"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        self.actionInformation.setText(_translate("MainWindow", "Information"))
        self.actionAverage.setText(_translate("MainWindow", "Average"))
        self.actionLightness.setText(_translate("MainWindow", "Lightness"))
        self.actionLuminosity.setText(_translate("MainWindow", "Luminosity"))
        self.actionInverse.setText(_translate("MainWindow", "Inverse"))
        self.actionPenjumlahan.setText(_translate("MainWindow", "Penjumlahan"))
        self.actionPenjumlahan.setObjectName("actionPenjumlahan")
        self.actionPenjumlahan.triggered.connect(self.operasiPenjumlahan)
        self.actionPengurangan.setText(_translate("MainWindow", "Pengurangan"))
        self.actionPengurangan.setObjectName("actionPengurangan")
        self.actionPengurangan.triggered.connect(self.operasiPengurangan)
        self.actionPerkalian.setText(_translate("MainWindow", "Perkalian"))
        self.actionPerkalian.setObjectName("actionPerkalian")
        self.actionPerkalian.triggered.connect(self.operasiPerkalian)
        self.actionPembagian.setText(_translate("MainWindow", "Pembagian"))
        self.actionPembagian.setObjectName("actionPembagian")
        self.actionPembagian.triggered.connect(self.operasiPembagian)
        self.actionAND.setText(_translate("MainWindow", "AND"))
        self.actionAND.setObjectName("actionAND")
        self.actionAND.triggered.connect(self.operasiAND)
        self.actionXOR.setText(_translate("MainWindow", "XOR"))
        self.actionXOR.setObjectName("actionXOR")
        self.actionXOR.triggered.connect(self.operasiXOR)
        self.actionNOT.setText(_translate("MainWindow", "NOT"))
        self.actionNOT.setObjectName("actionNOT")
        self.actionNOT.triggered.connect(self.operasiNOT)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
