import colorsys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import image_processing
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPixmapItem
from scipy.signal import convolve2d
import rembg

class Ui_MainWindow(object):
    def openImage(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(None, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)", options=options)
        if file_name:
            try:
                image = QtGui.QPixmap(file_name)
                label_width = self.label.width()
                label_height = self.label.height()
                scaled_image = image.scaled(label_width, label_height, QtCore.Qt.KeepAspectRatio)
                self.label.setPixmap(scaled_image)
                self.label.setAlignment(QtCore.Qt.AlignCenter)
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
            histogram = [0] * 256
            for y in range(height):
                for x in range(width):
                    r, g, b, _ = QtGui.QColor(image.pixel(x, y)).getRgb()
                    gray_value = int((r + g + b) / 3)
                    grayscale_image[y][x] = gray_value
                    histogram[gray_value] += 1
            cumulative_histogram = [sum(histogram[:i+1]) for i in range(256)]

            max_pixel_value = width * height
            normalized_cumulative_histogram = [(cumulative_histogram[i] / max_pixel_value) * 255 for i in range(256)]

            equalized_image = np.zeros((height, width), dtype=np.uint8)
            for y in range(height):
                for x in range(width):
                    equalized_image[y][x] = int(normalized_cumulative_histogram[grayscale_image[y][x]])

            equalized_qimage = QtGui.QImage(equalized_image.data, width, height, width, QtGui.QImage.Format_Grayscale8)
            equalized_pixmap = QtGui.QPixmap.fromImage(equalized_qimage)
            self.label_2.setPixmap(equalized_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)

            plt.figure(figsize=(12, 6))
            plt.subplot(121)
            plt.hist(np.array(grayscale_image).ravel(), bins=256, range=(0, 256), density=True, color='b', alpha=0.6)
            plt.title('Histogram Sebelum Equalization')
            plt.xlabel('Nilai Pixel')
            plt.ylabel('Frekuensi Relatif')

            plt.subplot(122)
            equalized_image_flat = np.array(equalized_image).ravel()
            plt.hist(equalized_image_flat, bins=256, range=(0, 256), density=True, color='r', alpha=0.6)
            plt.title('Histogram Sesudah Equalization')
            plt.xlabel('Nilai Pixel')
            plt.ylabel('Frekuensi Relatif')

            plt.tight_layout()
            plt.show()

    def fuzzyHERGB(self):
        pixmap = self.label.pixmap()
        if pixmap:
            image = pixmap.toImage()
            width = image.width()
            height = image.height()

        equalized_image = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)

        # Menghitung histogram untuk setiap saluran warna
        histograms = [np.zeros(256, dtype=int) for _ in range(3)]
        cumulative_histograms = [np.zeros(256, dtype=int) for _ in range(3)]

        for y in range(height):
            for x in range(width):
                r, g, b, _ = QtGui.QColor(image.pixel(x, y)).getRgb()
                histograms[0][r] += 1
                histograms[1][g] += 1
                histograms[2][b] += 1

        # Menghitung cumulative histogram untuk setiap saluran warna
        for i in range(3):
            cumulative_histograms[i][0] = histograms[i][0]
            for j in range(1, 256):
                cumulative_histograms[i][j] = cumulative_histograms[i][j - 1] + histograms[i][j]

        # Normalisasi cumulative histogram
        max_pixel_value = width * height
        normalized_cumulative_histograms = [cumulative_histograms[i] / max_pixel_value * 255 for i in range(3)]

        # Menerapkan fuzzy equalization pada citra
        for y in range(height):
            for x in range(width):
                r, g, b, _ = QtGui.QColor(image.pixel(x, y)).getRgb()
                new_r = int(normalized_cumulative_histograms[0][r])
                new_g = int(normalized_cumulative_histograms[1][g])
                new_b = int(normalized_cumulative_histograms[2][b])
                equalized_image.setPixel(x, y, QtGui.qRgb(new_r, new_g, new_b))

        equalized_pixmap = QtGui.QPixmap.fromImage(equalized_image)
        self.label_2.setPixmap(equalized_pixmap)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)

        # Buat histogram sebelum fuzzy equalization
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        for i, color in enumerate(['r', 'g', 'b']):
            plt.hist(np.array([QtGui.QColor(image.pixel(x, y)).getRgb()[i] for x in range(width) for y in range(height)]),
                     bins=256, range=(0, 256), density=True, color=color, alpha=0.6, label=color.upper())
        plt.title('Histogram Sebelum Fuzzy Equalization')
        plt.xlabel('Nilai Pixel')
        plt.ylabel('Frekuensi Relatif')
        plt.legend()

        # Buat histogram sesudah fuzzy equalization
        plt.subplot(122)
        for i, color in enumerate(['r', 'g', 'b']):
            plt.hist(np.array([QtGui.QColor(equalized_image.pixel(x, y)).getRgb()[i] for x in range(width) for y in range(height)]),
                     bins=256, range=(0, 256), density=True, color=color, alpha=0.6, label=color.upper())
        plt.title('Histogram Sesudah Fuzzy Equalization')
        plt.xlabel('Nilai Pixel')
        plt.ylabel('Frekuensi Relatif')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def fuzzyGreyscale(self):
        pixmap = self.label.pixmap()
        if pixmap:
            image = pixmap.toImage()
            width = image.width()
            height = image.height()

        grayscale_image = np.zeros((height, width), dtype=np.uint8)

        histogram = [0] * 256
        for y in range(height):
            for x in range(width):
                pixel_value = QtGui.qGray(image.pixel(x, y))
                grayscale_image[y][x] = pixel_value
                histogram[pixel_value] += 1

        cumulative_histogram = [sum(histogram[:i+1]) for i in range(256)]

        max_pixel_value = width * height
        normalized_cumulative_histogram = [(cumulative_histogram[i] / max_pixel_value) * 255 for i in range(256)]

        fuzzy_equalized_image = np.zeros((height, width), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                fuzzy_equalized_image[y][x] = int(normalized_cumulative_histogram[grayscale_image[y][x]])

        fuzzy_equalized_qimage = QtGui.QImage(fuzzy_equalized_image.data, width, height, width, QtGui.QImage.Format_Grayscale8)
        fuzzy_equalized_pixmap = QtGui.QPixmap.fromImage(fuzzy_equalized_qimage)
        self.label_2.setPixmap(fuzzy_equalized_pixmap)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)

        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.hist(np.array(grayscale_image).ravel(), bins=256, range=(0, 256), density=True, color='b', alpha=0.6)
        plt.title('Histogram Sebelum Fuzzy Equalization')
        plt.xlabel('Nilai Pixel')
        plt.ylabel('Frekuensi Relatif')

        plt.subplot(122)
        fuzzy_equalized_image_flat = np.array(fuzzy_equalized_image).ravel()
        plt.hist(fuzzy_equalized_image_flat, bins=256, range=(0, 256), density=True, color='r', alpha=0.6)
        plt.title('Histogram Sesudah Fuzzy Equalization')
        plt.xlabel('Nilai Pixel')
        plt.ylabel('Frekuensi Relatif')
        plt.tight_layout()
        plt.show()

    def translateImage(self):
        original_pixmap = self.label.pixmap()
        if original_pixmap:
            delta_x, _ = QtWidgets.QInputDialog.getInt(None, "Translate Image", "Enter Delta X:")
            delta_y, _ = QtWidgets.QInputDialog.getInt(None, "Translate Image", "Enter Delta Y:")
        
        translated_pixmap = QtGui.QPixmap(original_pixmap)
        painter = QtGui.QPainter(translated_pixmap)
        painter.translate(delta_x, delta_y)
        painter.drawPixmap(0, 0, original_pixmap)
        painter.end()
        
        self.label_2.setPixmap(translated_pixmap)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)

    def uniformScaling(self):
        original_pixmap = self.label.pixmap()
        if original_pixmap:
            scale_factor, _ = QtWidgets.QInputDialog.getDouble(None, "Uniform Scaling", "Enter scaling factor:")
            if scale_factor > 0:
                scaled_pixmap = original_pixmap.scaled(original_pixmap.size() * scale_factor, QtCore.Qt.KeepAspectRatio)
                self.label_2.setPixmap(scaled_pixmap)
                self.label_2.setAlignment(QtCore.Qt.AlignCenter)
            else:
                QtWidgets.QMessageBox.warning(None, "Invalid Input", "Please enter a positive scaling factor.")
    
    def nonUniformScaling(self):
        original_pixmap = self.label.pixmap()
        if original_pixmap:
            scale_factor_x, _ = QtWidgets.QInputDialog.getDouble(None, "Non-Uniform Scaling", "Enter X scaling factor:")
            scale_factor_y, _ = QtWidgets.QInputDialog.getDouble(None, "Non-Uniform Scaling", "Enter Y scaling factor:")
        
        if scale_factor_x > 0 and scale_factor_y > 0:
            width = int(original_pixmap.width() * scale_factor_x)
            height = int(original_pixmap.height() * scale_factor_y)
            scaled_pixmap = original_pixmap.scaled(width, height, QtCore.Qt.KeepAspectRatio)
            self.label_2.setPixmap(scaled_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        else:
            QtWidgets.QMessageBox.warning(None, "Invalid Input", "Please enter positive scaling factors for both X and Y.")

    def setBitDepth(self, bit_depth):
        pixmap = self.label.pixmap()

        if pixmap:
            image = pixmap.toImage()
            result_image = QtGui.QImage(image.size(), QtGui.QImage.Format_ARGB32)

            max_value = (2 ** bit_depth) - 1

            for x in range(image.width()):
                for y in range(image.height()):
                    color = image.pixel(x, y)

                    r, g, b, a = QtGui.qRed(color), QtGui.qGreen(color), QtGui.qBlue(color), QtGui.qAlpha(color)

                    # Sesuaikan nilai komponen warna ke bit depth yang diinginkan
                    r = int((r / 255) * max_value)
                    g = int((g / 255) * max_value)
                    b = int((b / 255) * max_value)

                    # Kembalikan ke nilai 8-bit
                    r = int((r / max_value) * 255)
                    g = int((g / max_value) * 255)
                    b = int((b / max_value) * 255)

                    result_image.setPixel(x, y, QtGui.qRgba(r, g, b, a))

            result_pixmap = QtGui.QPixmap.fromImage(result_image)
            label_width = self.label_2.width()
            label_height = self.label_2.height()
            scaled_image = result_pixmap.scaled(label_width, label_height, QtCore.Qt.KeepAspectRatio)
            self.label_2.setPixmap(scaled_image)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)

    def viewHistogramInput(self):
        original_pixmap = self.label.pixmap()
        if original_pixmap:
            image = original_pixmap.toImage()

            width = image.width()
            height = image.height()

            histogram = [0] * 256

            for y in range(height):
                for x in range(width):
                    r, g, b, _ = QtGui.QColor(image.pixel(x, y)).getRgb()
                    gray_value = int((r + g + b) / 3)
                    histogram[gray_value] += 1

            plt.figure(figsize=(6, 4))
            plt.bar(range(256), histogram, color='b', alpha=0.6)
            plt.title('Histogram Input')
            plt.xlabel('Gray Level')
            plt.ylabel('Frequency')
            plt.xlim(0, 255)
            plt.show()

    def viewHistogramOutput(self):
        output_pixmap = self.label_2.pixmap()
        if output_pixmap:
            output_image = output_pixmap.toImage()

            width = output_image.width()
            height = output_image.height()

            histogram = [0] * 256

            for y in range(height):
                for x in range(width):
                    gray_value = QtGui.qGray(output_image.pixel(x, y))
                    histogram[gray_value] += 1

            plt.figure(figsize=(6, 4))
            plt.bar(range(256), histogram, color='r', alpha=0.6)
            plt.title('Histogram Output')
            plt.xlabel('Gray Level')
            plt.ylabel('Frequency')
            plt.xlim(0, 255)
            plt.show()

    def viewHistogramInputOutput(self):
        original_pixmap = self.label.pixmap()
        output_pixmap = self.label_2.pixmap()
        
        if original_pixmap and output_pixmap:
            original_image = original_pixmap.toImage()
            output_image = output_pixmap.toImage()

            width = original_image.width()
            height = original_image.height()

            original_histogram = [0] * 256
            output_histogram = [0] * 256

            for y in range(height):
                for x in range(width):
                    r, g, b, _ = QtGui.QColor(original_image.pixel(x, y)).getRgb()
                    gray_value = int((r + g + b) / 3)
                    original_histogram[gray_value] += 1
                    
                    gray_value = QtGui.qGray(output_image.pixel(x, y))
                    output_histogram[gray_value] += 1

            plt.figure(figsize=(12, 6))
            plt.subplot(121)
            plt.bar(range(256), original_histogram, color='b', alpha=0.6)
            plt.title('Histogram Input')
            plt.xlabel('Gray Level')
            plt.ylabel('Frequency')
            plt.xlim(0, 255)

            plt.subplot(122)
            plt.bar(range(256), output_histogram, color='r', alpha=0.6)
            plt.title('Histogram Output')
            plt.xlabel('Gray Level')
            plt.ylabel('Frequency')
            plt.xlim(0, 255)

            plt.tight_layout()
            plt.show()
            
    def lowPassFilter(self):
        original_pixmap = self.label.pixmap()
        if original_pixmap:
            original_image = original_pixmap.toImage()
            width = original_image.width()
            height = original_image.height()

            # Define kernel for the low-pass filter (3x3 Gaussian)
            kernel = np.array([[1, 2, 1],
                            [2, 4, 2],
                            [1, 2, 1]])
            kernel = kernel / kernel.sum()

            # Create an empty image for the filtered result
            filtered_image = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32)

            # Apply the filter to each pixel in the image
            for y in range(height):
                for x in range(width):
                    r_sum, g_sum, b_sum = 0, 0, 0
                    kernel_sum = 0

                    for ky in range(-1, 2):
                        for kx in range(-1, 2):
                            pixel = original_image.pixel(x + kx, y + ky)
                            kernel_value = kernel[ky + 1][kx + 1]

                            r_sum += QtGui.qRed(pixel) * kernel_value
                            g_sum += QtGui.qGreen(pixel) * kernel_value
                            b_sum += QtGui.qBlue(pixel) * kernel_value
                            kernel_sum += kernel_value

                    r_value = int(r_sum / kernel_sum)
                    g_value = int(g_sum / kernel_sum)
                    b_value = int(b_sum / kernel_sum)

                    filtered_image.setPixel(x, y, QtGui.qRgb(r_value, g_value, b_value))

            filtered_pixmap = QtGui.QPixmap.fromImage(filtered_image)
            self.label_2.setPixmap(filtered_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)
            
    def highPassFilter(self):
        original_pixmap = self.label.pixmap()
        if original_pixmap:
            original_image = original_pixmap.toImage()
            width = original_image.width()
            height = original_image.height()

            # Define kernel for the low-pass filter (3x3 Gaussian)
            kernel = np.array([[1, 2, 1],
                            [2, 4, 2],
                            [1, 2, 1]])
            kernel = kernel / kernel.sum()

            # Create an empty image for the low-pass filtered result
            lowpass_image = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32)

            # Apply the low-pass filter to each pixel in the image
            for y in range(height):
                for x in range(width):
                    r_sum, g_sum, b_sum = 0, 0, 0
                    kernel_sum = 0

                    for ky in range(-1, 2):
                        for kx in range(-1, 2):
                            pixel = original_image.pixel(x + kx, y + ky)
                            kernel_value = kernel[ky + 1][kx + 1]

                            r_sum += QtGui.qRed(pixel) * kernel_value
                            g_sum += QtGui.qGreen(pixel) * kernel_value
                            b_sum += QtGui.qBlue(pixel) * kernel_value
                            kernel_sum += kernel_value

                    r_value = int(r_sum / kernel_sum)
                    g_value = int(g_sum / kernel_sum)
                    b_value = int(b_sum / kernel_sum)

                    lowpass_image.setPixel(x, y, QtGui.qRgb(r_value, g_value, b_value))

            # Create an empty image for the high-pass filtered result
            highpass_image = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32)

            # Apply the high-pass filter by subtracting the low-pass filtered image from the original image
            for y in range(height):
                for x in range(width):
                    original_color = original_image.pixel(x, y)
                    lowpass_color = lowpass_image.pixel(x, y)

                    r_value = QtGui.qRed(original_color) - QtGui.qRed(lowpass_color)
                    g_value = QtGui.qGreen(original_color) - QtGui.qGreen(lowpass_color)
                    b_value = QtGui.qBlue(original_color) - QtGui.qBlue(lowpass_color)

                    highpass_image.setPixel(x, y, QtGui.qRgb(r_value, g_value, b_value))

            highpass_pixmap = QtGui.QPixmap.fromImage(highpass_image)
            self.label_2.setPixmap(highpass_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)

    def sharpen(self):
        original_pixmap = self.label.pixmap()
        if original_pixmap:
            original_image = original_pixmap.toImage()
            width = original_image.width()
            height = original_image.height()

            # Define a sharpening kernel (3x3)
            kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])

            # Create an empty image for the sharpened result
            sharpened_image = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32)

            # Apply the sharpening filter to each pixel in the image
            for y in range(height):
                for x in range(width):
                    r_sum, g_sum, b_sum = 0, 0, 0

                    for ky in range(-1, 2):
                        for kx in range(-1, 2):
                            pixel = original_image.pixel(x + kx, y + ky)
                            kernel_value = kernel[ky + 1][kx + 1]

                            r_sum += QtGui.qRed(pixel) * kernel_value
                            g_sum += QtGui.qGreen(pixel) * kernel_value
                            b_sum += QtGui.qBlue(pixel) * kernel_value

                    r_value = max(0, min(255, int(r_sum)))
                    g_value = max(0, min(255, int(g_sum)))
                    b_value = max(0, min(255, int(b_sum)))

                    sharpened_image.setPixel(x, y, QtGui.qRgb(r_value, g_value, b_value))

            sharpened_pixmap = QtGui.QPixmap.fromImage(sharpened_image)
            self.label_2.setPixmap(sharpened_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)

    def gaussianBlur3x3(self):
        original_pixmap = self.label.pixmap()
        if original_pixmap:
            original_image = original_pixmap.toImage()
            width = original_image.width()
            height = original_image.height()

            # Define a Gaussian blur kernel (3x3)
            kernel = np.array([[1, 2, 1],
                               [2, 4, 2],
                               [1, 2, 1]])
            kernel = kernel / kernel.sum()

            # Create an empty image for the blurred result
            blurred_image = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32)

            # Apply the Gaussian blur filter to each pixel in the image
            for y in range(height):
                for x in range(width):
                    r_sum, g_sum, b_sum = 0, 0, 0
                    kernel_sum = 0

                    for ky in range(-1, 2):
                        for kx in range(-1, 2):
                            pixel = original_image.pixel(x + kx, y + ky)
                            kernel_value = kernel[ky + 1][kx + 1]

                            r_sum += QtGui.qRed(pixel) * kernel_value
                            g_sum += QtGui.qGreen(pixel) * kernel_value
                            b_sum += QtGui.qBlue(pixel) * kernel_value
                            kernel_sum += kernel_value

                    r_value = int(r_sum / kernel_sum)
                    g_value = int(g_sum / kernel_sum)
                    b_value = int(b_sum / kernel_sum)

                    blurred_image.setPixel(x, y, QtGui.qRgb(r_value, g_value, b_value))

            blurred_pixmap = QtGui.QPixmap.fromImage(blurred_image)
            self.label_2.setPixmap(blurred_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)

    def gaussianBlur5x5(self):
        original_pixmap = self.label.pixmap()
        if original_pixmap:
            original_image = original_pixmap.toImage()
            width = original_image.width()
            height = original_image.height()

            # Define a Gaussian blur kernel (5x5)
            kernel = np.array([[1, 4, 6, 4, 1],
                               [4, 16, 24, 16, 4],
                               [6, 24, 36, 24, 6],
                               [4, 16, 24, 16, 4],
                               [1, 4, 6, 4, 1]])
            kernel = kernel / kernel.sum()

            # Create an empty image for the blurred result
            blurred_image = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32)

            # Apply the Gaussian blur filter to each pixel in the image
            for y in range(height):
                for x in range(width):
                    r_sum, g_sum, b_sum = 0, 0, 0
                    kernel_sum = 0

                    for ky in range(-2, 3):
                        for kx in range(-2, 3):
                            pixel = original_image.pixel(x + kx, y + ky)
                            kernel_value = kernel[ky + 2][kx + 2]

                            r_sum += QtGui.qRed(pixel) * kernel_value
                            g_sum += QtGui.qGreen(pixel) * kernel_value
                            b_sum += QtGui.qBlue(pixel) * kernel_value
                            kernel_sum += kernel_value

                    r_value = int(r_sum / kernel_sum)
                    g_value = int(g_sum / kernel_sum)
                    b_value = int(b_sum / kernel_sum)

                    blurred_image.setPixel(x, y, QtGui.qRgb(r_value, g_value, b_value))

            blurred_pixmap = QtGui.QPixmap.fromImage(blurred_image)
            self.label_2.setPixmap(blurred_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)

    def unsharpMasking(self):
        original_pixmap = self.label.pixmap()
        if original_pixmap:
            original_image = original_pixmap.toImage()
            width = original_image.width()
            height = original_image.height()

            # Define a Gaussian blur kernel (5x5)
            kernel = np.array([[1, 4, 6, 4, 1],
                               [4, 16, 24, 16, 4],
                               [6, 24, 36, 24, 6],
                               [4, 16, 24, 16, 4],
                               [1, 4, 6, 4, 1]])
            kernel = kernel / kernel.sum()

            # Create an empty image for the blurred result
            blurred_image = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32)

            # Apply the Gaussian blur filter to each pixel in the image
            for y in range(height):
                for x in range(width):
                    r_sum, g_sum, b_sum = 0, 0, 0
                    kernel_sum = 0

                    for ky in range(-2, 3):
                        for kx in range(-2, 3):
                            pixel_x = max(0, min(x + kx, width - 1))
                            pixel_y = max(0, min(y + ky, height - 1))
                            pixel = original_image.pixel(pixel_x, pixel_y)
                            kernel_value = kernel[ky + 2][kx + 2]

                            r_sum += QtGui.qRed(pixel) * kernel_value
                            g_sum += QtGui.qGreen(pixel) * kernel_value
                            b_sum += QtGui.qBlue(pixel) * kernel_value
                            kernel_sum += kernel_value

                    r_value = int(r_sum / kernel_sum)
                    g_value = int(g_sum / kernel_sum)
                    b_value = int(b_sum / kernel_sum)

                    blurred_image.setPixel(x, y, QtGui.qRgb(r_value, g_value, b_value))

            # Subtract the blurred image from the original to create the mask
            mask_image = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32)

            for y in range(height):
                for x in range(width):
                    original_color = original_image.pixel(x, y)
                    blurred_color = blurred_image.pixel(x, y)

                    r_value = QtGui.qRed(original_color) - QtGui.qRed(blurred_color)
                    g_value = QtGui.qGreen(original_color) - QtGui.qGreen(blurred_color)
                    b_value = QtGui.qBlue(original_color) - QtGui.qBlue(blurred_color)

                    mask_image.setPixel(x, y, QtGui.qRgb(r_value, g_value, b_value))

            # Add the mask back to the original image for sharpening
            sharpened_image = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32)

            for y in range(height):
                for x in range(width):
                    original_color = original_image.pixel(x, y)
                    mask_color = mask_image.pixel(x, y)

                    r_value = QtGui.qRed(original_color) + QtGui.qRed(mask_color)
                    g_value = QtGui.qGreen(original_color) + QtGui.qGreen(mask_color)
                    b_value = QtGui.qBlue(original_color) + QtGui.qBlue(mask_color)

                    sharpened_image.setPixel(x, y, QtGui.qRgb(r_value, g_value, b_value))

            sharpened_pixmap = QtGui.QPixmap.fromImage(sharpened_image)

            self.label_2.setPixmap(sharpened_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)

    def edgeDetectionSobel(self):
        original_pixmap = self.label.pixmap()
        if original_pixmap:
            original_image = original_pixmap.toImage()
            width = original_image.width()
            height = original_image.height()

            # Define Sobel kernels for gradient calculation
            kernel_x = np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]])

            kernel_y = np.array([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]])

            # Create empty images for gradient magnitude and direction
            gradient_magnitude = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32)
            gradient_direction = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32)

            for y in range(height):
                for x in range(width):
                    gx_r, gx_g, gx_b = 0, 0, 0
                    gy_r, gy_g, gy_b = 0, 0, 0

                    for ky in range(-1, 2):
                        for kx in range(-1, 2):
                            pixel_x = max(0, min(x + kx, width - 1))
                            pixel_y = max(0, min(y + ky, height - 1))
                            pixel = original_image.pixel(pixel_x, pixel_y)

                            kernel_x_value = kernel_x[ky + 1][kx + 1]
                            kernel_y_value = kernel_y[ky + 1][kx + 1]

                            gx_r += QtGui.qRed(pixel) * kernel_x_value
                            gx_g += QtGui.qGreen(pixel) * kernel_x_value
                            gx_b += QtGui.qBlue(pixel) * kernel_x_value

                            gy_r += QtGui.qRed(pixel) * kernel_y_value
                            gy_g += QtGui.qGreen(pixel) * kernel_y_value
                            gy_b += QtGui.qBlue(pixel) * kernel_y_value

                    # Calculate gradient magnitude and direction
                    gradient_mag = int(np.sqrt(gx_r**2 + gy_r**2))
                    gradient_dir = np.arctan2(gy_r, gx_r)

                    gradient_magnitude.setPixel(x, y, QtGui.qRgb(gradient_mag, gradient_mag, gradient_mag))
                    gradient_direction.setPixel(x, y, QtGui.qRgb(int((gradient_dir + np.pi) * 127.5 / np.pi),
                                                                int((gradient_dir + np.pi) * 127.5 / np.pi),
                                                                int((gradient_dir + np.pi) * 127.5 / np.pi)))

            gradient_magnitude_pixmap = QtGui.QPixmap.fromImage(gradient_magnitude)
            self.label_2.setPixmap(gradient_magnitude_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)

            # Optionally, you can display the gradient direction as well
            # gradient_direction_pixmap = QtGui.QPixmap.fromImage(gradient_direction)
            # self.label_3.setPixmap(gradient_direction_pixmap)
            # self.label_3.setAlignment(QtCore.Qt.AlignCenter)


    def edgeDetectionPrewitt(self):
        original_pixmap = self.label.pixmap()
        if original_pixmap:
            original_image = original_pixmap.toImage()
            width = original_image.width()
            height = original_image.height()

            # Define Prewitt kernels for gradient calculation
            kernel_x = np.array([[-1, 0, 1],
                                [-1, 0, 1],
                                [-1, 0, 1]])

            kernel_y = np.array([[-1, -1, -1],
                                [0, 0, 0],
                                [1, 1, 1]])

            # Create an empty image for gradient magnitude
            gradient_magnitude = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32)

            for y in range(height):
                for x in range(width):
                    gx_r, gx_g, gx_b = 0, 0, 0
                    gy_r, gy_g, gy_b = 0, 0, 0

                    for ky in range(-1, 2):
                        for kx in range(-1, 2):
                            pixel_x = max(0, min(x + kx, width - 1))
                            pixel_y = max(0, min(y + ky, height - 1))
                            pixel = original_image.pixel(pixel_x, pixel_y)

                            kernel_x_value = kernel_x[ky + 1][kx + 1]
                            kernel_y_value = kernel_y[ky + 1][kx + 1]

                            gx_r += QtGui.qRed(pixel) * kernel_x_value
                            gx_g += QtGui.qGreen(pixel) * kernel_x_value
                            gx_b += QtGui.qBlue(pixel) * kernel_x_value

                            gy_r += QtGui.qRed(pixel) * kernel_y_value
                            gy_g += QtGui.qGreen(pixel) * kernel_y_value
                            gy_b += QtGui.qBlue(pixel) * kernel_y_value

                    # Calculate gradient magnitude
                    gradient_mag = int(np.sqrt(gx_r**2 + gy_r**2))

                    gradient_magnitude.setPixel(x, y, QtGui.qRgb(gradient_mag, gradient_mag, gradient_mag))

            gradient_magnitude_pixmap = QtGui.QPixmap.fromImage(gradient_magnitude)
            self.label_2.setPixmap(gradient_magnitude_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)


    def edgeDetectionRobert(self):
        original_pixmap = self.label.pixmap()
        if original_pixmap:
            original_image = original_pixmap.toImage()
            width = original_image.width()
            height = original_image.height()

            # Define Robert kernels for gradient calculation
            kernel_x = np.array([[1, 0],
                                [0, -1]])

            kernel_y = np.array([[0, 1],
                                [-1, 0]])

            # Create an empty image for gradient magnitude
            gradient_magnitude = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32)

            for y in range(height):
                for x in range(width):
                    gx_r, gx_g, gx_b = 0, 0, 0
                    gy_r, gy_g, gy_b = 0, 0, 0

                    for ky in range(2):
                        for kx in range(2):
                            pixel_x = max(0, min(x + kx, width - 1))
                            pixel_y = max(0, min(y + ky, height - 1))
                            pixel = original_image.pixel(pixel_x, pixel_y)

                            kernel_x_value = kernel_x[ky][kx]
                            kernel_y_value = kernel_y[ky][kx]

                            gx_r += QtGui.qRed(pixel) * kernel_x_value
                            gx_g += QtGui.qGreen(pixel) * kernel_x_value
                            gx_b += QtGui.qBlue(pixel) * kernel_x_value

                            gy_r += QtGui.qRed(pixel) * kernel_y_value
                            gy_g += QtGui.qGreen(pixel) * kernel_y_value
                            gy_b += QtGui.qBlue(pixel) * kernel_y_value

                    # Calculate gradient magnitude
                    gradient_mag = int(np.sqrt(gx_r**2 + gy_r**2))

                    gradient_magnitude.setPixel(x, y, QtGui.qRgb(gradient_mag, gradient_mag, gradient_mag))

            gradient_magnitude_pixmap = QtGui.QPixmap.fromImage(gradient_magnitude)
            self.label_2.setPixmap(gradient_magnitude_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)

    def konvolusiIdentify(self):
        original_pixmap = self.label.pixmap()
        if original_pixmap:
            original_image = original_pixmap.toImage()
            width = original_image.width()
            height = original_image.height()

            # Define the identity kernel
            kernel = np.array([[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]])

            # Create an empty image for the result
            result_image = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32)

            for y in range(height):
                for x in range(width):
                    # Apply the convolution operation
                    convolution_result = 0
                    for ky in range(3):
                        for kx in range(3):
                            pixel_x = max(0, min(x + kx - 1, width - 1))
                            pixel_y = max(0, min(y + ky - 1, height - 1))
                            pixel = original_image.pixel(pixel_x, pixel_y)

                            kernel_value = kernel[ky][kx]
                            convolution_result += QtGui.qRed(pixel) * kernel_value

                    # Set the pixel value in the result image
                    result_image.setPixel(x, y, QtGui.qRgb(convolution_result, convolution_result, convolution_result))

            # Create a QPixmap from the result image
            result_pixmap = QtGui.QPixmap.fromImage(result_image)

            # Display the result in label_2
            self.label_2.setPixmap(result_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)

    def segmentasiCitra(self):
        pass

    def roiFunction(self):
        pass

    def backgroundRemoval(self):
        # Ambil gambar dari label
        pixmap = self.label.pixmap()

        if pixmap:
            input_pixmap = pixmap
            input_image = input_pixmap.toImage()

            # Simpan pixmap sebagai berkas sementara
            temp_image_path = "temp_image.png"
            input_pixmap.save(temp_image_path)

            # Baca berkas gambar sebagai objek berkas yang terbuka
            with open(temp_image_path, 'rb') as image_file:
                # Gunakan rembg untuk menghapus latar belakang
                output_image_bytes = rembg.remove(image_file.read())

                # Tampilkan hasilnya di label_2
                output_pixmap = QtGui.QPixmap()
                output_pixmap.loadFromData(output_image_bytes)

                self.label_2.setPixmap(output_pixmap)
                self.label_2.setAlignment(QtCore.Qt.AlignCenter)

            # Hapus berkas sementara
            os.remove(temp_image_path)

    def setBrightness(self):
        pass

    def setContrast(self):
        pass

    def setThreshold(self):
        pass

    def performDilationSquare3(self):
        original_pixmap = self.label.pixmap()
        if original_pixmap:
            original_image = original_pixmap.toImage()
            width = original_image.width()
            height = original_image.height()

            # Buat kernel dilasi berbentuk square 3x3
            kernel = np.array([[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]])

            # Buat citra hasil dilasi dengan ukuran yang sama
            result_image = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32)

            for y in range(height):
                for x in range(width):
                    max_value = 0

                    for ky in range(3):
                        for kx in range(3):
                            pixel_x = max(0, min(x + kx - 1, width - 1))
                            pixel_y = max(0, min(y + ky - 1, height - 1))
                            pixel = original_image.pixel(pixel_x, pixel_y)

                            kernel_value = kernel[ky][kx]
                            pixel_value = QtGui.qRed(pixel)  # Ambil komponen merah sebagai contoh

                            # Dilasi: Ambil nilai maksimum dalam jendela 3x3
                            max_value = max(max_value, pixel_value * kernel_value)

                    # Set pixel di citra hasil dengan nilai maksimum
                    result_image.setPixel(x, y, QtGui.qRgb(max_value, max_value, max_value))

            # Buat QPixmap dari citra hasil
            result_pixmap = QtGui.QPixmap.fromImage(result_image)

            # Tampilkan hasil di label_2
            self.label_2.setPixmap(result_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)


    def performDilationSquare5(self):
        original_pixmap = self.label.pixmap()
        if original_pixmap:
            original_image = original_pixmap.toImage()
            width = original_image.width()
            height = original_image.height()

            # Buat kernel dilasi berbentuk square 5x5
            kernel = np.array([[1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1]])

            # Buat citra hasil dilasi dengan ukuran yang sama
            result_image = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32)

            for y in range(height):
                for x in range(width):
                    max_value = 0

                    for ky in range(5):
                        for kx in range(5):
                            pixel_x = max(0, min(x + kx - 2, width - 1))
                            pixel_y = max(0, min(y + ky - 2, height - 1))
                            pixel = original_image.pixel(pixel_x, pixel_y)

                            kernel_value = kernel[ky][kx]
                            pixel_value = QtGui.qRed(pixel)  # Ambil komponen merah sebagai contoh

                            # Dilasi: Ambil nilai maksimum dalam jendela 5x5
                            max_value = max(max_value, pixel_value * kernel_value)

                    # Set pixel di citra hasil dengan nilai maksimum
                    result_image.setPixel(x, y, QtGui.qRgb(max_value, max_value, max_value))

            # Buat QPixmap dari citra hasil
            result_pixmap = QtGui.QPixmap.fromImage(result_image)

            # Tampilkan hasil di label_2
            self.label_2.setPixmap(result_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)

    def performDilationCross3(self):
        original_pixmap = self.label.pixmap()
        if original_pixmap:
            original_image = original_pixmap.toImage()
            width = original_image.width()
            height = original_image.height()

            # Buat kernel dilasi berbentuk cross 3x3
            kernel = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]])

            # Buat citra hasil dilasi dengan ukuran yang sama
            result_image = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32)

            for y in range(height):
                for x in range(width):
                    max_value = 0

                    for ky in range(3):
                        for kx in range(3):
                            # Cek hanya pada piksel yang sesuai dengan kernel
                            if kernel[ky][kx] == 1:
                                pixel_x = max(0, min(x + kx - 1, width - 1))
                                pixel_y = max(0, min(y + ky - 1, height - 1))
                                pixel = original_image.pixel(pixel_x, pixel_y)

                                pixel_value = QtGui.qRed(pixel)  # Ambil komponen merah sebagai contoh

                                # Dilasi: Ambil nilai maksimum dalam jendela cross 3x3
                                max_value = max(max_value, pixel_value)

                    # Set pixel di citra hasil dengan nilai maksimum
                    result_image.setPixel(x, y, QtGui.qRgb(max_value, max_value, max_value))

            # Buat QPixmap dari citra hasil
            result_pixmap = QtGui.QPixmap.fromImage(result_image)

            # Tampilkan hasil di label_2
            self.label_2.setPixmap(result_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)

    def performErosionSquare3(self):
        original_pixmap = self.label.pixmap()
        if original_pixmap:
            original_image = original_pixmap.toImage()
            width = original_image.width()
            height = original_image.height()

            # Buat kernel erosi berbentuk square 3x3
            kernel = np.array([[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]])

            # Buat citra hasil erosi dengan ukuran yang sama
            result_image = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32)

            for y in range(height):
                for x in range(width):
                    min_value = 255

                    for ky in range(3):
                        for kx in range(3):
                            pixel_x = max(0, min(x + kx - 1, width - 1))
                            pixel_y = max(0, min(y + ky - 1, height - 1))
                            pixel = original_image.pixel(pixel_x, pixel_y)

                            kernel_value = kernel[ky][kx]
                            pixel_value = QtGui.qRed(pixel)  # Ambil komponen merah sebagai contoh

                            # Erosi: Ambil nilai minimum dalam jendela 3x3
                            min_value = min(min_value, pixel_value * kernel_value)

                    # Set pixel di citra hasil dengan nilai minimum
                    result_image.setPixel(x, y, QtGui.qRgb(min_value, min_value, min_value))

            # Buat QPixmap dari citra hasil
            result_pixmap = QtGui.QPixmap.fromImage(result_image)

            # Tampilkan hasil di label_2
            self.label_2.setPixmap(result_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)

    def performErosionSquare5(self):
        original_pixmap = self.label.pixmap()
        if original_pixmap:
            original_image = original_pixmap.toImage()
            width = original_image.width()
            height = original_image.height()

            # Buat kernel erosi berbentuk square 5x5
            kernel = np.array([[1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1]])

            # Buat citra hasil erosi dengan ukuran yang sama
            result_image = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32)

            for y in range(height):
                for x in range(width):
                    min_value = 255

                    for ky in range(5):
                        for kx in range(5):
                            pixel_x = max(0, min(x + kx - 2, width - 1))
                            pixel_y = max(0, min(y + ky - 2, height - 1))
                            pixel = original_image.pixel(pixel_x, pixel_y)

                            kernel_value = kernel[ky][kx]
                            pixel_value = QtGui.qRed(pixel)  # Ambil komponen merah sebagai contoh

                            # Erosi: Ambil nilai minimum dalam jendela 5x5
                            min_value = min(min_value, pixel_value * kernel_value)

                    # Set pixel di citra hasil dengan nilai minimum
                    result_image.setPixel(x, y, QtGui.qRgb(min_value, min_value, min_value))

            # Buat QPixmap dari citra hasil
            result_pixmap = QtGui.QPixmap.fromImage(result_image)

            # Tampilkan hasil di label_2
            self.label_2.setPixmap(result_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)

    def performErosionCross3(self):
        original_pixmap = self.label.pixmap()
        if original_pixmap:
            original_image = original_pixmap.toImage()
            width = original_image.width()
            height = original_image.height()

            # Buat kernel erosi berbentuk cross 3x3
            kernel = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]])

            # Buat citra hasil erosi dengan ukuran yang sama
            result_image = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32)

            for y in range(height):
                for x in range(width):
                    min_value = 255

                    for ky in range(3):
                        for kx in range(3):
                            # Cek hanya pada piksel yang sesuai dengan kernel
                            if kernel[ky][kx] == 1:
                                pixel_x = max(0, min(x + kx - 1, width - 1))
                                pixel_y = max(0, min(y + ky - 1, height - 1))
                                pixel = original_image.pixel(pixel_x, pixel_y)

                                pixel_value = QtGui.qRed(pixel)  # Ambil komponen merah sebagai contoh

                                # Erosi: Ambil nilai minimum dalam jendela cross 3x3
                                min_value = min(min_value, pixel_value)

                    # Set pixel di citra hasil dengan nilai minimum
                    result_image.setPixel(x, y, QtGui.qRgb(min_value, min_value, min_value))

            # Buat QPixmap dari citra hasil
            result_pixmap = QtGui.QPixmap.fromImage(result_image)

            # Tampilkan hasil di label_2
            self.label_2.setPixmap(result_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)

    def performOpeningSquare9(self):
        # Lakukan erosi diikuti dengan dilasi menggunakan kernel square 9x9
        self.performErosionSquare9()
        self.performDilationSquare9()

    def performErosionSquare9(self):
        original_pixmap = self.label.pixmap()
        if original_pixmap:
            original_image = original_pixmap.toImage()
            width = original_image.width()
            height = original_image.height()

            # Buat kernel erosi berbentuk square 9x9
            kernel = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1]])

            # Buat citra hasil erosi dengan ukuran yang sama
            result_image = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32)

            for y in range(height):
                for x in range(width):
                    min_value = 255

                    for ky in range(9):
                        for kx in range(9):
                            pixel_x = max(0, min(x + kx - 4, width - 1))
                            pixel_y = max(0, min(y + ky - 4, height - 1))
                            pixel = original_image.pixel(pixel_x, pixel_y)

                            kernel_value = kernel[ky][kx]
                            pixel_value = QtGui.qRed(pixel)  # Ambil komponen merah sebagai contoh

                            # Erosi: Ambil nilai minimum dalam jendela 9x9
                            min_value = min(min_value, pixel_value * kernel_value)

                    # Set pixel di citra hasil dengan nilai minimum
                    result_image.setPixel(x, y, QtGui.qRgb(min_value, min_value, min_value))

            # Buat QPixmap dari citra hasil
            result_pixmap = QtGui.QPixmap.fromImage(result_image)

            # Tampilkan hasil di label_2
            self.label_2.setPixmap(result_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)

    def performDilationSquare9(self):
        original_pixmap = self.label_2.pixmap()  # Gunakan hasil erosi sebagai input untuk dilasi
        if original_pixmap:
            original_image = original_pixmap.toImage()
            width = original_image.width()
            height = original_image.height()

            # Buat kernel dilasi berbentuk square 9x9
            kernel = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1]])

            # Buat citra hasil dilasi dengan ukuran yang sama
            result_image = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32)

            for y in range(height):
                for x in range(width):
                    max_value = 0

                    for ky in range(9):
                        for kx in range(9):
                            pixel_x = max(0, min(x + kx - 4, width - 1))
                            pixel_y = max(0, min(y + ky - 4, height - 1))
                            pixel = original_image.pixel(pixel_x, pixel_y)

                            kernel_value = kernel[ky][kx]
                            pixel_value = QtGui.qRed(pixel)  # Ambil komponen merah sebagai contoh

                            # Dilasi: Ambil nilai maksimum dalam jendela 9x9
                            max_value = max(max_value, pixel_value * kernel_value)

                    # Set pixel di citra hasil dengan nilai maksimum
                    result_image.setPixel(x, y, QtGui.qRgb(max_value, max_value, max_value))

            # Buat QPixmap dari citra hasil
            result_pixmap = QtGui.QPixmap.fromImage(result_image)

            # Tampilkan hasil di label_2
            self.label_2.setPixmap(result_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)

    def performClosingSquare9(self):
        # Lakukan dilasi diikuti dengan erosi menggunakan kernel square 9x9
        self.performDilationSquare9()
        self.performErosionSquare9()

    def performDilationSquare9(self):
        original_pixmap = self.label.pixmap()
        if original_pixmap:
            original_image = original_pixmap.toImage()
            width = original_image.width()
            height = original_image.height()

            # Buat kernel dilasi berbentuk square 9x9
            kernel = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1]])

            # Buat citra hasil dilasi dengan ukuran yang sama
            result_image = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32)

            for y in range(height):
                for x in range(width):
                    max_value = 0

                    for ky in range(9):
                        for kx in range(9):
                            pixel_x = max(0, min(x + kx - 4, width - 1))
                            pixel_y = max(0, min(y + ky - 4, height - 1))
                            pixel = original_image.pixel(pixel_x, pixel_y)

                            kernel_value = kernel[ky][kx]
                            pixel_value = QtGui.qRed(pixel)  # Ambil komponen merah sebagai contoh

                            # Dilasi: Ambil nilai maksimum dalam jendela 9x9
                            max_value = max(max_value, pixel_value * kernel_value)

                    # Set pixel di citra hasil dengan nilai maksimum
                    result_image.setPixel(x, y, QtGui.qRgb(max_value, max_value, max_value))

            # Buat QPixmap dari citra hasil
            result_pixmap = QtGui.QPixmap.fromImage(result_image)

            # Tampilkan hasil di label_2
            self.label_2.setPixmap(result_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)

    def performErosionSquare9(self):
        original_pixmap = self.label_2.pixmap()  # Gunakan hasil dilasi sebagai input untuk erosi
        if original_pixmap:
            original_image = original_pixmap.toImage()
            width = original_image.width()
            height = original_image.height()

            # Buat kernel erosi berbentuk square 9x9
            kernel = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1]])

            # Buat citra hasil erosi dengan ukuran yang sama
            result_image = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32)

            for y in range(height):
                for x in range(width):
                    min_value = 255

                    for ky in range(9):
                        for kx in range(9):
                            pixel_x = max(0, min(x + kx - 4, width - 1))
                            pixel_y = max(0, min(y + ky - 4, height - 1))
                            pixel = original_image.pixel(pixel_x, pixel_y)

                            kernel_value = kernel[ky][kx]
                            pixel_value = QtGui.qRed(pixel)  # Ambil komponen merah sebagai contoh

                            # Erosi: Ambil nilai minimum dalam jendela 9x9
                            min_value = min(min_value, pixel_value * kernel_value)

                    # Set pixel di citra hasil dengan nilai minimum
                    result_image.setPixel(x, y, QtGui.qRgb(min_value, min_value, min_value))

            # Buat QPixmap dari citra hasil
            result_pixmap = QtGui.QPixmap.fromImage(result_image)

            # Tampilkan hasil di label_2
            self.label_2.setPixmap(result_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)

    def featureExtractionRGB(self):
        original_pixmap = self.label.pixmap()
        if original_pixmap:
            original_image = original_pixmap.toImage()
            width = original_image.width()
            height = original_image.height()

            # Buat citra hasil ekstraksi fitur dengan ukuran yang sama
            result_image = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32)

            for y in range(height):
                for x in range(width):
                    pixel = original_image.pixel(x, y)

                    # Ambil komponen warna RGB dari piksel
                    red = QtGui.qRed(pixel)
                    green = QtGui.qGreen(pixel)
                    blue = QtGui.qBlue(pixel)

                    # Hitung nilai ekstraksi fitur (misalnya, rata-rata komponen warna)
                    feature_value = (red + green + blue) // 3  # Contoh: Rata-rata komponen warna

                    # Set pixel di citra hasil dengan nilai ekstraksi fitur
                    result_image.setPixel(x, y, QtGui.qRgb(feature_value, feature_value, feature_value))

            # Buat QPixmap dari citra hasil ekstraksi fitur
            result_pixmap = QtGui.QPixmap.fromImage(result_image)

            # Tampilkan hasil di label_2
            self.label_2.setPixmap(result_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)

    def featureExtractionRGBtoHSV(self):
        original_pixmap = self.label.pixmap()
        if original_pixmap:
            original_image = original_pixmap.toImage()
            width = original_image.width()
            height = original_image.height()

            # Buat citra hasil ekstraksi fitur dengan ukuran yang sama
            result_image = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32)

            for y in range(height):
                for x in range(width):
                    pixel = original_image.pixel(x, y)

                    # Ambil komponen warna RGB dari piksel
                    red = QtGui.qRed(pixel)
                    green = QtGui.qGreen(pixel)
                    blue = QtGui.qBlue(pixel)

                    # Ubah dari RGB ke HSV
                    hsv = colorsys.rgb_to_hsv(red / 255.0, green / 255.0, blue / 255.0)

                    # Hitung nilai ekstraksi fitur (misalnya, nilai hue)
                    feature_value = int(hsv[0] * 255)  # Contoh: Ekstraksi nilai hue

                    # Set pixel di citra hasil dengan nilai ekstraksi fitur
                    result_image.setPixel(x, y, QtGui.qRgb(feature_value, feature_value, feature_value))

            # Buat QPixmap dari citra hasil ekstraksi fitur
            result_pixmap = QtGui.QPixmap.fromImage(result_image)

            # Tampilkan hasil di label_2
            self.label_2.setPixmap(result_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)

    def featureExtractionRGBtoYCrCb(self):
        original_pixmap = self.label.pixmap()
        if original_pixmap:
            original_image = original_pixmap.toImage()
            width = original_image.width()
            height = original_image.height()

            # Buat citra hasil ekstraksi fitur dengan ukuran yang sama
            result_image = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32)

            for y in range(height):
                for x in range(width):
                    pixel = original_image.pixel(x, y)

                    # Ambil komponen warna RGB dari piksel
                    red = QtGui.qRed(pixel)
                    green = QtGui.qGreen(pixel)
                    blue = QtGui.qBlue(pixel)

                    # Ubah dari RGB ke YCrCb
                    ycrcb = cv2.cvtColor(np.array([[[red, green, blue]]], dtype=np.uint8), cv2.COLOR_RGB2YCrCb)[0][0]

                    # Hitung nilai ekstraksi fitur (misalnya, nilai Cr)
                    feature_value = ycrcb[1]  # Contoh: Ekstraksi nilai Cr

                    # Set pixel di citra hasil dengan nilai ekstraksi fitur
                    result_image.setPixel(x, y, QtGui.qRgb(feature_value, feature_value, feature_value))

            # Buat QPixmap dari citra hasil ekstraksi fitur
            result_pixmap = QtGui.QPixmap.fromImage(result_image)

            # Tampilkan hasil di label_2
            self.label_2.setPixmap(result_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1510, 820)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(7, 10, 740, 740))
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setText("")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(759, 10, 740, 740))
        self.label_2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1415, 31))
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

        self.actionUniformScaling = QtWidgets.QAction(MainWindow)
        self.actionUniformScaling.setObjectName("actionUniformScaling")
        self.actionUniformScaling.triggered.connect(self.uniformScaling)  # Connect to your method

        self.actionNonUniformScaling = QtWidgets.QAction(MainWindow)
        self.actionNonUniformScaling.setObjectName("actionNonUniformScaling")
        self.actionNonUniformScaling.triggered.connect(self.nonUniformScaling)  # Connect to your method

        self.actionTranslation = QtWidgets.QAction(MainWindow)
        self.actionTranslation.setObjectName("actionTranslation")
        self.actionTranslation.triggered.connect(self.translateImage)  # Connect to your method

        self.menuBit_Depth = QtWidgets.QMenu(self.menubar)  # Menu Bit Depth di luar menu lainnya
        self.menuBit_Depth.setObjectName("menuBit_Depth")
        self.menuBit_Depth.setTitle("Bit Depth")  # Teks untuk menu Bit Depth

        # Menambahkan menu "View Histogram" dan submenu
        self.menuViewHistogram = QtWidgets.QMenu(self.menubar)
        self.menuViewHistogram.setObjectName("menuViewHistogram")
        self.menuViewHistogram.setTitle("View Histogram")

        self.menuKonvolusi = QtWidgets.QMenu(self.menubar)
        self.menuKonvolusi.setObjectName("menuKonvolusi")
        self.menuKonvolusi.setTitle("Konvolusi")

        self.actionLowPassFilter = QtWidgets.QAction(MainWindow)
        self.actionLowPassFilter.setObjectName("actionLowPassFilter")
        self.actionLowPassFilter.setText("Low Pass Filter")
        self.actionLowPassFilter.triggered.connect(self.lowPassFilter) 

        self.actionHighPassFilter = QtWidgets.QAction(MainWindow)
        self.actionHighPassFilter.setObjectName("actionHighPassFilter")
        self.actionHighPassFilter.setText("High Pass Filter")
        self.actionHighPassFilter.triggered.connect(self.highPassFilter) 

        self.actionIdentify = QtWidgets.QAction(MainWindow)
        self.actionIdentify.setObjectName("actionIdentify")
        self.actionIdentify.setText("Identify")
        self.actionIdentify.triggered.connect(self.konvolusiIdentify)

        self.actionSharpen = QtWidgets.QAction(MainWindow)
        self.actionSharpen.setObjectName("actionSharpen")
        self.actionSharpen.setText("Sharpen")
        self.actionSharpen.triggered.connect(self.sharpen)

        self.actionGaussianBlur3x3 = QtWidgets.QAction(MainWindow)
        self.actionGaussianBlur3x3.setObjectName("actionGaussianBlur3x3")
        self.actionGaussianBlur3x3.setText("Gaussian Blur 3x3")
        self.actionGaussianBlur3x3.triggered.connect(self.gaussianBlur3x3)

        self.actionGaussianBlur5x5 = QtWidgets.QAction(MainWindow)
        self.actionGaussianBlur5x5.setObjectName("actionGaussianBlur5x5")
        self.actionGaussianBlur5x5.setText("Gaussian Blur 5x5")
        self.actionGaussianBlur5x5.triggered.connect(self.gaussianBlur5x5)

        self.actionUnsharpMasking = QtWidgets.QAction(MainWindow)
        self.actionUnsharpMasking.setObjectName("actionUnsharpMasking")
        self.actionUnsharpMasking.setText("Unsharp Masking")
        self.actionUnsharpMasking.triggered.connect(self.unsharpMasking)

        # Add a new submenu "Edge Detection" with submenus "Sobel," "Prewitt," and "Robert"
        self.menuEdgeDetection = QtWidgets.QMenu(self.menuKonvolusi)
        self.menuEdgeDetection.setObjectName("menuEdgeDetection")
        self.menuEdgeDetection.setTitle("Edge Detection")

        self.actionSobel = QtWidgets.QAction(MainWindow)
        self.actionSobel.setObjectName("actionSobel")
        self.actionSobel.setText("Sobel")
        self.actionSobel.triggered.connect(self.edgeDetectionSobel)

        self.actionPrewitt = QtWidgets.QAction(MainWindow)
        self.actionPrewitt.setObjectName("actionPrewitt")
        self.actionPrewitt.setText("Prewitt")
        self.actionPrewitt.triggered.connect(self.edgeDetectionPrewitt)

        self.actionRobert = QtWidgets.QAction(MainWindow)
        self.actionRobert.setObjectName("actionRobert")
        self.actionRobert.setText("Robert")
        self.actionRobert.triggered.connect(self.edgeDetectionRobert)

        self.menuMagic = QtWidgets.QMenu(self.menubar)
        self.menuMagic.setObjectName("menuMagic")
        self.menuMagic.setTitle("Magic")

        self.actionSegmentasiCitra = QtWidgets.QAction(MainWindow)
        self.actionSegmentasiCitra.setObjectName("actionSegmentasiCitra")
        self.actionSegmentasiCitra.setText("Segmentasi Citra")
        self.actionSegmentasiCitra.triggered.connect(self.segmentasiCitra)

        self.actionROI = QtWidgets.QAction(MainWindow)
        self.actionROI.setObjectName("actionROI")
        self.actionROI.setText("ROI")
        self.actionROI.triggered.connect(self.roiFunction)

        self.actionBackgroundRemoval = QtWidgets.QAction(MainWindow)
        self.actionBackgroundRemoval.setObjectName("actionBackgroundRemoval")
        self.actionBackgroundRemoval.setText("Background Removal")
        self.actionBackgroundRemoval.triggered.connect(self.backgroundRemoval)

        self.menuColors = QtWidgets.QMenu(self.menubar)
        self.menuColors.setObjectName("menuColors")
        self.menuColors.setTitle("Colors")  # Teks untuk menu "Colors"

        # Tambahkan submenu "Brightness" ke menu "Colors" dan tambahkan tindakan untuk mengatur kecerahan
        self.actionBrightness = QtWidgets.QAction(MainWindow)
        self.actionBrightness.setObjectName("actionBrightness")
        self.actionBrightness.setText("Brightness")
        self.actionBrightness.triggered.connect(self.setBrightness)  # Gantilah ini dengan fungsi yang sesuai

        self.menuColors.addAction(self.actionBrightness)

        # Tambahkan submenu "Contrast" ke menu "Colors" dan tambahkan tindakan untuk mengatur kontras
        self.actionContrast = QtWidgets.QAction(MainWindow)
        self.actionContrast.setObjectName("actionContrast")
        self.actionContrast.setText("Contrast")
        self.actionContrast.triggered.connect(self.setContrast)  # Gantilah ini dengan fungsi yang sesuai

        self.actionThreshold = QtWidgets.QAction(MainWindow)
        self.actionThreshold.setObjectName("actionThreshold")
        self.actionThreshold.setText("Threshold")
        self.actionThreshold.triggered.connect(self.setThreshold) 

        self.menuMorphology = QtWidgets.QMenu(self.menubar)
        self.menuMorphology.setObjectName("menuMorphology")
        self.menuMorphology.setTitle("Morfologi")

        self.menuDilation = QtWidgets.QMenu(self.menuMorphology)
        self.menuDilation.setObjectName("menuDilation")
        self.menuDilation.setTitle("Dilasi")

        self.actionDilationSquare3 = QtWidgets.QAction(MainWindow)
        self.actionDilationSquare3.setObjectName("actionDilationSquare5")
        self.actionDilationSquare3.setText("Square 3")
        self.actionDilationSquare3.triggered.connect(self.performDilationSquare3)
        self.menuDilation.addAction(self.actionDilationSquare3)

        self.actionDilationSquare5 = QtWidgets.QAction(MainWindow)
        self.actionDilationSquare5.setObjectName("actionDilationSquare5")
        self.actionDilationSquare5.setText("Square 5")
        self.actionDilationSquare5.triggered.connect(self.performDilationSquare5)
        self.menuDilation.addAction(self.actionDilationSquare5)

        self.actionDilationCross3 = QtWidgets.QAction(MainWindow)
        self.actionDilationCross3.setObjectName("actionDilationCross3")
        self.actionDilationCross3.setText("Cross 3")
        self.actionDilationCross3.triggered.connect(self.performDilationCross3)
        self.menuDilation.addAction(self.actionDilationCross3)

        self.menuerosion = QtWidgets.QMenu(self.menuMorphology)
        self.menuerosion.setObjectName("menuerosion")
        self.menuerosion.setTitle("Erosi")

        self.actionErosionSquare3 = QtWidgets.QAction(MainWindow)
        self.actionErosionSquare3.setObjectName("actionErosionSquare3")
        self.actionErosionSquare3.setText("Square 3")
        self.actionErosionSquare3.triggered.connect(self.performErosionSquare3)
        self.menuerosion.addAction(self.actionErosionSquare3)

        self.actionErosionSquare5 = QtWidgets.QAction(MainWindow)
        self.actionErosionSquare5.setObjectName("actionErosionSquare5")
        self.actionErosionSquare5.setText("Square 5")
        self.actionErosionSquare5.triggered.connect(self.performErosionSquare5)
        self.menuerosion.addAction(self.actionErosionSquare5)

        self.actionErosionCross3 = QtWidgets.QAction(MainWindow)
        self.actionErosionCross3.setObjectName("actionErosionCross3")
        self.actionErosionCross3.setText("Cross 3")
        self.actionErosionCross3.triggered.connect(self.performErosionCross3)
        self.menuerosion.addAction(self.actionErosionCross3)

        self.menuOpening = QtWidgets.QMenu(self.menuMorphology)
        self.menuOpening.setObjectName("menuOpening")
        self.menuOpening.setTitle("Opening")

        self.actionOpeningSquare9 = QtWidgets.QAction(MainWindow)
        self.actionOpeningSquare9.setObjectName("actionOpeningSquare9")
        self.actionOpeningSquare9.setText("Square 9")
        self.actionOpeningSquare9.triggered.connect(self.performOpeningSquare9)
        self.menuOpening.addAction(self.actionOpeningSquare9)

        self.menuClosing = QtWidgets.QMenu(self.menuMorphology)
        self.menuClosing.setObjectName("menuClosing")
        self.menuClosing.setTitle("Closing")

        self.actionClosingSquare9 = QtWidgets.QAction(MainWindow)
        self.actionClosingSquare9.setObjectName("actionClosingSquare9")
        self.actionClosingSquare9.setText("Square 3")
        self.actionClosingSquare9.triggered.connect(self.performClosingSquare9)
        self.menuClosing.addAction(self.actionClosingSquare9)

        # Tambahkan ini di dalam method setupUi setelah definisi menuHistogram_Processing
        self.menuFeature_Extraction = QtWidgets.QMenu(self.menubar)
        self.menuFeature_Extraction.setObjectName("menuFeature_Extraction")
        self.menuFeature_Extraction.setTitle("Ekstraksi Fitur")

        # Submenu 1: RGB
        self.actionRGB = QtWidgets.QAction(MainWindow)
        self.actionRGB.setObjectName("actionRGB")
        self.actionRGB.setText("RGB")
        self.actionRGB.triggered.connect(self.featureExtractionRGB)

        # Submenu 2: RGB to HSV
        self.actionRGB_to_HSV = QtWidgets.QAction(MainWindow)
        self.actionRGB_to_HSV.setObjectName("actionRGB_to_HSV")
        self.actionRGB_to_HSV.setText("RGB to HSV")
        self.actionRGB_to_HSV.triggered.connect(self.featureExtractionRGBtoHSV)

        # Submenu 3: RGB to YCrCb
        self.actionRGB_to_YCrCb = QtWidgets.QAction(MainWindow)
        self.actionRGB_to_YCrCb.setObjectName("actionRGB_to_YCrCb")
        self.actionRGB_to_YCrCb.setText("RGB to YCrCb")
        self.actionRGB_to_YCrCb.triggered.connect(self.featureExtractionRGBtoYCrCb)

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
        self.menuImage_Geometri.addAction(self.actionFlipVertical)
        self.menuImage_Geometri.addAction(self.actionRotateClockwise)
        self.menuHistogram_Processing.addAction(self.actionHistogram_Equalization)
        self.menuHistogram_Processing.addAction(self.actionFuzzy_HE_RGB)
        self.menuHistogram_Processing.addAction(self.actionFuzzy_Greyscale)
        self.menuImage_Geometri.addAction(self.actionUniformScaling)
        self.menuImage_Geometri.addAction(self.actionNonUniformScaling)
        self.menuImage_Geometri.addAction(self.actionTranslation)

        # submenu Bit Depth
        self.action1_Bit = QtWidgets.QAction(MainWindow)
        self.action1_Bit.setObjectName("action1_Bit")
        self.action1_Bit.triggered.connect(lambda: self.setBitDepth(1))
        self.menuBit_Depth.addAction(self.action1_Bit)
        
        self.action2_Bit = QtWidgets.QAction(MainWindow)
        self.action2_Bit.setObjectName("action2_Bit")
        self.action2_Bit.triggered.connect(lambda: self.setBitDepth(2))
        self.menuBit_Depth.addAction(self.action2_Bit)
        
        self.action3_Bit = QtWidgets.QAction(MainWindow)
        self.action3_Bit.setObjectName("action3_Bit")
        self.action3_Bit.triggered.connect(lambda: self.setBitDepth(3))
        self.menuBit_Depth.addAction(self.action3_Bit)
        
        self.action4_Bit = QtWidgets.QAction(MainWindow)
        self.action4_Bit.setObjectName("action4_Bit")
        self.action4_Bit.triggered.connect(lambda: self.setBitDepth(4))
        self.menuBit_Depth.addAction(self.action4_Bit)
        
        self.action5_Bit = QtWidgets.QAction(MainWindow)
        self.action5_Bit.setObjectName("action5_Bit")
        self.action5_Bit.triggered.connect(lambda: self.setBitDepth(5))
        self.menuBit_Depth.addAction(self.action5_Bit)
        
        self.action6_Bit = QtWidgets.QAction(MainWindow)
        self.action6_Bit.setObjectName("action6_Bit")
        self.action6_Bit.triggered.connect(lambda: self.setBitDepth(6))
        self.menuBit_Depth.addAction(self.action6_Bit)
        
        self.action7_Bit = QtWidgets.QAction(MainWindow)
        self.action7_Bit.setObjectName("action7_Bit")
        self.action7_Bit.triggered.connect(lambda: self.setBitDepth(7))
        self.menuBit_Depth.addAction(self.action7_Bit)

         # submenu View Histogram
        self.actionHistogramInput = QtWidgets.QAction(MainWindow)
        self.actionHistogramInput.setObjectName("actionHistogramInput")
        self.actionHistogramInput.triggered.connect(self.viewHistogramInput)
        self.actionHistogramOutput = QtWidgets.QAction(MainWindow)
        self.actionHistogramOutput.setObjectName("actionHistogramOutput")
        self.actionHistogramOutput.triggered.connect(self.viewHistogramOutput)
        self.actionHistogramInputOutput = QtWidgets.QAction(MainWindow)
        self.actionHistogramInputOutput.setObjectName("actionHistogramInputOutput")
        self.actionHistogramInputOutput.triggered.connect(self.viewHistogramInputOutput)

        self.menuKonvolusi.addAction(self.actionLowPassFilter)
        self.menuKonvolusi.addAction(self.actionHighPassFilter)
        self.menuKonvolusi.addAction(self.actionIdentify)
        self.menuKonvolusi.addAction(self.actionSharpen)
        self.menuKonvolusi.addAction(self.actionGaussianBlur3x3)
        self.menuKonvolusi.addAction(self.actionGaussianBlur5x5)
        self.menuKonvolusi.addAction(self.actionUnsharpMasking)

        self.menuEdgeDetection.addAction(self.actionSobel)
        self.menuEdgeDetection.addAction(self.actionPrewitt)
        self.menuEdgeDetection.addAction(self.actionRobert)

        self.menuMagic.addAction(self.actionSegmentasiCitra)
        self.menuMagic.addAction(self.actionROI)
        self.menuMagic.addAction(self.actionBackgroundRemoval)

        self.menuColors.addAction(self.actionBrightness)
        self.menuColors.addAction(self.actionContrast)
        self.menuColors.addAction(self.actionThreshold)
        
        self.menuKonvolusi.addMenu(self.menuEdgeDetection)
        self.menuMorphology.addMenu(self.menuDilation)
        self.menuMorphology.addMenu(self.menuerosion)
        self.menuMorphology.addMenu(self.menuOpening)
        self.menuMorphology.addMenu(self.menuClosing)

        self.menuFeature_Extraction.addAction(self.actionRGB)
        self.menuFeature_Extraction.addAction(self.actionRGB_to_HSV)
        self.menuFeature_Extraction.addAction(self.actionRGB_to_YCrCb)
        
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuImage_Processing.menuAction())
        self.menubar.addAction(self.menuImage_Geometri.menuAction()) 
        self.menubar.addAction(self.menuHistogram_Processing.menuAction())
        self.menubar.addAction(self.menuAritmatics_Operation.menuAction())
        self.menubar.addMenu(self.menuColors)
        self.menubar.addAction(self.menuBit_Depth.menuAction()) 
        self.menubar.addAction(self.menuViewHistogram.menuAction())
        self.menubar.addAction(self.menuKonvolusi.menuAction())
        self.menubar.addAction(self.menuMagic.menuAction())
        self.menubar.addMenu(self.menuMorphology)
        self.menubar.addMenu(self.menuFeature_Extraction)
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
        self.actionUniformScaling.setText(_translate("MainWindow", "Uniform Scalling"))
        self.actionNonUniformScaling.setText(_translate("MainWindow", "Non Uniform Scalling"))
        self.actionTranslation.setText(_translate("MainWindow", "Translation"))
        self.actionHistogram_Equalization.setText(_translate("MainWindow", "Histogram Equalization"))
        self.actionFuzzy_HE_RGB.setText(_translate("MainWindow", "Fuzzy HE RGB"))
        self.actionFuzzy_Greyscale.setText(_translate("MainWindow", "Fuzzy Greyscale"))
        self.menuBit_Depth.setTitle(_translate("MainWindow", "Bit Depth"))

        self.action1_Bit.setText(_translate("MainWindow", "1 bit"))
        self.action2_Bit.setText(_translate("MainWindow", "2 bit"))
        self.action3_Bit.setText(_translate("MainWindow", "3 bit"))
        self.action4_Bit.setText(_translate("MainWindow", "4 bit"))
        self.action5_Bit.setText(_translate("MainWindow", "5 bit"))
        self.action6_Bit.setText(_translate("MainWindow", "6 bit"))
        self.action7_Bit.setText(_translate("MainWindow", "7 bit"))

        self.menuViewHistogram.addAction(self.actionHistogramInput)
        self.menuViewHistogram.addAction(self.actionHistogramOutput)
        self.menuViewHistogram.addAction(self.actionHistogramInputOutput)

        # Teks menu "View Histogram"
        self.menuViewHistogram.setTitle(_translate("MainWindow", "View Histogram"))
        self.actionHistogramInput.setText(_translate("MainWindow", "Histogram Input"))
        self.actionHistogramOutput.setText(_translate("MainWindow", "Histogram Output"))
        self.actionHistogramInputOutput.setText(_translate("MainWindow", "Histogram Input dan Output"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())