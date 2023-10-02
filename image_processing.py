from PyQt5 import QtGui
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtGui import QPixmap

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