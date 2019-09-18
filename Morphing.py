#! /usr/bin/env python3.4
from scipy.interpolate import polate
from scipy.sparse import find
import imageio as io
import numpy as np
from scipy.spatial import Delaunay
from PIL import Image, ImageDraw

import time
import os


class Affine:
    def __init__(self, source, destination):
        if source.dtype != 'float64':
            raise ValueError("source must be a numpy array of type float64.")
        if source.shape != (3, 2):
            raise ValueError("source must be a 3x2 array.")

        if destination.dtype != 'float64':
            raise ValueError("destination must be a numpy array of type float64.")
        if destination.shape != (3, 2):
            raise ValueError("destination must be a 3x2 array.")

        A = np.array([[source[0, 0], source[0, 1], 1, 0, 0, 0],
                      [0, 0, 0, source[0, 0], source[0, 1], 1],
                      [source[1, 0], source[1, 1], 1, 0, 0, 0],
                      [0, 0, 0, source[1, 0], source[1, 1], 1],
                      [source[2, 0], source[2, 1], 1, 0, 0, 0],
                      [0, 0, 0, source[2, 0], source[2, 1], 1]], dtype = 'float64')

        b = np.reshape(destination, (6, 1))
        h = np.linalg.solve(A, b)
        matrix = np.vstack([np.reshape(h, (2, 3)), [0, 0, 1]])

        self.source = source
        self.destination = destination
        self.matrix = matrix

    def transform(self, sourceImage, destinationImage):
        if not isinstance(sourceImage, np.ndarray):
            raise TypeError("sourceImage must be a numpy array.")

        if not isinstance(destinationImage, np.ndarray):
            raise TypeError("destinationImage must be a numpy array.")

        hInv = np.linalg.inv(self.matrix)
        getCoord = np.vectorize(lambda x, y, a: hInv[a, 0] * y + hInv[a, 1] * x + hInv[a, 2], otypes = [np.float64])

        xRange = np.arange(np.amin(self.source[:, 1]), np.amax(self.source[:, 1]), 1)
        yRange = np.arange(np.amin(self.source[:, 0]), np.amax(self.source[:, 0]), 1)
        zRange = sourceImage[int(xRange[0]):int(xRange[-1] + 1), int(yRange[0]):int(yRange[-1] + 1)]
        spline = interpolate.RectBivariateSpline(xRange, yRange, zRange, kx = 1, ky = 1)
        mask = Image.new('L', (destinationImage.shape[1], destinationImage.shape[0]), 0)
        ImageDraw.Draw(mask).polygon(self.destination.ravel().tolist(), outline = 255, fill = 255)
        xp, yp = np.nonzero(mask)
        x = getCoord(xp, yp, 1)
        y = getCoord(xp, yp, 0)
        destinationImage[xp, yp] = spline.ev(x, y)

    def _Mask(self, destinationImage):

        height = destinationImage.shape[0]
        width = destinationImage.shape[1]
        image = Image.new('L', (width, height), 0)

        # Draw the triangle, using tuple vertices and fill it in with a white value
        vertices = [(self.destination[0, 0], self.destination[0, 1]),
                    (self.destination[1, 0], self.destination[1, 1]),
                    (self.destination[2, 0], self.destination[2, 1])]

        ImageDraw.Draw(image).polygon(vertices, outline=255, fill=255)

        # Convert to numpy array
        return np.array(image)

class Blender:
    def __init__(self, startImage, startPoints, endImage, endPoints):
        if not isinstance(startImage, np.ndarray):
            raise TypeError("startImage must be a numpy array.")

        if not isinstance(startPoints, np.ndarray):
            raise TypeError("startPoints must be a numpy array.")

        if not isinstance(endImage, np.ndarray):
            raise TypeError("endImage must be a numpy array.")

        if not isinstance(endPoints, np.ndarray):
            raise TypeError("endPoints must be a numpy array.")

        # Define triangulation.
        simplices = spatial.Delaunay(startPoints).simplices

        # Initialize blender data.
        self.startImage = startImage
        self.startPoints = startPoints
        self.endImage = endImage
        self.endPoints = endPoints
        self.simplices = simplices

    def getBlendedImage(self, alpha):
        # Initialize blended image components.
        target1 = np.empty(self.startImage.shape, dtype = 'float64')
        target2 = np.empty(self.endImage.shape, dtype = 'float64')

        # Process all triangles.
        targets = (1 - alpha) * self.startPoints + alpha * self.endPoints
        for triangle in self.simplices.tolist():
            # Define relevant points.
            src = self.startPoints[triangle]
            dst = self.endPoints[triangle]
            tar = targets[triangle]

            # Define affine transforms.
            Affine(src, tar).transform(self.startImage, target1)
            Affine(dst, tar).transform(self.endImage, target2)

        # Alpha blend images.
        target = (1 - alpha) * target1 + alpha * target2

        return target.astype('uint8')


class ColorAffine:
    def __init__(self, source, destination):
        # Validate source input.
        if source.dtype != 'float64':
            raise ValueError("source must be a numpy array of type float64.")
        if source.shape != (3, 2):
            raise ValueError("source must be a 3x2 array.")

        # Validate destination input.
        if destination.dtype != 'float64':
            raise ValueError("destination must be a numpy array of type float64.")
        if destination.shape != (3, 2):
            raise ValueError("destination must be a 3x2 array.")

        # Define combined source data matrix.
        A = np.array([[source[0, 0], source[0, 1], 1, 0, 0, 0],
                      [0, 0, 0, source[0, 0], source[0, 1], 1],
                      [source[1, 0], source[1, 1], 1, 0, 0, 0],
                      [0, 0, 0, source[1, 0], source[1, 1], 1],
                      [source[2, 0], source[2, 1], 1, 0, 0, 0],
                      [0, 0, 0, source[2, 0], source[2, 1], 1]], dtype = 'float64')

        # Define combined destination matrix.
        b = np.reshape(destination, (6, 1))

        # Solve for transformation matrix.
        h = np.linalg.solve(A, b)

        # Define combined transformation matrix data.
        matrix = np.vstack([np.reshape(h, (2, 3)), [0, 0, 1]])

        # Initialize affine transform data.
        self.source = source
        self.destination = destination
        self.matrix = matrix

    def transform(self, sourceImage, destinationImage):
        # Validate sourceImage input.
        if not isinstance(sourceImage, np.ndarray):
            raise TypeError("sourceImage must be a numpy array.")

        # Validate destinationImage input.
        if not isinstance(destinationImage, np.ndarray):
            raise TypeError("destinationImage must be a numpy array.")

        # Define vectorizations.
        hInv = np.linalg.inv(self.matrix)
        getCoord = np.vectorize(lambda x, y, a: hInv[a, 0] * y + hInv[a, 1] * x + hInv[a, 2], otypes = [np.float64])

        # Generate spline.
        xRange = np.arange(np.amin(self.source[:, 1]), np.amax(self.source[:, 1]), 1)
        yRange = np.arange(np.amin(self.source[:, 0]), np.amax(self.source[:, 0]), 1)
        zRange = sourceImage[int(xRange[0]):int(xRange[-1] + 1), int(yRange[0]):int(yRange[-1] + 1)]
        rSpline = interpolate.RectBivariateSpline(xRange, yRange, np.compress([1, 0, 0], zRange, axis = 2), kx = 1, ky = 1)
        gSpline = interpolate.RectBivariateSpline(xRange, yRange, np.compress([0, 1, 0], zRange, axis = 2), kx = 1, ky = 1)
        bSpline = interpolate.RectBivariateSpline(xRange, yRange, np.compress([0, 0, 1], zRange, axis = 2), kx = 1, ky = 1)


        # Generate transformation mask.
        mask = Image.new('L', (destinationImage.shape[1], destinationImage.shape[0]), 0)
        ImageDraw.Draw(mask).polygon(self.destination.ravel().tolist(), outline = 255, fill = 255)
        xp, yp = np.nonzero(mask)

        # Find transformed coordinates and values.
        x = getCoord(xp, yp, 1)
        y = getCoord(xp, yp, 0)
        destinationImage[xp, yp] = np.transpose([rSpline.ev(x, y), gSpline.ev(x, y), bSpline.ev(x, y)])

class ColorBlender:
    def __init__(self, startImage, startPoints, endImage, endPoints):
        if not isinstance(startImage, np.ndarray):
            raise TypeError("startImage must be a numpy array.")

        if not isinstance(startPoints, np.ndarray):
            raise TypeError("startPoints must be a numpy array.")

        if not isinstance(endImage, np.ndarray):
            raise TypeError("endImage must be a numpy array.")

        # Validate endPoints input.
        if not isinstance(endPoints, np.ndarray):
            raise TypeError("endPoints must be a numpy array.")

        # Define triangulation.
        simplices = spatial.Delaunay(startPoints).simplices

        # Initialize blender data.
        self.startImage = startImage
        self.startPoints = startPoints
        self.endImage = endImage
        self.endPoints = endPoints
        self.simplices = simplices

    def getBlendedImage(self, alpha):
        # Initialize blended image components.
        target1 = np.empty(self.startImage.shape, dtype = 'float64')
        target2 = np.empty(self.endImage.shape, dtype = 'float64')

        # Process all triangles.
        targets = (1 - alpha) * self.startPoints + alpha * self.endPoints
        for triangle in self.simplices.tolist():
            # Define relevant points.
            src = self.startPoints[triangle]
            dst = self.endPoints[triangle]
            tar = targets[triangle]

            # Define affine transforms.
            ColorAffine(src, tar).transform(self.startImage, target1)
            ColorAffine(dst, tar).transform(self.endImage, target2)

        # Alpha blend images.
        target = (1 - alpha) * target1 + alpha * target2

        return target.astype('uint8')

if __name__ == "__main__":
    compare = imageio.imread('test_data/Alpha50Gray.png')
    startPoint = np.load('test_data/startPoints.npy')
    endPoint = np.load('test_data/endPoints.npy')
    startImage = imageio.imread('test_data/StartSmallGray.png')
    endImage = imageio.imread('test_data/EndSmallGray.png')
    blend = Blender(startImage, startPoint, endImage, endPoint)
    A = blend.getBlendedImage(.5)

    print(count)
    temp = Image.fromarray(A, 'L')
    temp.show()

    startPoints = np.loadtxt('Tiger2Color.jpg.txt')
    endPoints = np.loadtxt('WolfColor.jpg.txt')
    startImage = imageio.imread('Tiger2Color.jpg')
    endImage = imageio.imread('WolfColor.jpg')
    B = ColorBlender(startImage, startPoints, endImage, endPoints)
    temp = Image.fromarray(B, 'L')
    temp.show()