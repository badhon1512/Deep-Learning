import numpy as np
import matplotlib.pyplot as plt


class Checker:
    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None

    def draw(self):

        black = np.zeros((self.tile_size, self.tile_size))
        white = np.ones((self.tile_size, self.tile_size))
        # tile is a 2X2 matrix
        #  [b,w]
        #  [w, b
        tile = np.concatenate((np.concatenate((black, white)), np.concatenate((white, black))), axis=1)
        self.output = np.tile(tile, (self.resolution // (2 * self.tile_size), self.resolution // (2 * self.tile_size)))
        return self.output.copy()

    def show(self):
        self.draw()
        plt.imshow(self.output, cmap = 'gray')
        plt.show()


class Circle:

    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = np.zeros((self.resolution, self.resolution))

    def draw(self):

        x = y = np.arange(self.resolution)

        # xx, yy are 2d meshgrid of x and y
        xx, yy = np.meshgrid(x, y)


        # calculate the distance from the center
        dist_from_center = (xx - self.position[0]) ** 2 + (yy - self.position[1]) ** 2

        # if the distances belongs insides the radius**2, these are the circle
        self.output = np.where(dist_from_center <= self.radius ** 2, 1, 0)

        return self.output.copy()

    def show(self):
        self.draw()
        plt.imshow(self.output, cmap='gray')
        plt.show()


class Spectrum:

    def __init__(self, resolution):
        self.resolution = resolution
        self.output = np.zeros([self.resolution, self.resolution, 3])

    def draw(self):

        # green must be broadcast vertically as the value is increasing vertically
        # therefore it must be column vector
        r = np.linspace(0, 1, self.resolution).reshape(1, self.resolution)
        g = np.linspace(0, 1, self.resolution).reshape(self.resolution, 1)
        b = np.linspace(1, 0, self.resolution).reshape(1, self.resolution)
        self.output[:, :, 0] = r
        self.output[:, :, 1] = g
        self.output[:, :, 2] = b

        return self.output.copy()

    def show(self):
        self.draw()
        plt.imshow(self.output)
        plt.show()
