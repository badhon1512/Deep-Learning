import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import math
# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:

    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.c_epoch = 0
        self.c_l_image_i = -1
        self.labels = json.load(open(self.label_path, 'r'))

        self.images = []
        for file in os.listdir(self.file_path):
            img = np.load(os.path.join(self.file_path, file))
            img = resize(img, (self.image_size[0],self.image_size[1],self.image_size[2]))
            if self.mirroring or self.rotation:
                img = self.augment(img)
            self.images.append((img, self.labels[file[:-4]]))

        self.images = np.array(self.images, dtype=object)
        # self.images = np.array(list(map(np.load, glob.glob(f"{self.file_path}/*.npy"))))
        # self.images = self.images.reshape(self.images.shape[0], *self.image_size)

    def next(self):

        # current image index always smaller than the batch size if new epoc started
        # in previous iteration according to our logic
        if self.c_l_image_i <=  self.batch_size:
            if self.shuffle:
                np.random.shuffle(self.images)

        images = []
        labels = []
        for i in range(self.batch_size):
            self.c_l_image_i += 1

            if self.c_l_image_i >= self.images.shape[0]:
                self.c_epoch += 1
                self.c_l_image_i = 0

            img, l = self.images[self.c_l_image_i]
            images.append(img)
            labels.append(l)

        return np.array(images), np.array(labels)

    def augment(self,img):
        if self.mirroring:
            img = np.flip(img, axis=1)
        if self.rotation:
            img = np.rot90(img, k=np.random.randint(1, 4))
        return img

    def current_epoch(self):
        return self.c_epoch

    def class_name(self, x):
        # This function returns the class name for a specific input
        return self.class_dict.get(x)
    def show(self):
        images, labels = self.next()
        labels = list(labels)
        fig = plt.figure()
        for i in range(0, self.batch_size):
            fig.add_subplot(math.ceil(self.batch_size / 4) , 4, i+1)
            plt.imshow(images[i])
            plt.axis("off")
            plt.title(self.class_name(labels[i]))
        plt.show()

