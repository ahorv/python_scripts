#!/usr/bin/env python
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from glob import glob

global hdr_path
#hdr_path = r'C:\Hoa_Python_Projects\python_scripts\hist\input\20171025_153519',0  #GESCHAEFTS-PC
hdr_path = r'C:\Users\tahorvat\PycharmProjects\python_scripts\hist\input\20171025_153519'



def load_images():
    try:
        fetched_imgs = []
        for file in sorted(glob(hdr_path + '/*.jpg')):
            fetched_imgs.append(file)
            print(file)

        image_stack = np.empty(len(fetched_imgs), dtype=object)

        for n in range(0, len(fetched_imgs)):
            image_stack[n] = cv2.imread(join(hdr_path, fetched_imgs[n]), cv2.IMREAD_COLOR)

        return image_stack


    except Exception as e:
        print('Could not load image: ' + str(e))


def load_img_as_nparr():
    try:
        onlyfiles = [f for f in listdir(hdr_path) if isfile(join(hdr_path, f))]
        image_stack = np.empty(len(onlyfiles))
        for n in range(0, len(onlyfiles)):
            img = cv2.imread(join(hdr_path, onlyfiles[n]), cv2.IMREAD_COLOR)
            image_stack[n] = np.zeros(img)
        print('All images loaded as numpy arraies!')

        return image_stack

    except Exception as e:
        print('Could not load image: ' + str(e))


def oldcv_hist(image, titel):
    try:
        color = ('b', 'g', 'r')
        for channel, col in enumerate(color):
            histr = cv2.calcHist(image, [channel], None, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])
        plt.title(titel)
        plt.show()
    except Exception as e:
        print('Could not generate histogram: '+ str(e))


def cv_hist_all_imgs(image_stack, cols=1, titles=None):
    try:
        color = ('b', 'g', 'r')

        assert ((titles is None) or (len(image_stack) == len(titles)))
        n_images = len(image_stack)
        if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
        fig = plt.figure()
        for n, (image, title) in enumerate(zip(image_stack, titles)):
            a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
            a.set_title(title)

            for channel, col in enumerate(color):
                histr = cv2.calcHist(image, [channel], None, [256], [0, 256])
                plt.plot(histr, color=col)
                plt.xlim([0, 256])

        fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
        plt.show()

    except Exception as e:
        print('Could not generate histogram: '+ str(e))



def plot_all_imgs(image_stack, cols=1, titles=None):
    try:
            """Display a list of images in a single figure with matplotlib.

            Parameters
            ---------
            images: List of np.arrays compatible with plt.imshow.

            cols (Default = 1): Number of columns in figure (number of rows is 
                                set to np.ceil(n_images/float(cols))).

            titles: List of titles corresponding to each image. Must have
                    the same length as titles.
            """
            assert ((titles is None) or (len(image_stack) == len(titles)))
            n_images = len(image_stack)
            if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
            fig = plt.figure()
            for n, (image, title) in enumerate(zip(image_stack, titles)):
                a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
                if image.ndim == 2:
                    plt.gray()
                plt.imshow(image)
                a.set_title(title)
            fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
            plt.show()

    except Exception as e:
        print('Could not generate histogram: '+ str(e))