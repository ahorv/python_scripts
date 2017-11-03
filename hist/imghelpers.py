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
        for file in sorted(glob(hdr_path + '/*.jpg')):
            print(file)

        return


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


def cv_hist(image, titel):
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