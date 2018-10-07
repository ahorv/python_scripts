#!/usr/bin/env python

from __future__ import print_function

import os
import cv2
import sys
import time
import shutil
import exifread
from shutil import copy2
from glob import glob
import subprocess
import zipfile
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join
import numpy as np
from fractions import Fraction

print('Version opencv: ' + cv2.__version__)

######################################################################
## Hoa: 07.10.2018 Version 1 : postprocess.py
######################################################################
# Reads pictures and plots their histograms
#
######################################################################
global Path_to_raw

Path_to_raw = r'C:\Users\ati\Desktop\camera_1\wellExp\0226.jpg'


def main():
    try:
        global Path_to_raw

        s = r'I:\SkY_CAM_IMGS\picam\camera_3\20181007\20181007_103010'

        my_split = s.split('\\')

        print(my_split)
        found = my_split[-1]



        print('Found: {}'.format(found))

        return
        image = cv2.imread(Path_to_raw)
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        plt.axis("off")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()




        print('Postprocess.py done')

    except Exception as e:
        print('MAIN: Error in main: ' + str(e))


if __name__ == '__main__':
    main()