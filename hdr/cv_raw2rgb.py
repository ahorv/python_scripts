#!/usr/bin/env python
from __future__ import division
import cv2
import numpy as np
np.set_printoptions(threshold=np.nan)
import matplotlib.pyplot as plt
import matplotlib.cm as cm

global images_in
images_in = r'./raw_data/20180325_151016'  # @home
image_out = r'./output'  # @home
#images_in = r'C:\Users\tahorvat\PycharmProjects\python_scripts\hdr\input\20171025_140139'  # @lab
#image_out = r'C:\Users\tahorvat\PycharmProjects\python_scripts\hdr\output'  #@ home  # @ lab

######################################################################
## Hoa: 09.11.2017 Version 1 : raw2rgb.py
######################################################################
# Using opencv to debayer a numpy array of an image.
# Given a numpy array of an image, the script debayers and interpolates
# and shows the result as (LDR) jpg image.
#
# New /Changes:
# ----------------------------------------------------------------------
#
# 09.11.2017 : first implemented
#
#
######################################################################

def main():
    try:
        global images_in
        global image_out

        images_in = images_in + '/data5_.data'

        imrows = 2464
        imcols = 3296

        imsize = imrows*imcols

        with open(images_in, "rb") as rawimage:
            img = np.fromfile(rawimage, np.dtype('u2'), imsize).reshape((imrows, imcols))


        db_rgb = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)
        db_rgb = (db_rgb - np.min(db_rgb)) / (np.max(db_rgb) - np.min(db_rgb)) # normalize image

        show_img = True

        if(show_img):
            plt.imshow(db_rgb, interpolation='nearest', cmap=cm.binary)
            plt.title('RGB')
            plt.show()
        else:
            raw_img_name = 'cv_data.rgb'
            with open(image_out + "/" + raw_img_name, 'wb') as g:
                db_rgb.tofile(g)

    except Exception as e:
        print('Error in Main: ' + str(e))

if __name__ == '__main__':
    main()
