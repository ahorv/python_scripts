#!/usr/bin/env python
from __future__ import division
import cv2
import numpy as np
from os.path import isfile, join
np.set_printoptions(threshold=np.nan)
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

global Path_to_raw
Path_to_raw = r'C:\Users\tahorvat\Desktop\20180409_090414'

def data2rgb(path_to_raw):

    try:

        imrows = 2464
        imcols = 3296

        imsize = imrows*imcols

        with open(path_to_raw, "rb") as rawimage:
            img = np.fromfile(rawimage, np.dtype('u2'), imsize).reshape((imrows, imcols))


        db_rgb = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)
        db_rgb = (db_rgb - np.min(db_rgb)) / (np.max(db_rgb) - np.min(db_rgb)) # normalize image

        return db_rgb

    except Exception as e:
        print('Error in data2rgb: ' + str(e))


def main():
    try:
        global Path_to_raw
        raw_img_path = Path_to_raw + '/data5_.data'

        show_img = False
        save_img = True

        db_rgb = data2rgb(raw_img_path)

        if(show_img):
            plt.imshow(db_rgb, interpolation='nearest', cmap=cm.binary)
            plt.title('RGB')
            plt.show()
        if(save_img):
            raw_img_name = 'cv_data.rgb'
            print('rgb file path: '+ str(Path_to_raw + '\\' + raw_img_name))
            with open(join(Path_to_raw,raw_img_name), 'wb') as g:
                db_rgb.tofile(g)

        print('Done')

    except Exception as e:
        print('Error in Main: ' + str(e))

if __name__ == '__main__':
    main()
