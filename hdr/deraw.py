#!/usr/bin/env python

import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from os.path import isfile, join

print('Version opencv: ' + cv2.__version__)

######################################################################
## Hoa: 14.10.2018 Version 1 : deraw.py
######################################################################
# Generates from image raw data a down sampled viewable image.
# deraw.py is according to paper 'Processing RAW images in Python' from
# Pavel Rojtberg 06.03.2017
#
# data2rgb simplified version
#
# New /Changes:
# ----------------------------------------------------------------------
#
# 14.10.2018 : first implemented
#
#
######################################################################

global Path_to_Dir
global image_name
Path_to_Dir = r'C:\Users\ati\Desktop\test'
image_path = join(Path_to_Dir,'data-2.data')

def deraw(path):
    try:
        mosaic = np.fromfile(path, dtype='uint16')

        black = mosaic.min()  # proc.imgdata.color.black
        saturation = mosaic.max()

        uint14_max = 2 ** 14 - 1
        mosaic -= black  # black subtraction
        mosaic *= int(uint14_max / (saturation - black))
        mosaic = np.clip(mosaic, 0, uint14_max)  # clip to range

        vb_gain = 37/32
        vg_gain = 1.0  # raspi raw has already gain = 1 of green channel
        vr_gain = 63/32

        mosaic = mosaic.reshape([2464, 3296])
        mosaic = mosaic.astype('float')
        print('dtype: {}'.format(mosaic.dtype))
        mosaic[0::2, 1::2] *= vb_gain  # Blue
        mosaic[1::2, 0::2] *= vr_gain  # Red
        mosaic = np.clip(mosaic, 0, uint14_max)  # clip to range
        mosaic *= 2 ** 2

        # demosaic
        p1 = mosaic[0::2, 1::2]  # Blue
        p2 = mosaic[0::2, 0::2]  # Green
        p3 = mosaic[1::2, 1::2]  # Green
        p4 = mosaic[1::2, 0::2]  # Red

        blue = p1
        green = np.clip((p2 // 2 + p3 // 2), 0, 2 ** 16 - 1)
        red = p4

        image = np.dstack([red, green, blue])  # 16 - bit 'image'

        # down sample to RGB 8 bit image
        image = image // 2 ** 8  # reduce dynamic range to 8bpp
        image = np.clip(image, 0, 255).astype(np.uint8)

        cv2.imwrite(join(Path_to_Dir, 'deraw_2.jpg'), image)

        print('Done')

    except Exception as e:
        print('Error in deraw: {}'.format(e))

def data2rgb (path):
    try:
        # aus raw2img.py
        data = np.fromfile(path, dtype='uint16')
        data = data.reshape([2464, 3296])

        p1 = data[0::2, 1::2]  # Blue
        p2 = data[0::2, 0::2]  # Green
        p3 = data[1::2, 1::2]  # Green
        p4 = data[1::2, 0::2]  # Red

        blue = p1
        green = ((p2 + p3)) / 2
        red = p4

        gamma = 1.0  # gamma correction           # neu : 1.55
        # b, g and r gain;  wurden rausgelesen aus den picam Aufnahmedaten
        vb = 37/32 # 87 / 64.  = 1.359375           # neu : 0.56
        vg = 1.0       # 1.                             # neu : 1
        vr = 63/32  # 235 / 128.  = 1.8359375        # neu : 0.95

        # color conversion matrix (from raspi_dng/dcraw)
        # R        g        b
        cvm = np.array(
            [[1.20, -0.30, 0.00],
             [-0.05, 0.80, 0.14],
             [0.20, 0.20, 0.7]])

        s = (1232, 1648, 3)
        rgb = np.zeros(s)

        rgb[:, :, 0] = vr * 1023 * (red   / 1023.) ** gamma
        rgb[:, :, 1] = vg * 1023 * (green / 1023.) ** gamma
        rgb[:, :, 2] = vb * 1023 * (blue  / 1023.) ** gamma

        #rgb = rgb.dot(cvm)

        rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb))

        height, width = rgb.shape[:2]

        img = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_CUBIC)
        #normalizeImage
        out = np.zeros(img.shape, dtype=np.float)
        min = img.min()
        out = img - min

        # get the max from out after normalizing to 0
        max = out.max()
        out *= (255 / max)

        out = np.uint8(out)
        cv2.imwrite(join(Path_to_Dir, 'data2rgb_2.jpg'), out)

    except Exception as e:
        print('Error in data2rgb: ' + str(e))



def main():
    try:
        global image_path

        deraw(image_path)
        data2rgb(image_path)


    except Exception as e:
        print('MAIN: Error in main: ' + str(e))


if __name__ == '__main__':
    main()