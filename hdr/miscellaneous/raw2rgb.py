#!/usr/bin/env python
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.nan)
import scipy
from scipy import ndimage
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
# From : https://github.com/fschill/pyraw/blob/master/show_raw.py
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

        images_in = images_in + '/data5_.data'
        data = np.fromfile(images_in, dtype='uint16')  #uint16
        data = data.reshape([2464, 3296])

        # de-bayering
        # s = (1232, 1648, 3, dtype=float)
        # db_rgb = np.ones(s)
        #db_rgb = np.zeros([1232, 1648] + (3,), dtype=float)
        db_rgb = np.zeros((data.shape[0] - 1, data.shape[1] - 1, 3), dtype=float)


        # color channel weights
        rw = 0.83
        gw = 1.0
        bw = 1.15

        # color conversion matrix (from raspi_dng/dcraw)
        # R        g        b
        cvm = np.array(
            [[1.20, -0.30, 0.00],
             [-0.05, 0.80, 0.14],
             [0.20, 0.20, 0.7]])

        # cvm= np.array([8032,-3478,-274,-1222,5560,-240,100,-2714,6716], float).reshape(3,3)/10000.0
        #print(cvm)
        # reorder bayer values (RGGB) into intermediate full-color points (o)
        # green is weighted down a bit to give a more neutral color balance
        #
        # G   B   G   B
        #   o   o   o
        # R   G   R   G
        #   o   o   o
        # G   B   G   B

        db_rgb[0::2, 0::2, 0] = rw * (data[1::2, 0::2])  # Red
        db_rgb[0::2, 0::2, 1] = 0.5 * gw * data[0::2, 0::2] \
                                + 0.5 * gw * data[1::2, 1::2]  # Green
        db_rgb[0::2, 0::2, 2] = bw * data[0::2, 1::2]  # Blue

        db_rgb[0::2, 1::2, 0] = rw * data[1::2, 2::2]  # Red
        db_rgb[0::2, 1::2, 1] = 0.5 * gw * data[0::2, 2::2] \
                                + 0.5 * gw * data[1::2, 1::2][:, :-1]  # Green
        db_rgb[0::2, 1::2, 2] = bw * data[0::2, 1::2][:, :-1]  # Blue

        db_rgb[1::2, 0::2, 0] = rw * data[1::2, 0::2][:-1]  # Red
        db_rgb[1::2, 0::2, 1] = 0.5 * gw * data[2::2, 0::2] \
                                + 0.5 * gw * data[1::2, 1::2][:-1, :]  # Green
        db_rgb[1::2, 0::2, 2] = bw * data[2::2, 1::2][:, :]  # Blue

        db_rgb[1::2, 1::2, 0] = rw * data[1::2, 2::2][:-1, :]  # Red
        db_rgb[1::2, 1::2, 1] = 0.5 * gw * data[2::2, 2::2] \
                                + 0.5 * gw * data[1::2, 1::2][:-1, :-1]  # Green
        db_rgb[1::2, 1::2, 2] = bw * data[2::2, 1::2][:, :-1]  # Blue

        #db_rgb = db_rgb.dot(cvm)

        #print("Min/max values:", np.min(db_rgb), np.max(db_rgb))
        db_rgb = (db_rgb - np.min(db_rgb)) / (np.max(db_rgb) - np.min(db_rgb))

        # apply log to brighten dark tones (the added value reduces effect by flattening the curve)
        db_rgb = np.log(db_rgb + 0.1)

        # normalize image
        db_rgb = (db_rgb - np.min(db_rgb)) / (np.max(db_rgb) - np.min(db_rgb))

        red   = db_rgb[:, :, 0]
        green = db_rgb[:, :, 1]
        blue  = db_rgb[:, :, 2]

        # show color channels and RGB reconstruction

        show_all = False
        show_image = True

        if(show_image):
            if(show_all):
                fig = plt.figure()
                ax1 = fig.add_subplot(221)
                ax2 = fig.add_subplot(222)
                ax3 = fig.add_subplot(223)
                ax4 = fig.add_subplot(224)
                ax1.title.set_text('RED')
                ax2.title.set_text('GREEN')
                ax3.title.set_text('BLUE')
                ax4.title.set_text('RGB')
                ax1.imshow(-red, interpolation='nearest', cmap=cm.binary)
                ax2.imshow(-green, interpolation='nearest', cmap=cm.binary)
                ax3.imshow(-blue, interpolation='nearest', cmap=cm.binary)
                ax4.imshow(db_rgb, interpolation='nearest', cmap=cm.binary)
                plt.show()
            else:
                plt.imshow(db_rgb, interpolation='nearest', cmap=cm.binary)
                plt.title('RGB')
                plt.show()
        else:
            raw_img_name = 'data.rgb'
            with open(image_out + "/" + raw_img_name, 'wb') as g:
                db_rgb.tofile(g)

    except Exception as e:
        print('Error in Main: ' + str(e))


if __name__ == '__main__':
    main()
