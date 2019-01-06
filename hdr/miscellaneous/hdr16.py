# coding: utf-8

import os
import cv2
import math
import numpy as np
import shutil
import matplotlib.pyplot as plt
from os.path import join

global img_dir
global output_hdr_filename


img_dir = r'C:\Users\ati\Desktop\HDR-imaging-master\20181009_133912'
output_hdr_filename = join(img_dir,'output_hdr')

###############################################################################
## Hoa: 18.10.2018 Version 1 : hdr16.py
###############################################################################
# Adapted from : https://github.com/SSARCandy/HDR-imaging/blob/master/HDR-playground.py
# See also: https://github.com/vivianhylee/high-dynamic-range-image/blob/master/hdr.py
#
# Creates from a stack of jpg images one HDR image. *txt file with shuter spseeds
# must be provided in the source directory. Format of image_list.txt:
#
# Filename  exposure  1/shutter_speed f/stop gain(db) ND_filters
# data0.data   32      626.174  8 0 0
# data-2.data  16      2583.98  8 0 0
# data-4.data   8      11764.7  8 0 0
#
# New /Changes:
# -----------------------------------------------------------------------------
#
# 03.10.2018 : first implemented
#
###############################################################################


if not os.path.exists(output_hdr_filename):
    os.makedirs(output_hdr_filename)

def demosaic1(mosaic, awb_gains = None):
    try:
        black = mosaic.min()
        saturation = mosaic.max()

        uint14_max = 2 ** 14 - 1
        mosaic -= black  # black subtraction
        mosaic *= int(uint14_max / (saturation - black))
        mosaic = np.clip(mosaic, 0, uint14_max)  # clip to range

        if awb_gains is None:
            vb_gain = 1.0
            vg_gain = 1.0
            vr_gain = 1.0
        else:
            vb_gain = awb_gains[1]
            vg_gain = 1.0
            vr_gain = awb_gains[0]

        mosaic = mosaic.reshape([2464, 3296])
        mosaic = mosaic.astype('float')
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

        # down sample to RGB 8 bit image use: self.deraw2rgb1(image)

        return image

    except Exception as e:
        print('Error in demosaic1: {}'.format(e))

def toRGB_1(data):
    '''
    Belongs to deraw1
    :param data:
    :return:
    '''
    image = data // 256  # reduce dynamic range to 8 bpp
    image = np.clip(image, 0, 255).astype(np.uint8)

    return image

def read_data(path_to_image):
    data = np.fromfile(path_to_image, dtype='uint16')
    data = data.reshape([2464, 3296])
    raw = demosaic1(data)

    return raw.astype('uint16')

def load_exposures_data(source_dir, channel=0):
    '''
    Reads raw (16 bit) data files
    :param source_dir:
    :param channel:
    :return:
    '''
    is_data_type = False
    filenames = []
    exposure_times = []
    f = open(os.path.join(source_dir, 'image_list.txt'))
    for line in f:
        if (line[0] == '#'):
            continue
        (filename, exposure, *rest) = line.split()
        filenames += [filename]
        exposure_times += [exposure]

        img_list = [toRGB_1(read_data(os.path.join(source_dir, f))) for f in filenames]
        img_list = [img[:, :, channel] for img in img_list]
        exposure_times = np.array(exposure_times, dtype=np.float32)

def load_exposures(source_dir, channel=0):
    '''
    Reads either of jpg or raw depending on file extension
    in the image_list.txt - file.
    :param source_dir:
    :param channel:
    :return:
    '''
    is_data_type = False
    filenames = []
    exposure_times = []
    f = open(os.path.join(source_dir, 'image_list.txt'))
    for line in f:
        if (line[0] == '#'):
            continue
        (filename, exposure, *rest) = line.split()
        if 'data' in filename: is_data_type = True
        filenames += [filename]
        exposure_times += [exposure]

    if is_data_type:
        img_list = [toRGB_1(read_data(os.path.join(source_dir, f))) for f in filenames]
        img_list = [img[:, :, channel] for img in img_list]
        exposure_times = np.array(exposure_times, dtype=np.float32)
    else:
        img_list = [cv2.imread(os.path.join(source_dir, f), 1) for f in filenames]
        img_list = [img[:, :, channel] for img in img_list]
        exposure_times = np.array(exposure_times, dtype=np.float32)

    return (img_list, exposure_times)

def load_exposures_jpg(source_dir, channel=0):
    '''
    Reads jpg images
    :param source_dir:
    :param channel:
    :return:
    '''
    filenames = []
    exposure_times = []
    f = open(os.path.join(source_dir, 'image_list.txt'))
    for line in f:
        if (line[0] == '#'):
            continue
        (filename, exposure, *rest) = line.split()
        filenames += [filename]
        exposure_times += [exposure]

    img_list = [cv2.imread(os.path.join(source_dir, f), 1) for f in filenames]
    img_list = [img[:, :, channel] for img in img_list]
    exposure_times = np.array(exposure_times, dtype=np.float32)

    return (img_list, exposure_times)

# MTB implementation
def median_threshold_bitmap_alignment(img_list):
    median = [np.median(img) for img in img_list]
    binary_thres_img = [cv2.threshold(img_list[i], median[i], 255, cv2.THRESH_BINARY)[1] for i in range(len(img_list))]
    mask_img = [cv2.inRange(img_list[i], median[i] - 20, median[i] + 20) for i in range(len(img_list))]

    plt.imshow(mask_img[0], cmap='gray')
    plt.show()

    max_offset = np.max(img_list[0].shape)
    levels = 5

    global_offset = []
    for i in range(0, len(img_list)):
        offset = [[0, 0]]
        for level in range(levels, -1, -1):
            scaled_img = cv2.resize(binary_thres_img[i], (0, 0), fx=1 / (2 ** level), fy=1 / (2 ** level))
            ground_img = cv2.resize(binary_thres_img[0], (0, 0), fx=1 / (2 ** level), fy=1 / (2 ** level))
            ground_mask = cv2.resize(mask_img[0], (0, 0), fx=1 / (2 ** level), fy=1 / (2 ** level))
            mask = cv2.resize(mask_img[i], (0, 0), fx=1 / (2 ** level), fy=1 / (2 ** level))

            level_offset = [0, 0]
            diff = float('Inf')
            for y in [-1, 0, 1]:
                for x in [-1, 0, 1]:
                    off = [offset[-1][0] * 2 + y, offset[-1][1] * 2 + x]
                    error = 0
                    for row in range(ground_img.shape[0]):
                        for col in range(ground_img.shape[1]):
                            if off[1] + col < 0 or off[0] + row < 0 or off[1] + col >= ground_img.shape[1] or off[
                                0] + row >= ground_img.shape[1]:
                                continue
                            if ground_mask[row][col] == 255:
                                continue
                            error += 1 if ground_img[row][col] != scaled_img[y + off[0]][x + off[1]] else 0
                    if error < diff:
                        level_offset = off
                        diff = error
            offset += [level_offset]
        global_offset += [offset[-1]]
    return global_offset

def hdr_debvec(img_list, exposure_times):
    B = [math.log(e, 2) for e in exposure_times]
    l = 50 # lambda sets amount of smoothness
    w = [z if z <= 0.5 * 255 else 255 - z for z in range(256)]

    small_img = [cv2.resize(img, (10, 10)) for img in img_list]
    Z = [img.flatten() for img in small_img]

    return response_curve_solver(Z, B, l, w)

# Implementation of paper's Equation(3) with weight
def response_curve_solver(Z, B, l, w):
    n = 256
    A = np.zeros(shape=(np.size(Z, 0) * np.size(Z, 1) + n + 1, n + np.size(Z, 1)), dtype=np.float32)
    b = np.zeros(shape=(np.size(A, 0), 1), dtype=np.float32)

    # Include the data−fitting equations
    k = 0
    for i in range(np.size(Z, 1)):
        for j in range(np.size(Z, 0)):
            z = Z[j][i]
            wij = w[z]
            A[k][z] = wij
            A[k][n + i] = -wij
            b[k] = wij * B[j]
            k += 1

    # Fix the curve by setting its middle value to 0
    A[k][128] = 1
    k += 1

    # Include the smoothness equations
    for i in range(n - 1):
        A[k][i] = l * w[i + 1]
        A[k][i + 1] = -2 * l * w[i + 1]
        A[k][i + 2] = l * w[i + 1]
        k += 1

    # Solve the system using SVD
    x = np.linalg.lstsq(A, b,rcond=None)[0]
    g = x[:256]
    lE = x[256:]

    return g, lE

# Implementation of paper's Equation(6)
def construct_radiance_map(g, Z, ln_t, w):
    acc_E = [0] * len(Z[0])
    ln_E = [0] * len(Z[0])

    pixels, imgs = len(Z[0]), len(Z)
    for i in range(pixels):
        acc_w = 0
        for j in range(imgs):
            z = Z[j][i]
            acc_E[i] += w[z] * (g[z] - ln_t[j])
            acc_w += w[z]
        ln_E[i] = acc_E[i] / acc_w if acc_w > 0 else acc_E[i]
        acc_w = 0

    return ln_E

def construct_hdr(img_list, response_curve, exposure_times):
    # Construct radiance map for each channels
    img_size = img_list[0][0].shape
    w = [z if z <= 0.5 * 255 else 255 - z for z in range(256)]
    ln_t = np.log2(exposure_times)

    vfunc = np.vectorize(lambda x: math.exp(x))
    hdr = np.zeros((img_size[0], img_size[1], 3), 'float32')

    # construct radiance map for BGR channels
    for i in range(3):
        print(' - Constructing radiance map for {0} channel .... '.format('BGR'[i]), end='', flush=True)
        Z = [img.flatten().tolist() for img in img_list[i]]
        E = construct_radiance_map(response_curve[i], Z, ln_t, w)
        # Exponational each channels and reshape to 2D-matrix
        hdr[..., i] = np.reshape(vfunc(E), img_size)
        print('done')

    return hdr

# Save HDR image as .hdr file format
# Code based on https://gist.github.com/edouardp/3089602
def save_hdr(hdr, filename):
    image = np.zeros((hdr.shape[0], hdr.shape[1], 3), 'float32')
    image[..., 0] = hdr[..., 2]
    image[..., 1] = hdr[..., 1]
    image[..., 2] = hdr[..., 0]

    print('Path to save HDR: {}'.format(filename))

    f = open(filename, 'wb')
    f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
    header = '-Y {0} +X {1}\n'.format(image.shape[0], image.shape[1])
    f.write(bytes(header, encoding='utf-8'))

    brightest = np.maximum(np.maximum(image[..., 0], image[..., 1]), image[..., 2])
    mantissa = np.zeros_like(brightest)
    exponent = np.zeros_like(brightest)
    np.frexp(brightest, mantissa, exponent)
    scaled_mantissa = mantissa * 256.0 / brightest
    rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    rgbe[..., 0:3] = np.around(image[..., 0:3] * scaled_mantissa[..., None])
    rgbe[..., 3] = np.around(exponent + 128)

    rgbe.flatten().tofile(f)
    f.close()

if __name__ == '__main__':

    # Loading exposure images into a list
    print('Reading input images.... ', end='')
    img_list_b, exposure_times = load_exposures(img_dir, 0)
    img_list_g, exposure_times = load_exposures(img_dir, 1)
    img_list_r, exposure_times = load_exposures(img_dir, 2)
    print('done')

    # Solving response curves
    print('Solving response curves .... ', end='')
    gb, _ = hdr_debvec(img_list_b, exposure_times)
    gg, _ = hdr_debvec(img_list_g, exposure_times)
    gr, _ = hdr_debvec(img_list_r, exposure_times)
    print('done')

    # Show response curve
    print('Saving response curves plot .... ', end='')
    plt.figure(figsize=(10, 10))
    plt.plot(gr, range(256), 'rx')
    plt.plot(gg, range(256), 'gx')
    plt.plot(gb, range(256), 'bx')
    plt.ylabel('pixel value Z')
    plt.xlabel('log exposure X')
    plt.savefig(join(output_hdr_filename,'response-curve.png'))
    print('done')

    print('Constructing HDR image: ')
    hdr = construct_hdr([img_list_b, img_list_g, img_list_r], [gb, gg, gr], exposure_times)
    print('done')

    # Display Radiance map with pseudo-color image (log value)
    print('Saving pseudo-color radiance map .... ', end='')
    plt.figure(figsize=(12, 8))
    plt.imshow(np.log2(cv2.cvtColor(hdr, cv2.COLOR_BGR2GRAY)), cmap='jet')
    plt.colorbar()
    plt.savefig(join(output_hdr_filename,'radiance-map.png'))
    print('done')

    print('Saving HDR image .... ', end='')
    save_hdr(hdr, join(output_hdr_filename,'my_HDR.hdr'))
    print('done')




