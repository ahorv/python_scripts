#!/usr/bin/env python

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import exifread
import math
from os import listdir
from os.path import isfile, join
from fractions import Fraction
import shutil

print('Version opencv: ' + cv2.__version__)

######################################################################
## Hoa: 17.10.2018 Version 3 : hdr.py
######################################################################
# Generates with aid of opencv a HDR image either from 3 jpg - images
# or from 3 *.data files.
#
# New /Changes:
# ----------------------------------------------------------------------
#
# 10.11.2017 : Added new logging
# 18.09.2018 : reads *.data - files
# 17.10.2018 : added auto color (white) balance function
#
#
######################################################################

global input_path
global input_path_jpg
global input_path_raw
global SELECTION
global LogFileName

input_path  = r'C:\Users\ati\Desktop\20181009_130227'
output_path_jpg = join(input_path,'_output_jpg')
output_path_raw = join(input_path,'_output_raw')
SELECTION = [0, -2, -4] # ['0','5','9']
#SELECTION = ['0','5','9']
LogFileName = 'camstats.log'

def getShutterTimes(file_path, file_name=LogFileName):
    try:
        '''
        returns shutter_time in microseconds as np.float32 type
        '''
        listOfSS = np.empty(3, dtype=np.float32)

        f = open(join(file_path, file_name), 'r')
        logfile = f.readlines()
        logfile.pop(0)  # remove non relevant lines
        logfile.pop(0)  # remove non relevant lines

        pos = 0
        for line in logfile:
            value = line.split("ss:", 1)[1]
            value = value.split(',', 1)[0]
            value = value.strip()
            value += '/1000000'
            val_float = np.float32(Fraction(str(value)))
            listOfSS[pos] = val_float
            pos += 1

        return listOfSS

    except Exception as e:
        print('Error in getShutterTimes: ' + str(e))

def getAWB_Gains(file_path, file_name=LogFileName):
    try:
        '''
        returns shutter_time in microseconds as np.float32 type
        '''
        awb_gains = np.empty([3, 2], dtype=np.float32)

        f = open(join(file_path, file_name), 'r')
        logfile = f.readlines()
        logfile.pop(0)  # remove non relevant lines
        logfile.pop(0)  # remove non relevant lines

        pos = 0
        for line in logfile:
            value = line.split("awb:[", 1)[1]
            value = value.split('],', 1)[0].replace('Fraction', '').replace('(', '', 1).replace('))', ')').replace(
                " ", "")
            red_gain = value.split('),', 1)[0].strip('(').replace(',', '/')
            blue_gain = value.split(',(', 1)[1].strip(')').replace(',', '/')
            red_gain = np.float32(Fraction(str(red_gain)))
            blue_gain = np.float32(Fraction(str(blue_gain)))
            awb_gains[pos] = [red_gain, blue_gain]
            pos += 1

        return awb_gains

    except Exception as e:
        print('Error in getAWB_Gains: ' + str(e))

def getEXIF_TAG(file_path, field):
    try:
        foundvalue = '0'
        with open(file_path, 'rb') as f:
            exif = exifread.process_file(f)

        for k in sorted(exif.keys()):
            if k not in ['JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote']:
                if k == field:
                    #print('%s = %s' % (k, exif[k]))
                    foundvalue = np.float32(Fraction(str(exif[k])))
                    break

        return foundvalue

    except Exception as e:
        print('EXIF: Could not read exif data ' + str(e))

def readImagesAndExpos(mypath,piclist=[0,5,9]):
    try:
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        image_stack = np.empty(len(piclist), dtype=object)    # Achtung len = onlyfiles für alle bilder
        expos_stack = np.empty(len(piclist), dtype=np.float32)# Achtung len = onlyfiles für alle bilder
        for n in range(0, len(onlyfiles)):
            picnumber = ''.join(filter(str.isdigit, onlyfiles[n]))
            pos = 0
            for pic in piclist:
                if picnumber == pic:
                    expos_stack[pos] = getEXIF_TAG(join(mypath, onlyfiles[n]), "EXIF ExposureTime")

                    if onlyfiles[n].endswith('.jpg'):
                        img = cv2.imread(join(mypath, onlyfiles[n]), cv2.IMREAD_COLOR)
                        masked_img = maske_image(img)
                        image_stack[pos] = masked_img
                pos = pos + 1

        return image_stack, expos_stack

    except Exception as e:
        print('readImagesAndExpos: Could not read images ' + str(e))

def readDataAndExpos(mypath, piclist=[0,5,9]):
    try:
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        image_stack = np.empty(len(piclist), dtype=object)    # Achtung len = onlyfiles für alle bilder
        expos_stack = np.empty(len(piclist), dtype=np.float32)# Achtung len = onlyfiles für alle bilder
        for n in range(0, len(onlyfiles)):
            picnumber = ''.join(filter(str.isdigit, onlyfiles[n]))
            pos = 0
            for pic in piclist:
                if picnumber == pic:
                    expos_stack[pos] = getEXIF_TAG(join(mypath, onlyfiles[n]), "EXIF ExposureTime")

                    if onlyfiles[n].endswith('.data'):
                        image_stack[pos] = data2rgb(join(mypath, onlyfiles[n]))
                pos = pos + 1

        return image_stack, expos_stack

    except Exception as e:
        print('readDataAndExpos: Could not read data files ' + str(e))

def new_readRawImages(mypath, piclist = SELECTION):
    try:
        onlyfiles_data = [f for f in listdir(mypath) if isfile(join(mypath, f)) & f.endswith('.data')]
        image_stack = np.empty(len(piclist), dtype=object)
        expos_stack = getShutterTimes(mypath)
        awb_stack = getAWB_Gains(mypath)

        # Importing and debayering raw images
        for n in range(0, len(onlyfiles_data)):
            picnumber = [str(s.replace('.', '')) for s in re.findall(r'-?\d+\.?\d*', onlyfiles_data[n])]
            pos = 0
            for pic in piclist:
                if str(picnumber[0]) == str(pic):
                    image_stack[pos] = new_data2rgb(join(mypath, onlyfiles_data[n]),awb_stack[pos])
                pos +=1

        return image_stack, expos_stack

    except Exception as e:
        print('readRawImages: Could not read *.data files ' + str(e))

def new_readImagesAndExpos(mypath, piclist = SELECTION):
    try:
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) & f.endswith('.jpg')]
        image_stack = np.empty(len(piclist), dtype=object)       # Achtung len = onlyfiles für alle bilder
        expos_stack = getShutterTimes(mypath)

        for n in range(0, len(onlyfiles)):
            picnumber = [str(s.replace('.', '')) for s in re.findall(r'-?\d+\.?\d*', onlyfiles[n])]
            pos = 0
            for pic in piclist:
                if str(picnumber[0]) == str(pic):
                    image_stack[pos] = cv2.imread(join(mypath, onlyfiles[n]), cv2.IMREAD_COLOR)
                    #print('Pic {}, reading data from : {}, exif: {}'.format(str(picnumber), onlyfiles[n], expos_stack[n]))
                pos +=1

        return image_stack, expos_stack

    except Exception as e:
        print('Error in readImagesAndExpos: ' + str(e))

def cmask(index, radius, array):
    """Generates the mask for a given input image.
    The generated mask is needed to remove occlusions during post-processing steps.

    Args:
        index (numpy array): Array containing the x- and y- co-ordinate of the center of the circular mask.
        radius (float): Radius of the circular mask.
        array (numpy array): Input sky/cloud image for which the mask is generated.

    Returns:
        numpy array: Generated mask image."""

    a, b = index
    is_rgb = len(array.shape)

    if is_rgb == 3:
        ash = array.shape
        nx = ash[0]
        ny = ash[1]
    else:
        nx, ny = array.shape

    s = (nx, ny)
    image_mask = np.zeros(s)
    y, x = np.ogrid[-a:nx - a, -b:ny - b]
    mask = x * x + y * y <= radius * radius
    image_mask[mask] = 1

    return (image_mask)

def maske_image(input_image, size = [1944, 2592, 3],centre = [972, 1296], radius = 1350):  # 880,1190, r = 1450

    empty_img = np.zeros(size, dtype=np.uint8)
    mask = cmask(centre, radius, empty_img)

    red   = input_image[:, :, 0]
    green = input_image[:, :, 1]
    blue  = input_image[:, :, 2]

    r_img = red.astype(float)   * mask
    g_img = green.astype(float) * mask
    b_img = blue.astype(float)  * mask

    dimension = (input_image.shape[0], input_image.shape[1], 3)
    output_img = np.zeros(dimension, dtype=np.uint8)

    output_img[..., 0] = r_img[:, :]
    output_img[..., 1] = g_img[:, :]
    output_img[..., 2] = b_img[:, :]

    return output_img

def data2rgb (path_to_img):
    '''
    Old version
    :param path_to_img:
    :return:
    '''
    try:
        # aus raw2img.py
        data = np.fromfile(path_to_img, dtype='uint16')
        data = data.reshape([2464, 3296])

        p1 = data[0::2, 1::2]  # Blue
        p2 = data[0::2, 0::2]  # Green
        p3 = data[1::2, 1::2]  # Green
        p4 = data[1::2, 0::2]  # Red

        blue = p1
        green = ((p2 + p3)) / 2
        red = p4

        gamma = 1.6  # gamma correction           # neu : 1.55
        # b, g and r gain;  wurden rausgelesen aus den picam Aufnahmedaten
        vb = 1.3   # 87 / 64.  = 1.359375           # neu : 0.56
        vg = 1.80       # 1.                             # neu : 1
        vr = 1.8  # 235 / 128.  = 1.8359375        # neu : 0.95

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
        #img = img.astype(np.float32)

        #normalizeImage
        out = np.zeros(img.shape, dtype=np.float)
        min = img.min()
        out = img - min

        # get the max from out after normalizing to 0
        max = out.max()
        out *= (255 / max)

        out = np.uint8(out)

        out  = maske_image(np.uint8(out), [1232, 1648, 3], (616,824), 1000)

        return out


    except Exception as e:
        print('data2rgb: Could not convert data to rgb: ' + str(e))

def new_data2rgb(path_to_img, awb_gains):

    mosaic = np.fromfile(path_to_img, dtype='uint16')
    mosaic = demosaic1(mosaic, awb_gains)
    img = toRGB_1(mosaic)
    return img

def demosaic1(mosaic, awb_gains = None):
    try:
        black = mosaic.min()
        saturation = mosaic.max()

        uint14_max = 2 ** 14 - 1
        mosaic -= black  # black subtraction
        mosaic *= int(uint14_max / (saturation - black))
        mosaic = np.clip(mosaic, 0, uint14_max)  # clip to range

        if awb_gains is None:
            vb_gain = 37 / 32
            vg_gain = 1.0  # raspi raw has already gain = 1 of green channel
            vr_gain = 63 / 32
        else:
            vb_gain = awb_gains[1]
            vg_gain = 1.0  # raspi raw has already gain = 1 of green channel
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
        print('Error in deraw: {}'.format(e))

def demosiac2(data, awb_gains = None):
    try:
        p1 = data[0::2, 1::2]  # Blue
        p2 = data[0::2, 0::2]  # Green
        p3 = data[1::2, 1::2]  # Green
        p4 = data[1::2, 0::2]  # Red

        blue = p1
        green = ((p2 + p3)) / 2
        red = p4

        if awb_gains is None:
            vb_gain = 1.3
            vr_gain = 1.8
        else:
            vb_gain = awb_gains[1]
            vr_gain = awb_gains[0]

        gamma = 1  # gamma correction
        vb = vb_gain
        vg = 1
        vr = vr_gain

        # color conversion matrix (from raspi_dng/dcraw)
        # R        g        b
        cvm = np.array(
            [[1.20, -0.30, 0.00],
             [-0.05, 0.80, 0.14],
             [0.20, 0.20, 0.7]])

        s = (1232, 1648, 3)
        rgb = np.zeros(s)

        rgb[:, :, 0] = vr * 1023 * (red / 1023.)   ** gamma
        rgb[:, :, 1] = vg * 1023 * (green / 1023.) ** gamma
        rgb[:, :, 2] = vb * 1023 * (blue / 1023.)  ** gamma

        # rgb = rgb.dot(cvm)

        rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb))

        height, width = rgb.shape[:2]

        img = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_CUBIC) # 16 - bit 'image'

        # down sample to RGB 8 bit image, use: self.deraw2rgb2(data)

        return img

    except Exception as e:
        print('data2rgb: Could not convert data to rgb: ' + str(e))

def toRGB_1(data):
    '''
    Belongs to deraw1
    :param data:
    :return:
    '''
    image = data // 256  # reduce dynamic range to 8 bpp
    image = np.clip(image, 0, 255).astype(np.uint8)

    return image

def toRGB_2(data):
    '''
    Belongs to deraw2
    :param data:
    :return:
    '''
    image = np.zeros(data.shape, dtype=np.float)
    min = data.min()
    image = data - min

    # get the max from out after normalizing to 0
    max = image.max()
    image *= (255 / max)
    image = np.uint8(image)

    return image

def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()

def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix

def simplest_cb(path_to_image, percent):

    data = np.fromfile(path_to_image, dtype='uint16')
    data = data.reshape([2464, 3296])
    img = demosaic1(data)

    assert img.shape[2] == 3
    assert percent > 0 and percent < 100

    half_percent = percent / 200.0

    channels = cv2.split(img)

    out_channels = []
    for channel in channels:
        assert len(channel.shape) == 2
        # find the low and high precentile values (based on the input percentile)
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)

        assert len(flat.shape) == 1

        flat = np.sort(flat)

        n_cols = flat.shape[0]

        low_val = flat[math.floor(n_cols * half_percent)]
        high_val = flat[math.ceil(n_cols * (1.0 - half_percent))]

        print("Lowval: ", low_val)
        print("Highval: ", high_val)

        # saturate below the low percentile and above the high percentile
        thresholded = apply_threshold(channel, low_val, high_val)
        # scale the channel
        img_16bit = (2**16) -1
        img_8bit = 255
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, img_16bit, cv2.NORM_MINMAX)
        out_channels.append(normalized)
        img = cv2.merge(out_channels)
        img = toRGB_1(img)
    return img

def resize_img(img, size = [1648,1232], set_equal_size = False):
    try:
        if (set_equal_size):
            w = size[0]
            h = size[1]
            img = cv2.resize(img,(w,h))
        return img

    except Exception as e:
        print("Could not resize image: "+ str(e))
        return img

def createHDR(images, times, output_path = None, prefix = "_"):

    try:
        # images, times = readImagesAndExpos(mypath,piclist) # orginal
        # images, times = readDataAndExpos(mypath,piclist) # data


        cnt_imgs  = str(images.size)
        cnt_times = str(times.size)

        if cnt_imgs == '0' or cnt_times == '0' :
            if not times:
                print("Could not create HDR images, no exposure time found! ")
            else:
                print("Could not create HDR images, no source images present!")
            return


        # Align input images
        print("Aligning images ... ")
        alignMTB = cv2.createAlignMTB()
        alignMTB.process(images, images)

         # Obtain Camera Response Function (CRF)
        print("Calculating Camera Response Function (CRF) ... ")
        calibrateDebevec = cv2.createCalibrateDebevec()
        responseDebevec = calibrateDebevec.process(images, times)

        # Merge images into an HDR linear image
        print("Merging images into one HDR image ... ")
        mergeDebevec = cv2.createMergeDebevec()
        hdrDebevec = mergeDebevec.process(images, times, responseDebevec)
        # Save HDR image.
        cv2.imwrite(join(output_path,prefix+"hdrDebevec.hdr"), hdrDebevec)
        print("saved "+prefix+"hdrDebevec.hdr ")

        # Tonemap using Drago's method to obtain 24-bit color image
        print("Tonemaping using Drago's method ... ")
        tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)
        ldrDrago = tonemapDrago.process(hdrDebevec)
        ldrDrago = 3 * ldrDrago
        cv2.imwrite(join(output_path,prefix+"ldr-Drago.jpg"), resize_img(ldrDrago * 255,False))
        print("saved "+prefix+"ldr-Drago.jpg")

        # Tonemap using Durand's method obtain 24-bit color image
        print("Tonemaping using Durand's method ... ")
        tonemapDurand = cv2.createTonemapDurand(1.5, 4, 1.0, 1, 1)
        ldrDurand = tonemapDurand.process(hdrDebevec)
        ldrDurand = 3 * ldrDurand
        cv2.imwrite(join(output_path,prefix+"ldr-Durand.jpg"), resize_img(ldrDurand * 255,False))
        print("saved "+prefix+"ldr-Durand.jpg")

        # Tonemap using Reinhard's method to obtain 24-bit color image
        print("Tonemaping using Reinhard's method ... ")
        tonemapReinhard = cv2.createTonemapReinhard(1.5, 0, 0, 0)
        ldrReinhard = tonemapReinhard.process(hdrDebevec)
        cv2.imwrite(join(output_path,prefix+"ldr-Reinhard.jpg"), resize_img(ldrReinhard * 255,False))
        print("saved "+prefix+"ldr-Reinhard.jpg")

        # Tonemap using Mantiuk's method to obtain 24-bit color image
        print("Tonemaping using Mantiuk's method ... ")
        tonemapMantiuk = cv2.createTonemapMantiuk(2.2, 0.85, 1.2)
        ldrMantiuk = tonemapMantiuk.process(hdrDebevec)
        ldrMantiuk = 3 * ldrMantiuk
        cv2.imwrite(join(output_path,prefix+"ldr-Mantiuk.jpg"), resize_img(ldrMantiuk * 255,False))
        print("saved "+prefix+"ldr-Mantiuk.jpg")
        print("\n Gespeichert unter: " + output_path)

    except Exception as e:
        print('createHDR: Error occured while creating HDR image: ' + str(e))

def setOwnerAndPermission(pathToFile):
    try:
        #uid = pwd.getpwnam('pi').pw_uid
        #gid = grp.getgrnam('pi').gr_gid
        #os.chown(pathToFile, uid, gid)
        #os.chmod(pathToFile, 0o777)
        return
    except IOError as e:
        print('PERM : Could not set permissions for file: ' + str(e))

def createNewFolder(Path):
    try:
        if os.path.exists(Path):
            shutil.rmtree(Path)
        os.makedirs(Path)
            #setOwnerAndPermission(Path)
    except IOError as e:
        print('DIR : Could not create new folder: ' + str(e))

def show(final,title=""):
    print('display')
    plt.imshow(final)
    plt.title(title)
    plt.show()

def whiten_maske(img):
    #out = image[np.where((image == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
    img[np.where((img <= [50, 50, 50]).all(axis=2))] = [255, 255, 255]

    return img

def main():
    try:
        global input_path

        createNewFolder(output_path_jpg)
        createNewFolder(output_path_raw)

        set_equal_size = False
        #images, times = readDataAndExpos(input_path, piclist)  # data
        images, times = new_readRawImages(input_path, SELECTION)  # data
        createHDR(images, times, output_path_raw)

        #images, times = readImagesAndExpos(input_path, piclist)  # jpg
        images, times = new_readImagesAndExpos(input_path, SELECTION)  # jpg
        createHDR(images, times, output_path_jpg)


    except Exception as e:
        print('MAIN: Error in main: ' + str(e))


if __name__ == '__main__':
    main()