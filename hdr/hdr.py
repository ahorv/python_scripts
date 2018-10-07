#!/usr/bin/env python

import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import exifread
#import pwd
#import grp
from os import listdir
from os.path import isfile, join
from glob import glob
from fractions import Fraction
import shutil

print('Version opencv: ' + cv2.__version__)

######################################################################
## Hoa: 18.09.2018 Version 2 : hdr.py
######################################################################
# Generates with aid of opencv a HDR image either from 3 jpg - images
# or from 3 *.data files.
#
# New /Changes:
# ----------------------------------------------------------------------
#
# 10.11.2017 : Added new logging
# 18.09,2018 : reads *.data - files
#
#
######################################################################

global input_path
global output_path
input_path  = r'I:\SkY_CAM_IMGS\picam\camera_3\20181007\20181007_120426'
output_path = r'I:I:\SkY_CAM_IMGS\picam\camera_3\20181007\20181007_120426\_output'

global set_equal_size
set_equal_size = True

global mask

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

def resize_img(img, size = [1648,1232]):
    try:
        if (set_equal_size):
            w = size[0]
            h = size[1]
            img = cv2.resize(img,(w,h))
        return img

    except Exception as e:
        print("Could not resize image: "+ str(e))
        return img

def createHDR(images, times, prefix = "_"):

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
        cv2.imwrite(join(output_path,prefix+"ldr-Drago.jpg"), resize_img(ldrDrago * 255))
        print("saved "+prefix+"ldr-Drago.jpg")

        # Tonemap using Durand's method obtain 24-bit color image
        print("Tonemaping using Durand's method ... ")
        tonemapDurand = cv2.createTonemapDurand(1.5, 4, 1.0, 1, 1)
        ldrDurand = tonemapDurand.process(hdrDebevec)
        ldrDurand = 3 * ldrDurand
        cv2.imwrite(join(output_path,prefix+"ldr-Durand.jpg"), resize_img(ldrDurand * 255))
        print("saved "+prefix+"ldr-Durand.jpg")

        # Tonemap using Reinhard's method to obtain 24-bit color image
        print("Tonemaping using Reinhard's method ... ")
        tonemapReinhard = cv2.createTonemapReinhard(1.5, 0, 0, 0)
        ldrReinhard = tonemapReinhard.process(hdrDebevec)
        cv2.imwrite(join(output_path,prefix+"ldr-Reinhard.jpg"), resize_img(ldrReinhard * 255))
        print("saved "+prefix+"ldr-Reinhard.jpg")

        # Tonemap using Mantiuk's method to obtain 24-bit color image
        print("Tonemaping using Mantiuk's method ... ")
        tonemapMantiuk = cv2.createTonemapMantiuk(2.2, 0.85, 1.2)
        ldrMantiuk = tonemapMantiuk.process(hdrDebevec)
        ldrMantiuk = 3 * ldrMantiuk
        cv2.imwrite(join(output_path,prefix+"ldr-Mantiuk.jpg"), resize_img(ldrMantiuk * 255))
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
            setOwnerAndPermission(Path)
    except IOError as e:
        print('DIR : Could not create new folder: ' + str(e))

def white_balance(img):
    #https://stackoverflow.com/questions/46390779/automatic-white-balancing-with-grayworld-assumption
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

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
        global set_equal_size
        piclist = ['0','5','9']

        # Show mask
        #img_path = r'F:\SkyCam\camera_1\20180429_raw_cam1\Test\kopie_20180904_100917\_output\ldr-Drago.jpg'
        #img = white_balance(cv2.imread(img_path))
        #show(img,"HDR image white balanced")

        set_equal_size = False
        images, times = readDataAndExpos(input_path, piclist)  # data
        createHDR(images, times, prefix="")

        return

        images, times = readImagesAndExpos(input_path, piclist)  # jpg
        createHDR(images, times)



    except Exception as e:
        print('MAIN: Error in main: ' + str(e))


if __name__ == '__main__':
    main()