#!/usr/bin/env python
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import cv2
import exifread
from fractions import Fraction
import shutil
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from glob import glob

global images_path
global hdr_path
#hdr_path = r'C:\Hoa_Python_Projects\python_scripts\hist\input\20171025_153519',0  #GESCHAEFTS-PC
images_path = r'C:\Users\tahorvat\PycharmProjects\python_scripts\hist\input\20171025_140139'
hdr_path = r'C:\Users\tahorvat\PycharmProjects\python_scripts\hist\output\20171025_140139'

def load_data_as_img():
    try:
        data_arrays = []
        for data in sorted(glob(images_path + '/*.data')):
            #load binary file to array
            np_arr = np.fromfile(data,dtype=np.uint16)
            print('shape: ' + str(np.shape(np_arr)))
            data_arrays.append(np_arr)
            #print(np_arr)

        return data_arrays

    except Exception as e:
        print('Could not load image: ' + str(e))

def data2img(raw):

    print('nraw matrix shape: '+ str(np.shape(raw)))
    p1 = raw[1:2:, 1:2:] # blue channel
    p2 = raw[1:2:, 2:2:] # green
    p3 = raw[2:2:, 1:2:] # scnd green
    p4 = raw[2:2:, 2:2:] # red



def data_to_image(fetched_img_data):
    print("length:" + str(len(fetched_img_data)))

    for n in range(0, len(fetched_img_data)):
        print('converting'+str(n))



def load_images_EXIF():
    try:
        fetched_imgs = []
        for file in sorted(glob(images_path + '/*.jpg')):
            fetched_imgs.append(file)
            print(file)

        image_stack = np.empty(len(fetched_imgs), dtype=object)
        expos_stack = np.empty(len(fetched_imgs), dtype=np.float32)

        for n in range(0, len(fetched_imgs)):
            image_stack[n] = cv2.imread(join(images_path, fetched_imgs[n]), cv2.IMREAD_COLOR)
            expos_stack[n] = getEXIF_TAG(join(images_path, fetched_imgs[n]), "EXIF ExposureTime")

        return image_stack, expos_stack

    except Exception as e:
        print('Could not load image: ' + str(e))


def load_images():
    try:
        fetched_imgs = []
        for file in sorted(glob(images_path + '/*.jpg')):
            fetched_imgs.append(file)
            print(file)

        image_stack = np.empty(len(fetched_imgs), dtype=object)

        for n in range(0, len(fetched_imgs)):
            image_stack[n] = cv2.imread(join(images_path, fetched_imgs[n]), cv2.IMREAD_COLOR)

        return image_stack

    except Exception as e:
        print('Could not load image: ' + str(e))


def load_img_as_nparr():
    try:
        onlyfiles = [f for f in listdir(images_path) if isfile(join(images_path, f))]
        image_stack = np.empty(len(onlyfiles))
        for n in range(0, len(onlyfiles)):
            img = cv2.imread(join(images_path, onlyfiles[n]), cv2.IMREAD_COLOR)
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

        return  foundvalue

    except Exception as e:
        print('EXIF: Could not read exif data ' + str(e))


#cols=1, titles=None

def create_HDR_images(image_stack, exposures, cols=1, titles=None):
    try:

        hdr_images = []

        # Align input images
        print("Aligning images ... ")
        alignMTB = cv2.createAlignMTB()
        alignMTB.process(image_stack, image_stack)

        # Obtain Camera Response Function (CRF)
        print("Calculating Camera Response Function (CRF) ... ")
        calibrateDebevec = cv2.createCalibrateDebevec()
        responseDebevec = calibrateDebevec.process(image_stack, exposures)

        # Merge images into an HDR linear image
        print("Merging images into one HDR image ... ")
        mergeDebevec = cv2.createMergeDebevec()
        hdrDebevec = mergeDebevec.process(image_stack, exposures, responseDebevec)
        # Save HDR image.
        cv2.imwrite(hdr_path + "/hdrDebevec.hdr", hdrDebevec)
        print("saved hdrDebevec.hdr ")
        hdr_images.append(hdrDebevec)

        # Tonemap using Drago's method to obtain 24-bit color image
        print("Tonemaping using Drago's method ... ")
        tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)
        ldrDrago = tonemapDrago.process(hdrDebevec)
        ldrDrago = 3 * ldrDrago
        cv2.imwrite(hdr_path +"/ldr-Drago.jpg", ldrDrago * 255)
        print("saved ldr-Drago.jpg")
        hdr_images.append(ldrDrago)

        # Tonemap using Durand's method obtain 24-bit color image
        print("Tonemaping using Durand's method ... ")
        tonemapDurand = cv2.createTonemapDurand(1.5, 4, 1.0, 1, 1)
        ldrDurand = tonemapDurand.process(hdrDebevec)
        ldrDurand = 3 * ldrDurand
        cv2.imwrite(hdr_path + "/ldr-Durand.jpg", ldrDurand * 255)
        print("saved ldr-Durand.jpg")
        hdr_images.append(ldrDurand)

        # Tonemap using Reinhard's method to obtain 24-bit color image
        print("Tonemaping using Reinhard's method ... ")
        tonemapReinhard = cv2.createTonemapReinhard(1.5, 0, 0, 0)
        ldrReinhard = tonemapReinhard.process(hdrDebevec)
        cv2.imwrite(hdr_path + "/ldr-Reinhard.jpg", ldrReinhard * 255)
        print("saved ldr-Reinhard.jpg")
        hdr_images.append(ldrReinhard)

        # Tonemap using Mantiuk's method to obtain 24-bit color image
        print("Tonemaping using Mantiuk's method ... ")
        tonemapMantiuk = cv2.createTonemapMantiuk(2.2, 0.85, 1.2)
        ldrMantiuk = tonemapMantiuk.process(hdrDebevec)
        ldrMantiuk = 3 * ldrMantiuk
        cv2.imwrite(hdr_path +"/ldr-Mantiuk.jpg", ldrMantiuk * 255)
        print("saved ldr-Mantiuk.jpg")
        hdr_images.append(ldrMantiuk)

        print("Show hdr images:")

        n_images = len(hdr_images)
        if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
        fig = plt.figure()
        for n, (image, title) in enumerate(zip(hdr_images, titles)):
            a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
            if image.ndim == 2:
                plt.gray()
            plt.imshow(image)
            a.set_title(title)
        fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
        plt.show()


    except Exception as e:
        print('readImageAndTimes: Could not read images ' + str(e))
