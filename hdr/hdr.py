#!/usr/bin/env python

import cv2
import numpy as np
import os
import exifread
#import pwd
#import grp
from os import listdir
from os.path import isfile, join
from fractions import Fraction
import shutil

print('Version opencv: ' + cv2.__version__)

'''
hdr.py 
Version 2
29.10.2017
Attila Horvath
'''

global Path_to_raw
Path_to_raw ='./jpg/2'


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


def readImagesAndExpos(mypath,piclist):
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
                    image_stack[pos] = cv2.imread(join(mypath, onlyfiles[n]), cv2.IMREAD_COLOR)
                pos = pos + 1

        return image_stack, expos_stack

    except Exception as e:
        print('readImagesAndExpos: Could not read images ' + str(e))


def createHDR(mypath,piclist):

    try:
        images, times = readImagesAndExpos(mypath,piclist)

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
        cv2.imwrite("./output/hdrDebevec.hdr", hdrDebevec)
        print("saved hdrDebevec.hdr ")

        # Tonemap using Drago's method to obtain 24-bit color image
        print("Tonemaping using Drago's method ... ")
        tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)
        ldrDrago = tonemapDrago.process(hdrDebevec)
        ldrDrago = 3 * ldrDrago
        cv2.imwrite("./output/ldr-Drago.jpg", ldrDrago * 255)
        print("saved ldr-Drago.jpg")

        # Tonemap using Durand's method obtain 24-bit color image
        print("Tonemaping using Durand's method ... ")
        tonemapDurand = cv2.createTonemapDurand(1.5, 4, 1.0, 1, 1)
        ldrDurand = tonemapDurand.process(hdrDebevec)
        ldrDurand = 3 * ldrDurand
        cv2.imwrite("./output/ldr-Durand.jpg", ldrDurand * 255)
        print("saved ldr-Durand.jpg")

        # Tonemap using Reinhard's method to obtain 24-bit color image
        print("Tonemaping using Reinhard's method ... ")
        tonemapReinhard = cv2.createTonemapReinhard(1.5, 0, 0, 0)
        ldrReinhard = tonemapReinhard.process(hdrDebevec)
        cv2.imwrite("./output/ldr-Reinhard.jpg", ldrReinhard * 255)
        print("saved ldr-Reinhard.jpg")

        # Tonemap using Mantiuk's method to obtain 24-bit color image
        print("Tonemaping using Mantiuk's method ... ")
        tonemapMantiuk = cv2.createTonemapMantiuk(2.2, 0.85, 1.2)
        ldrMantiuk = tonemapMantiuk.process(hdrDebevec)
        ldrMantiuk = 3 * ldrMantiuk
        cv2.imwrite("./output/ldr-Mantiuk.jpg", ldrMantiuk * 255)
        print("saved ldr-Mantiuk.jpg")

    except Exception as e:
        print('readImageAndTimes: Could not read images ' + str(e))

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

def main():
    try:
        global Path_to_raw
        createNewFolder('./ouput')
        createHDR(Path_to_raw,['3','4','5'])

    except Exception as e:
        print('MAIN: Error in main: ' + str(e))


if __name__ == '__main__':
    main()