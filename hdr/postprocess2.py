#!/usr/bin/env python

from __future__ import print_function

import os
import cv2
import sys
import time
import re
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

###############################################################################
## Hoa: 03.10.2018 Version 1 : postprocess2.py
###############################################################################
# Collects *.jpg and .data images and creates HDR pictures. In addition:
#
# - creates video from pictures
# - shows pictures as slideshow
#
# Remarks:
# - images taken by picam are already masked
# - Shuterspeed in microseconds (picamera)
#
# New /Changes:
# -----------------------------------------------------------------------------
#
# 03.10.2018 : New version of postprocess adapted for pictures taken by picam
#
###############################################################################

global Path_to_raw
global Path_to_copy_wellExp
global Path_to_ffmpeg
global Avoid_This_Directories
global CAM
global LogFileName
global SELECTION
global NameWellExpImg
Path_to_raw = r'C:\Users\ati\Desktop\camera_1'
#Path_to_raw = r'G:\SkyCam\camera_1\20180429_raw_cam1'  # ACHTUNG BEACHTE LAUFWERKS BUCHSTABEN
Path_to_copy_wellExp = os.path.join(Path_to_raw, 'wellExp')
Path_to_copy_HDR = os.path.join(Path_to_raw,'hdr')
Path_to_ffmpeg = r'C:\ffmpeg\bin\ffmpeg.exe'
Avoid_This_Directories = ['wellExp','hdr','rest']
CAM = Path_to_raw.rstrip('\\').rpartition('\\')[-1][-1]  # determin if cam1 or cam2
LogFileName = 'camstats.log'
SELECTION = [0, -2, -4]
NameWellExpImg = 'raw_img0'


class Helpers:
    def createVideo(self):
        try:

            global Path_to_copy_wellExp                                   # path to images
            global Path_to_ffmpeg                                       # path to ffmpeg executable
            fsp = ' -r 10 '                                             # frame per sec images taken
            stnb = '-start_number 0001 '                   # what image to start at
            imgpath = '-i ' + join(Path_to_copy_wellExp, '%4d.jpg') + ' '  # path to images
            res = '-s 2592x1944 '                                       # output resolution
            outpath = Path_to_copy_wellExp + '\sky_video.mp4 '            # output file name
            codec = '-vcodec libx264'                                   # codec to use

            command = Path_to_ffmpeg + fsp + stnb + imgpath + res + outpath + codec

            if sys.platform == "linux":
                subprocess(command, shell=True)
            else:
                print('\n{}'.format(command))
                ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE ,stdout = subprocess.PIPE)
                out, err = ffmpeg.communicate()
                if (err): print(err)
                print('Ffmpeg done.')

        except Exception as e:
            print('createVideo from jpg\'s: Error: ' + str(e))

    def getDirectories(self,pathToDirectories):
        try:
            global Avoid_This_Directories
            allDirs = []
            img_cnt = 1

            for dirs in sorted(glob(os.path.join(pathToDirectories, "*", ""))):
                if os.path.isdir(dirs):
                    if dirs.rstrip('\\').rpartition('\\')[-1] not in Avoid_This_Directories:
                        allDirs.append(dirs)
                        #print('{}'.format(str(dirs)))
                        img_cnt +=1

            print('All images loaded! - Found {} images.'.format(img_cnt))

            return allDirs

        except Exception as e:
            print('getDirectories: Error: ' + str(e))

    def getZipDirs(self,pathToDirectories):
        allZipFiles = []
        cnt = 0

        for zipfile in sorted(glob(os.path.join(pathToDirectories, "*.zip"))):
            if os.path.isfile(zipfile):
                allZipFiles.append(zipfile)
                cnt +=1

        print('Found {} *.ZIP files '.format(cnt))
        return allZipFiles

    def readAllImages(self,allDirs):
        try:
            global Path_to_raw
            list_names = []
            list_images = []
            cnt = 1
            print('Converting jpg to opencv, may take some time!')

            t_start = time.time()
            for next_dir in allDirs:
                next_dir += NameWellExpImg
                img = cv2.imread(next_dir, 1)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                new_img = cv2.resize(img_rgb, None, fx=0.25, fy=0.25)
                list_images.append(new_img)
                cnt += 1

            t_end = time.time()
            measurement = t_end-t_start
            print('Total number of images: {}'.format(cnt))
            print('Time to load and convert imgs: {}'.format(measurement))

            return list_images

        except Exception as e:
            print('readAllImages: Error: ' + str(e))

    def copyAndTagAllWellExposed(self, list_alldirs):

        try:
            imgproc = IMGPROC()
            global Path_to_copy_wellExp
            global CAM
            cnt = 1

            if not os.path.exists(Path_to_copy_wellExp):
                os.makedirs(Path_to_copy_wellExp)

            for next_dir in list_alldirs:
                newimg = join(next_dir, NameWellExpImg + '.jpg')
                next_img = cv2.imread(newimg)
                dateAndTime = (next_dir.rstrip('\\').rpartition('\\')[-1]).replace('_',' ')
                prefix = '{0:04d}'.format(cnt)
                year  = dateAndTime[:4]
                month = dateAndTime[4:6]
                day   = dateAndTime[6:8]
                hour  = dateAndTime[9:11]
                min   = dateAndTime[11:13]
                sec   = dateAndTime[13:15]

                img_txt = imgproc.write2img(next_img, 'cam ' + CAM, (30, 70))
                img_txt = imgproc.write2img(next_img,year+" "+month+" "+day,(30,1720))
                img_txt = imgproc.write2img(next_img, '#: ' + str(cnt), (30,1800))
                img_txt = imgproc.write2img(next_img, hour+":"+min+":"+sec, (30, 1880))
                new_img_path = join(Path_to_copy_wellExp, '{}.jpg'.format(prefix))

                cv2.imwrite(new_img_path,next_img)
                cnt += 1

            print(' {} images copyied to wellExp.'.format(cnt))

        except Exception as e:
            print('Error in copyAndMaskAll_img5: {}'.format(e))

    def unzipall(self,path_to_extract):

        try:
            allzipDirs = self.getZipDirs(path_to_extract)
            numb_to_unzip = len(allzipDirs)
            cnt = 0

            for dirs in allzipDirs:

                newDirname = dirs.replace('.zip','')

                if newDirname:

                    zipfilepath = os.path.join(dirs)
                    cnt +=1
                    zf = zipfile.ZipFile(zipfilepath, "r")
                    zf.extractall(os.path.join(newDirname))
                    zf.close()
                    percent = round(cnt/numb_to_unzip*100,2)
                    print(str(percent)+' percent completed' + '\r')

            print('unzip finished.')

        except IOError as e:
            print('unzipall: Error: ' + str(e))

    def delAllZIP(self,path_to_extract):

        try:
            allzipDirs = self.getZipDirs(path_to_extract)
            numb_to_unzip = len(allzipDirs)
            cnt = 0

            for zipdir in allzipDirs:
                # delete unzipped directory
                print('deleting: '+str(zipdir))
                os.remove(zipdir)

            print('deleted all ZIP files.')

        except IOError as e:
            print('unzipall: Error: ' + str(e))

class HDR:
    def getShutterTimes(self, file_path, file_name = LogFileName):
        try:
            '''
            returns shutter_time in microseconds as np.float32 type
            '''
            listOfSS = np.empty(3, dtype=np.float32)

            f = open(join(file_path, file_name), 'r')
            logfile = f.readlines()
            logfile.pop(0) # remove non relevant lines
            logfile.pop(0) # remove non relevant lines

            pos = 0
            for line in logfile:
                value = line.split("ss:", 1)[1]
                value = value.split(',', 1)[0]
                value = value.strip()
                print('new ss: {}'.format(value))
                value += '/1000000'
                val_float = np.float32(Fraction(str(value)))
                listOfSS[pos] = val_float
                pos +=1

            return listOfSS

        except Exception as e:
            print('Error in getShutterTimes: ' + str(e))

    def getEXIF_TAG(self, file_path, field):
        try:
            foundvalue = '0'
            with open(file_path, 'rb') as f:
                exif = exifread.process_file(f)

            for k in sorted(exif.keys()):
                if k not in ['JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote']:
                    if k == field:
                        # print('%s = %s' % (k, exif[k]))
                        foundvalue = np.float32(Fraction(str(exif[k])))
                        break

            return foundvalue

        except Exception as e:
            print('EXIF: Could not read exif data ' + str(e))

    def readRawImages(self, mypath, piclist = SELECTION):
        try:
            onlyfiles_data = [f for f in listdir(mypath) if isfile(join(mypath, f)) & f.endswith('.data')]
            image_stack = np.empty(len(piclist), dtype=object)
            expos_stack = self.getShutterTimes(mypath)

            imrows = 2464
            imcols = 3296

            imsize = imrows * imcols

            # Importing and debayering raw images
            for n in range(0, len(onlyfiles_data)):
                picnumber = [str(s.replace('.', '')) for s in re.findall(r'-?\d+\.?\d*', onlyfiles_data[n])]
                pos = 0
                for pic in piclist:
                    if str(picnumber[0]) == str(pic):

                        with open(join(mypath, onlyfiles_data[n]), "rb") as rawimage:
                            # images[image_idx] = cv2.imread(path)
                            img_np = np.fromfile(rawimage, np.dtype('u2'), imsize).reshape([imrows,imcols])
                            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BAYER_BG2BGR)

                        image_stack[pos] = img_rgb
                        values = [str(picnumber), onlyfiles_data[n], str(round(expos_stack[n],5))]
                        print('Pic {}, reading data: {}, shutter time: {}'.format(values))
                    pos +=1

            return image_stack, expos_stack

        except Exception as e:
            print('readRawImages: Could not read *.data files ' + str(e))

    def readImagesAndExpos(self, mypath, piclist = SELECTION):
        try:
            onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) & f.endswith('.jpg')]
            image_stack = np.empty(len(piclist), dtype=object)       # Achtung len = onlyfiles für alle bilder
            #expos_stack = np.empty(len(piclist), dtype=np.float32)  # Achtung len = onlyfiles für alle bilder
            expos_stack = self.getShutterTimes(mypath)

            for n in range(0, len(onlyfiles)):
                picnumber = [str(s.replace('.', '')) for s in re.findall(r'-?\d+\.?\d*', onlyfiles[n])]
                pos = 0
                for pic in piclist:
                    if str(picnumber[0]) == str(pic):
                        image_stack[pos] = cv2.imread(join(mypath, onlyfiles[n]), cv2.IMREAD_COLOR)
                        print('Pic {}, reading data from : {}, exif: {}'.format(str(picnumber), onlyfiles[n], expos_stack[n]))
                    pos +=1

            return image_stack, expos_stack

        except Exception as e:
            print('Error in readImagesAndExpos: ' + str(e))

    def composeOneHDRimgJpg(self, oneDirsPath, piclist = SELECTION):
        try:

            images, times = self.readImagesAndExpos(oneDirsPath, piclist)

            # Align input images
            alignMTB = cv2.createAlignMTB()
            alignMTB.process(images, images)

            # Obtain Camera Response Function (CRF)
            calibrateDebevec = cv2.createCalibrateDebevec()
            responseDebevec = calibrateDebevec.process(images, times)

            # Merge images into an HDR linear image
            mergeDebevec = cv2.createMergeDebevec()
            hdrDebevec = mergeDebevec.process(images, times, responseDebevec)

            # Tonemap using Reinhard's method to obtain 24-bit color image
            tonemapReinhard = cv2.createTonemapReinhard(1.5, 0, 0, 0)
            ldrReinhard = tonemapReinhard.process(hdrDebevec)

            _ldrReinhard = ldrReinhard * 255

            return _ldrReinhard

        except Exception as e:
            print('composeOneHDRimg: Error: ' + str(e))

    def composeOneHDRimgData(self, oneDirsPath, piclist = SELECTION):
        try:

            images, times = self.readRawImages(oneDirsPath, piclist)

            # Align input images
            alignMTB = cv2.createAlignMTB()
            alignMTB.process(images, images)

            # Obtain Camera Response Function (CRF)
            calibrateDebevec = cv2.createCalibrateDebevec()
            responseDebevec = calibrateDebevec.process(images, times)

            # Merge images into an HDR linear image
            mergeDebevec = cv2.createMergeDebevec()
            hdrDebevec = mergeDebevec.process(images, times, responseDebevec)

            # Tonemap using Reinhard's method to obtain 24-bit color image
            tonemapReinhard = cv2.createTonemapReinhard(1.5, 0, 0, 0)
            ldrReinhard = tonemapReinhard.process(hdrDebevec)

            _ldrReinhard = ldrReinhard * 255

            return _ldrReinhard

        except Exception as e:
            print('composeOneHDRimgData: Error: ' + str(e))

    def makeHDR_from_jpg(self, ListofAllDirs):
        global Path_to_raw
        global CAM
        imgproc = IMGPROC()
        try:
            cnt = 0
            if not os.path.exists(join(Path_to_raw,'hdr')):
                os.makedirs(join(Path_to_raw,'hdr'))

            for next_dir in ListofAllDirs:
                cnt += 1
                prefix = '{0:04d}'.format(cnt)
                ldrReinhard = self.composeOneHDRimgJpg(next_dir)

                remasked_img = imgproc.maske_jpg_Image(ldrReinhard)

                dateAndTime = (next_dir.rstrip('\\').rpartition('\\')[-1]).replace('_',' ')
                year  = dateAndTime[:4]
                month = dateAndTime[4:6]
                day   = dateAndTime[6:8]
                hour  = dateAndTime[9:11]
                min   = dateAndTime[11:13]
                sec   = dateAndTime[13:15]

                ldrReinhard_txt = imgproc.write2img(remasked_img,'cam '+CAM,(30,70))
                ldrReinhard_txt = imgproc.write2img(remasked_img,year+" "+month+" "+day,(30,1720))
                ldrReinhard_txt = imgproc.write2img(remasked_img, '#: ' + str(cnt), (30,1800))
                ldrReinhard_txt = imgproc.write2img(remasked_img, hour+":"+min+":"+sec, (30, 1880))

                cv2.imwrite(join(Path_to_raw, 'hdr', "{}.jpg".format(prefix)), ldrReinhard_txt)

            print("Done creating all HDR images")

        except Exception as e:
            print('createAllHDR: Error: ' + str(e))

    def makeHDR_from_data(self, ListofAllDirs):
        global Path_to_raw
        try:
            cnt = 0
            if not os.path.exists(join(Path_to_raw,'raw_hdr')):
                os.makedirs(join(Path_to_raw,'raw_hdr'))

            for oneDir in ListofAllDirs:
                cnt += 1
                ldrReinhard = self.composeOneHDRimgData(oneDir)
                cv2.imwrite(join(Path_to_raw,'raw_hdr',str(cnt) + "_ldr-Reinhard.jpg"), ldrReinhard)

            print("Done creating all HDR images")

        except Exception as e:
            print('createAllHDR: Error: ' + str(e))

    def createHDRVideo(self):
        try:
            global Path_to_raw
            hdrpath = join(Path_to_raw, 'hdr')
            global Path_to_ffmpeg                                 # path to ffmpeg executable
            fsp = ' -r 10 '                                       # frame per sec images taken
            stnb = '-start_number 0001 '                          # what image to start at
            imgpath = '-i ' + join(hdrpath ,'%4d.jpg ')           # path to images
            res = '-s 2592x1944 '                                 # output resolution
            outpath = Path_to_copy_HDR+'\sky_HDR_video.mp4 '      # output file name
            codec = '-vcodec libx264'                             # codec to use

            command = Path_to_ffmpeg + fsp + stnb + imgpath + res + outpath + codec

            if sys.platform == "linux":
                subprocess(command, shell=True)
            else:
                print(' {}'.format(command))
                ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE ,stdout = subprocess.PIPE)
                out, err = ffmpeg.communicate()
                if (err): print(err)
                print('ffmpeg ldr Video done.')

        except Exception as e:
            print('createVideo: Error: ' + str(e))

class IMGPROC(object):

    def __init__(self):
        global CAM
        # Create image mask
        size = 1944, 2592, 3
        empty_img = np.zeros(size, dtype=np.uint8)
        if CAM is '1':
            self.mask = self.cmask([880, 1190], 1117, empty_img)
        if CAM is '2':
            self.mask = self.cmask([950, 1340], 1117, empty_img)

    def cmask(self, index, radius, array):
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

    def maske_jpg_Image(self, input_image):

        red   = input_image[:, :, 0]
        green = input_image[:, :, 1]
        blue  = input_image[:, :, 2]

        r_img = red.astype(float)   * self.mask
        g_img = green.astype(float) * self.mask
        b_img = blue.astype(float) * self.mask

        dimension = (input_image.shape[0], input_image.shape[1], 3)
        output_img = np.zeros(dimension, dtype=np.uint8)

        output_img[..., 0] = r_img[:, :]
        output_img[..., 1] = g_img[:, :]
        output_img[..., 2] = b_img[:, :]

        return output_img

    def write2img(self,input_image,text, xy):

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = xy
        fontScale = 2
        fontColor = (255, 255, 255)
        lineType = 3

        output_image = cv2.putText(input_image, text,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        return output_image

def main():
    try:
        global Path_to_raw
        unzipall          = False
        delallzip         = False
        runslideshow      = False
        copyAndTag        = False
        hdr_pics_from_jpg = True
        creat_HDR_Video   = False
        hdr_from_data     = False

        if not os.path.isdir(Path_to_raw):
            print('\nError: Image directory does not exist! -> Aborting.')
            return;

        help = Helpers()
        hdr = HDR()
        allDirs = help.getDirectories(Path_to_raw)

        if unzipall:
            help.unzipall(Path_to_raw)

        if delallzip:
            help.delAllZIP(Path_to_raw)

        if copyAndTag:
            help.copyAndTagAllWellExposed(allDirs)
            help.createVideo()

        if hdr_pics_from_jpg:
            hdrstart = time.time()
            hdr.makeHDR_from_jpg(allDirs)
            hdrend = time.time()
            print('Time to create HDR images: {}'.format(hdrend-hdrstart))

        if creat_HDR_Video:
            hdrstart = time.time()
            hdr.createHDRVideo()
            hdrend = time.time()
            print('Time to create HDR Video: {}'.format(hdrend-hdrstart))

        if hdr_from_data:
            hdr.makeHDR_from_data(allDirs)

        if runslideshow:

            counter = 0
            fig = plt.figure()
            ax = plt.gca()

            list_images = help.readAllImages(allDirs)
            cur_window = ax.imshow(list_images[0])

            while counter < len(list_images):

                next = list_images[counter]
                plt.title('Image: {}'.format(counter))
                cur_window.set_data(next)

                plt.pause(.05)
                plt.draw()

                counter += 1

        print('Postprocess.py done')

    except Exception as e:
        print('MAIN: Error in main: ' + str(e))


if __name__ == '__main__':
    main()