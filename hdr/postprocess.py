#!/usr/bin/env python

from __future__ import print_function

import mysql.connector
from mysql.connector import Error
import os
import cv2
import sys
import time
import exifread
from glob import glob
import subprocess
import zipfile
import shutil
from os import listdir
from os.path import isfile, join
import numpy as np
from fractions import Fraction
import logging
import logging.handlers
from datetime import datetime

if sys.platform == "linux":
    import pwd
    import grp


print('Version opencv: ' + cv2.__version__)

######################################################################
## Hoa: 18.10.2018 Version 1 : postprocess.py
######################################################################
# Reads all images from FTP - server. Creates HDR images. Data and
# Images are stored in SQLite database.
# Remarks:
# - use phpmyadmin to see contents of database#
#
# New /Changes:
# ----------------------------------------------------------------------
#
# 18.10.2018 : first implemented
#
######################################################################


class Image_Data(object):
    """Container class for image data.
    """
    img_nr = 0
    shots = 0
    time = '?'
    fstop ='?'
    ss = '?'
    exp ='?'
    iso = 0
    ag = '?'
    awb_red = '?'
    awb_blue = '?'
    ldr = 0
    hdr = 0

    def __init__(self, state_map={}):
        self.img_nr = state_map.get('img_nr',0)
        self.shots = state_map.get('shots', 0)
        self.time = state_map.get('time','?' )
        self.fstop = state_map.get('fstop', '?')
        self.ss = state_map.get('ss', '?')
        self.exp = state_map.get('exp', '?')
        self.iso = state_map.get('iso', 0)
        self.ag = state_map.get('ag', '?')
        self.awb_red = state_map.get('awb_red', '?')
        self.awb_blue = state_map.get('awb_blue', '?')
        self.ldr = state_map.get('ldr', 0)
        self.hdr = state_map.get('hdr', 0)

        Image_Data.img_nr = self.img_nr
        Image_Data.shots = self.shots
        Image_Data.time = self.time
        Image_Data.fstop = self.fstop
        Image_Data.ss = self.ss
        Image_Data.exp = self.exp
        Image_Data.iso = self.iso
        Image_Data.ag = self.ag
        Image_Data.awb_red = self.awb_red
        Image_Data.awb_blue = self.awb_blue
        Image_Data.ldr = self.ldr
        Image_Data.hdr = self.hdr

    def to_dict(self):
        return {
            'img_nr': Image_Data.img_nr,
            'time':   Image_Data.time,
            'fstop':  Image_Data.fstop,
            'ss' :    Image_Data.ss,
            'exp':    Image_Data.exp,
            'iso':    Image_Data.iso,
            'ag':     Image_Data.ag,
            'awb_red' : Image_Data.awb_red,
            'awb_blue': Image_Data.awb_blue,
            'ldr': Image_Data.ldr,
            'hdr': Image_Data.hdr,
        }

class Camera_Data(object):
    """Container class for camera data.
    """
    sw_vers = '?'
    cam_id = '?'
    image_date = '?'
    dont_use = 0
    was_clearsky = 0
    was_rainy = 0
    was_biased = 0
    was_foggy = 0
    had_nimbocum = 0

    def __init__(self, state_map={}):
        self.sw_vers = state_map.get('sw_vers','?')
        self.cam_id = state_map.get('cam_id','?' )
        self.image_date = state_map.get('image_date', '?')
        self.dont_use     = state_map.get('dont_use', 0)
        self.was_clearsky = state_map.get('was_clearsky', 0)
        self.was_rainy    = state_map.get('was_rainy', 0)
        self.was_biased   = state_map.get('was_biased', 0)
        self.was_foggy    = state_map.get('was_foggy', 0)
        self.had_nimbocum = state_map.get('had_nimbocum', 0)

        Camera_Data.sw_vers = self.sw_vers
        Camera_Data.cam_id = self.cam_id
        Camera_Data.image_date = self.image_date
        Camera_Data.dont_use = self.dont_use
        Camera_Data.was_clearsky = self.was_clearsky
        Camera_Data.was_rainy = self.was_rainy
        Camera_Data.was_biased = self.was_biased
        Camera_Data.was_foggy = self.was_foggy
        Camera_Data.had_nimbocum = self.had_nimbocum

    def to_dict(self):
        return {
            'sw_vers'       :Camera_Data.sw_vers,
            'cam_id'        :Camera_Data.cam_id,
            'image_date'    :Camera_Data.image_date,
            'dont_use'      :Camera_Data.dont_use,
            'was_clearsky'  :Camera_Data.was_clearsky,
            'was_rainy'     :Camera_Data.was_rainy,
            'was_biased'    :Camera_Data.was_biased,
            'was_foggy'     :Camera_Data.was_foggy,
            'had_nimbocum'  :Camera_Data.had_nimbocum,
        }

class Config(object):
    """Container class for configuration.
    """

    sourceDirectory = '?'
    camera_1_Directory = '?'
    camera_2_Directory = '?'
    databaseName = '?'
    databaseDirectory = '?'

    def __init__(self, state_map={}):
        self.sourceDirectory = state_map.get('sourceDirectory', '?')
        self.camera_1_Directory = state_map.get('camera_1_Directory', '?')
        self.camera_2_Directory = state_map.get('camera_2_Directory', '?')
        self.databaseName = state_map.get('databaseName','sky_db')
        self.databaseDirectory = state_map.get('databaseDirectory', '?')

        Config.sourceDirectory = self.sourceDirectory
        Config.camera_1_Directory = self.camera_1_Directory
        Config.camera_2_Directory = self.camera_2_Directory
        Config.databaseName = self.databaseName
        Config.databaseDirectory = self.databaseDirectory

class Logger:
    def __init__(self):
        self.logger = None

    def getFileLogger(self):
        try:
            PATH = Config.databaseDirectory

            FILEPATH = os.path.join(PATH, 'allFilesProcessed.log')
            LOGFILEPATH = join(r'\\',FILEPATH)

            logFormatter = logging.Formatter('%(message)s')
            fileHandler = logging.FileHandler(LOGFILEPATH)
            name = 'filesProcessedLogger'

            # configure file handler
            fileHandler.setFormatter(logFormatter)

            # configure stream handler
            consoleHandler = logging.StreamHandler()
            consoleHandler.setFormatter(logFormatter)

            # get the logger instance
            self.logger = logging.getLogger(name)

            # set the logging level
            self.logger.setLevel(logging.INFO)

            if not len(self.logger.handlers):
                self.logger.addHandler(fileHandler)
                self.logger.addHandler(consoleHandler)

            helper = Helpers()
            if sys.platform == "linux":
                helper.setOwnerAndPermission(LOGFILEPATH)
            return self.logger

        except Exception as e:
            print('Error Filelogger:' + str(e))

    def getLogger(self, newLogPath=None):

        try:
            PATH = Config.databaseDirectory

            if newLogPath is None:
                FILEPATH = os.path.join(PATH, 'postprocessor.log')
                LOGFILEPATH = join(r'\\',FILEPATH)

                logFormatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
                fileHandler = logging.FileHandler(LOGFILEPATH)
                name = 'rootlogger'
            else:
                LOGFILEPATH = newLogPath
                logFormatter = logging.Formatter('%(message)s')
                fileHandler = logging.FileHandler(LOGFILEPATH)
                name = 'postprocessLogger'

            # configure file handler
            fileHandler.setFormatter(logFormatter)

            # configure stream handler
            consoleHandler = logging.StreamHandler()
            consoleHandler.setFormatter(logFormatter)

            # get the logger instance
            self.logger = logging.getLogger(name)

            # set the logging level
            self.logger.setLevel(logging.INFO)

            if not len(self.logger.handlers):
                self.logger.addHandler(fileHandler)
                self.logger.addHandler(consoleHandler)

            helper = Helpers()
            if sys.platform == "linux":
                helper.setOwnerAndPermission(LOGFILEPATH)
            return self.logger

        except IOError as e:
            print('Error Logger:' + str(e))

    def closeLogHandler(self):
        try:
            handlers = self.logger.handlers[:]
            for handler in handlers:
                handler.close()
                self.logger.removeHandler(handler)

        except IOError as e:
            print('Error logger:' + str(e))

class DB_handler:

    def __init__(self):
        self.connection = None
        self.cursor = None

    def connect2MySQL(self):
        try:
            s = Logger()
            logger = s.getLogger()
            self.connection = mysql.connector.connect(
                            host='192.168.1.10',
                            user='root',
                            password='123ihomelab'
                            )
            if self.connection.is_connected():
                return self.connection

        except Error as e:
            logger.error('Could not connect to NAS: {}'.format(e))
            self.connection.close()

    def connect2DB(self):
        try:
            s = Logger()
            logger = s.getLogger()
            cb_name = Config.databaseName
            self.connection = mysql.connector.connect(
                host='192.168.1.10',
                user='root',
                password='123ihomelab',
                database = cb_name,
                connect_timeout=1000
            )
            if self.connection .is_connected():
                return self.connection

        except Error as e:
            logger.error('Could not get database cursor: {}'.format(e))
            self.connection.close()

    def con_close(self):
        try:
            s = Logger()
            logger = s.getLogger()
            if self.cursor:
                self.cursor.close()

            if self.connection.is_connected():
                self.connection.close()

        except Error as e:
            logger.error('Could not close database connection: {}'.format(e))

    def createDB(self):
        try:
            success = False
            s = Logger()
            logger = s.getLogger()
            db_con = self.connect2MySQL()
            db_name = Config.databaseName

            if(db_con):
                myDB = db_con.cursor()
                myDB.execute("CREATE DATABASE IF NOT EXISTS {} DEFAULT CHARACTER SET 'utf8'".format(db_name))
                self.con_close()
                ok_table = self.create_data_camera_table()
                if(not ok_table): raise IOError
            else:
                raise IOError

            success = True
            return success

        except IOError as e:
            if not db_con:
                logger.error('CreateDB: failed to create new {} database with error: {}').format(db_name,e)
                self.con_close()
                return success

    def create_data_camera_table(self):
        try:
            success = False
            s = Logger()
            logger = s.getLogger()
            con = self.connect2DB()
            curs = con.cursor()
            curs.execute("""CREATE TABLE IF NOT EXISTS data_camera
                 (  sw_vers VARCHAR(5) NOT NULL,
                    cam_id VARCHAR(5),
                    image_date VARCHAR(12) NOT NULL,
                    dont_use BOOLEAN,
                    was_clearsky BOOLEAN,
                    was_rainy BOOLEAN,
                    was_biased BOOLEAN,
                    was_foggy BOOLEAN,
                    had_nimbocum BOOLEAN,
                    UNIQUE KEY (image_date)
                  )"""
                  )
            self.con_close()
            success = True
            return success
        except Exception as e:
            self.con_close()
            logger.error('DB  : Error creating data_camera Table: ' + str(e))
            return success

    def create_new_image_table(self, date):
        try:
            success = False
            s = Logger()
            root_logger = s.getLogger()
            table_name = 'images_' + date
            con = self.connect2DB()
            con.reconnect(attempts=2, delay=0)
            curs = con.cursor()
            sql = """CREATE TABLE IF NOT EXISTS %s
                 (
                    img_nr INTEGER NOT NULL,
                    shots INTEGER NOT NULL,     
                    time VARCHAR(10) NOT NULL,
                    fstop VARCHAR(4),
                    ss VARCHAR(200),
                    exp VARCHAR(50),
                    iso INTEGER,
                    ag DOUBLE,
                    dg DOUBLE,
                    awb_red VARCHAR(200),
                    awb_blue VARCHAR(200),             
                    ldr LONGBLOB,
                    hdr LONGBLOB,
                    UNIQUE KEY (time)
                  )
               """ % table_name

            curs.execute(sql)
            self.con_close()
            success = True
            return success
        except Exception as e:
            self.con_close()
            root_logger.error('create_new_image_table: ' + str(e))
            return success

    def insert_camera_data(self):
        try:
            success = False
            s = Logger()
            logger = s.getLogger()
            con = self.connect2DB()
            curs = con.cursor()
            param_list = 'sw_vers, cam_id, image_date, dont_use, was_clearsky, was_rainy, was_biased, was_foggy, had_nimbocum'
            cameradata = Camera_Data.to_dict()
            values = list(cameradata.values())
            format_strings = ','.join(['%s'] * len(values))

            sql = "INSERT IGNORE INTO data_camera " \
                  "("+ param_list +") " \
                  "VALUES (%s)" % format_strings

            curs.execute(sql,values)
            self.con_close()
            success = True
            return success
        except Exception as e:
            self.con_close()
            logger.error('insert_camera_data: {}' + str(e))
            return success

    def insert_image_data(self, date):
        try:
            success = False
            s = Logger()
            logger = s.getLogger()
            table_name = 'images_' + date
            con = self.connect2DB()
            curs = con.cursor()
            param_list = 'img_nr, time, fstop, ss, exp, iso, ag, dg, awb_red, awb_blue, ldr, hdr'
            imagedata = Image_Data.to_dict()
            values = list(imagedata.values())
            print('values: '.format(values))
            format_strings = ','.join(['%s'] * len(values))

            sql = "INSERT IGNORE INTO {} ".format(table_name) + \
                  "("+ param_list +") " \
                  "VALUES (%s)" % format_strings

            print('SQL: ' + sql)
            curs.execute(sql,values)

            self.con_close()
            success = True
            return success
        except Exception as e:
            self.con_close()
            logger.error('insert_image_data ' + str(e))
            return success

class HDR:


    def data2rgb(self, path_to_img):
        try:
            imgproc = IMAGEPROC()
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
            vb = 1.3  # 87 / 64.  = 1.359375           # neu : 0.56
            vg = 1.80  # 1.                             # neu : 1
            vr = 1.8  # 235 / 128.  = 1.8359375        # neu : 0.95

            # color conversion matrix (from raspi_dng/dcraw)
            # R        g        b
            cvm = np.array(
                [[1.20, -0.30, 0.00],
                 [-0.05, 0.80, 0.14],
                 [0.20, 0.20, 0.7]])

            s = (1232, 1648, 3)
            rgb = np.zeros(s)

            rgb[:, :, 0] = vr * 1023 * (red / 1023.) ** gamma
            rgb[:, :, 1] = vg * 1023 * (green / 1023.) ** gamma
            rgb[:, :, 2] = vb * 1023 * (blue / 1023.) ** gamma

            # rgb = rgb.dot(cvm)

            rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb))

            height, width = rgb.shape[:2]

            img = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_CUBIC)
            # img = img.astype(np.float32)

            # ormalizeImage
            out = np.zeros(img.shape, dtype=np.float)
            min = img.min()
            out = img - min

            # get the max from out after normalizing to 0
            max = out.max()
            out *= (255 / max)

            out = np.uint8(out)

            #out = imgproc.maske_image(np.uint8(out), [1232, 1648, 3], (616, 824), 1000)

            return out


        except Exception as e:
            print('data2rgb: Could not convert data to rgb: ' + str(e))

    def readRawImages(self,mypath, piclist = [0,5,9]):
        try:
            onlyfiles_data = [f for f in listdir(mypath) if isfile(join(mypath, f)) & f.endswith('.data')]
            onlyfiles_jpg  = [f for f in listdir(mypath) if isfile(join(mypath, f)) & f.endswith('.jpg')]
            image_stack = np.empty(len(piclist), dtype=object)
            expos_stack = np.empty(len(piclist), dtype=np.float32)

            # Importing and debayering raw images
            for n in range(0, len(onlyfiles_data)):
                picnumber = ''.join(filter(str.isdigit, onlyfiles_data[n]))
                pos = 0
                for pic in piclist:
                    if str(picnumber) == str(pic):
                        image_stack[pos] = self.data2rgb(join(mypath, onlyfiles_data[n]))
                        print('Pic {}, reading data : {}'.format(str(picnumber), onlyfiles_data[n]))
                    pos +=1

            #Importing exif data from jpg images
            for n in range(0, len(onlyfiles_jpg)):
                picnumber = ''.join(filter(str.isdigit, onlyfiles_jpg[n]))
                pos = 0
                for pic in piclist:
                    if str(picnumber) == str(pic):
                        expos_stack[pos] = self.getEXIF_TAG(join(mypath, onlyfiles_jpg[n]), "EXIF ExposureTime")
                        print('Pic {}, reading exif: {}'.format(str(picnumber), expos_stack[n]))
                    pos +=1

            return image_stack, expos_stack

        except Exception as e:
            print('readRawImages: Could not read *.data files ' + str(e))

    def readImagesAndExpos(self, mypath, piclist=[0,5,9]):
        postproc = IMAGEPROC()
        try:
            onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) & f.endswith('.jpg')]
            image_stack = np.empty(len(piclist), dtype=object)  # Achtung len = onlyfiles für alle bilder
            expos_stack = np.empty(len(piclist), dtype=np.float32)  # Achtung len = onlyfiles für alle bilder
            for n in range(0, len(onlyfiles)):
                picnumber = ''.join(filter(str.isdigit, onlyfiles[n]))
                pos = 0
                for pic in piclist:
                    if str(picnumber) == str(pic):
                        expos_stack[pos] = self.getEXIF_TAG(join(mypath, onlyfiles[n]), "EXIF ExposureTime")
                        image_stack[pos] = postproc.maske_jpg_Image(cv2.imread(join(mypath, onlyfiles[n]), cv2.IMREAD_COLOR))
                        print('Pic {}, reading data from : {}, exif: {}'.format(str(picnumber), onlyfiles[n], expos_stack[n]))
                    pos +=1

            return image_stack, expos_stack

        except Exception as e:
            print('readImagesAndExpos: Could not read images ' + str(e))

    def composeOneHDRimgJpg(self, oneDirsPath, piclist = [0, 5, 9]):
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

    def composeOneHDRimgData(self,oneDirsPath, piclist = [0,5,9]):
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
        global Path_to_sourceDir
        global CAM
        imgproc = IMAGEPROC()
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
        global Path_to_sourceDir
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
            global Path_to_sourceDir
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

class IMAGEPROC:

    def __init__(self, cam_data=None):

        if cam_data == None:
            print('Empty camera data !')

        self.cam_data = cam_data

        CAM = cam_data.cam_id

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

    def image2binary(self, path_to_binary):

        try:
            image_bytes = None
            s = Logger()
            logger = s.getLogger()

            with open(path_to_binary, 'rb') as f:
                image_bytes = f.read()
            f.close()

            return image_bytes

        except Exception as e:
            logger.error('read_image_as_binarry: {}'.format(e))

class Helpers:

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

    def getShutterTimes(self, path):
        try:
            '''
            returns shutter_time in microseconds as np.float32 type
            '''
            s = Logger()
            logger = s.getLogger()
            sw_vers = Camera_Data.sw_vers
            types = ('*.txt', '*.log')
            ss_to_db = []

            for typ in types:
                for file in sorted(glob(os.path.join(path,typ))):
                    logfile = file

            f = open(join(logfile), 'r')
            logfile = f.readlines()

            if sw_vers == 1:
                listOfSS = np.empty(10, dtype=np.float32)

                if os.stat(file).st_size == 0:
                    ss_to_db = '0,0,0,0,0,0,0,0,0,0'
                    logger.error('Empty camstat - file: {}'.format(file))
                    return listOfSS, ss_to_db

                logfile.pop(0)  # remove non relevant lines
                logfile.pop(0)
                logfile.pop(0)
                pos = 0
                for line in logfile:
                    value = line.split("camera shutter speed:", 1)[1].replace('[','').replace(']','')
                    value = value.split('|', 1)[0]
                    value = value.strip()
                    ss_to_db.append(value + ",")
                    value += '/1000000'
                    val_float = np.float32(Fraction(str(value)))
                    listOfSS[pos] = val_float
                    pos +=1

            else :

                listOfSS = np.empty(3, dtype=np.float32)

                if os.stat(file).st_size == 0:
                    ss_to_db = '0,0,0'
                    logger.error('Empty camstat - file: {}'.format(file))
                    return listOfSS, ss_to_db

                if sw_vers == 2:
                    logfile.pop(0)
                if sw_vers == 3:
                    logfile.pop(0)
                    logfile.pop(0)

                pos = 0
                for line in logfile:
                    value = line.split("ss:", 1)[1]
                    value = value.split(',', 1)[0]
                    value = value.strip()
                    ss_to_db.append(value + ",")
                    value += '/1000000'
                    val_float = np.float32(Fraction(str(value)))
                    listOfSS[pos] = val_float
                    pos +=1

            ss_to_db_str = ''.join(ss_to_db)

            return listOfSS, ss_to_db_str.rstrip(',')

        except Exception as e:
            print('Error in getShutterTimes: ' + str(e))

    def getAWB_Gains(self, path):
        try:
            '''
            returns auto white balance values [red,blue]
            '''
            s = Logger()
            logger = s.getLogger()
            types = ('*.txt', '*.log')
            sw_vers = Camera_Data.sw_vers

            for typ in types:
                for file in sorted(glob(os.path.join(path,typ))):
                    logfile = file

            f = open(join(logfile), 'r')
            logfile = f.readlines()

            awb_red_to_db = []
            awb_blue_to_db = []

            if sw_vers == 1:
                listOfAWB = np.empty([10, 2], dtype=np.float32)
                logfile.pop(0)  # remove non relevant lines
                logfile.pop(0)
                logfile.pop(0)
            else:
                listOfAWB = np.empty([3, 2], dtype=np.float32)
                if sw_vers == 2:
                    logfile.pop(0)
                if sw_vers == 3:
                    logfile.pop(0)
                    logfile.pop(0)

            pos = 0
            for line in logfile:
                if sw_vers == 1:
                    temp = line.split("awb (", 1)[1]
                    temp_red = (((temp.split('),', 1)[0]).replace('Fraction(', '')).replace('Fraction(', '')).replace(" ", "")
                    red_gain = temp_red.replace(',', '/')
                    awb_red_to_db.append(red_gain + ',')

                    temp_blue = (((line.split("awb (", 1)[-1]).split('), Fraction')[-1]).split(')),')[0]).replace('(','')
                    blue_gain = temp_blue.replace(' ','').replace(',', '/')
                    awb_blue_to_db.append(blue_gain + ',')

                    red_gain = np.float32(Fraction(str(red_gain)))
                    blue_gain = np.float32(Fraction(str(blue_gain)))
                    listOfAWB[pos] = [red_gain, blue_gain]
                    pos += 1
                else:
                    value = line.split("awb:[", 1)[1]
                    value = value.split('],', 1)[0].replace('Fraction', '').replace('(', '', 1).replace('))', ')').replace(
                        " ", "")
                    red_gain = value.split('),', 1)[0].strip('(').replace(',', '/')
                    awb_red_to_db.append(red_gain + ',')
                    blue_gain = value.split(',(', 1)[1].strip(')').replace(',', '/')
                    awb_blue_to_db.append(blue_gain + ',')
                    red_gain = np.float32(Fraction(str(red_gain)))
                    blue_gain = np.float32(Fraction(str(blue_gain)))
                    listOfAWB[pos] = [red_gain, blue_gain]
                    pos += 1

            awb_red_to_db_str  = ''.join(awb_red_to_db)
            awb_blue_to_db_str = ''.join(awb_blue_to_db)
            return   listOfAWB, awb_red_to_db_str.rstrip(','), awb_blue_to_db_str.rstrip(',')

        except Exception as e:
            print('Error in getAWB_Gains: ' + str(e))

    def getISO(self, path):
        try:
            '''
            returns iso value
            '''
            s = Logger()
            logger = s.getLogger()
            types = ('*.txt', '*.log')
            sw_vers = Camera_Data.sw_vers

            for typ in types:
                for file in sorted(glob(os.path.join(path,typ))):
                    logfile = file

            f = open(join(logfile), 'r')
            logfile = f.readlines()

            iso_to_db = []

            if sw_vers == 1:
                listOfISO = np.empty(10, dtype=np.float32)
                iso_to_db = ['0,0,0,0,0,0,0,0,0,0']
            else:
                listOfISO = np.empty(3, dtype=np.float32)
                if sw_vers == 2:
                    logfile.pop(0)
                if sw_vers == 3:
                    logfile.pop(0)
                    logfile.pop(0)

            pos = 0
            for line in logfile:
                if sw_vers == 1:
                    break
                else:
                    iso = (line.split("iso:")[-1]).split(" exp")[0]
                    iso_to_db.append(iso + ',')
                    listOfISO[pos] = iso
                    pos += 1

            iso_to_db_str = ''.join(iso_to_db)
            return   listOfISO, iso_to_db_str.rstrip(',')

        except Exception as e:
            print('Error in getISO: ' + str(e))

    def getEXP(self, path):
        try:
            '''
            returns exposure value
            '''
            s = Logger()
            logger = s.getLogger()
            types = ('*.txt', '*.log')
            sw_vers = Camera_Data.sw_vers

            for typ in types:
                for file in sorted(glob(os.path.join(path,typ))):
                    logfile = file

            f = open(join(logfile), 'r')
            logfile = f.readlines()

            exp_to_db = []

            if sw_vers == 1:
                listOfEXP = np.empty(10, dtype=np.float32)
                logfile.pop(0)
                logfile.pop(0)
                logfile.pop(0)
            else:
                listOfEXP = np.empty(3, dtype=np.float32)
                if sw_vers == 2:
                    logfile.pop(0)
                if sw_vers == 3:
                    logfile.pop(0)
                    logfile.pop(0)

            pos = 0
            for line in logfile:
                if sw_vers == 1:
                    exp = (line.split("exposure time ")[-1]).split(',')[0]
                    exp_to_db.append(exp + ',')
                    listOfEXP[pos] = exp
                    pos += 1
                else:
                    exp = (line.split("exp:")[-1]).split(',')[0]
                    exp_to_db.append(exp + ',')
                    listOfEXP[pos] = exp
                    pos += 1

            exp_to_db_str = ''.join(exp_to_db)
            return listOfEXP, exp_to_db_str.rstrip(',')

        except Exception as e:
            print('Error in getEXP: ' + str(e))

    def get_ag(self, path):
        try:
            '''
            returns exposure value
            '''
            s = Logger()
            logger = s.getLogger()
            types = ('*.txt', '*.log')
            sw_vers = Camera_Data.sw_vers

            for typ in types:
                for file in sorted(glob(os.path.join(path,typ))):
                    logfile = file

            f = open(join(logfile), 'r')
            logfile = f.readlines()

            ag_to_db = []

            if sw_vers == 1:
                listOfAG = np.empty(10, dtype=np.float32)
                logfile.pop(0)
                logfile.pop(0)
                logfile.pop(0)
            else:
                listOfAG = np.empty(3, dtype=np.float32)
                if sw_vers == 2:
                    logfile.pop(0)
                if sw_vers == 3:
                    logfile.pop(0)
                    logfile.pop(0)

            pos = 0
            for line in logfile:
                if sw_vers == 1:
                    ag = (line.split("ag ")[-1]).split(',')[0]
                    ag_to_db.append(ag + ',')
                    listOfAG[pos] = ag
                    pos += 1
                else:
                    ag = (line.split("ag:")[-1]).split(',')[0]
                    ag_to_db.append(ag + ',')
                    listOfAG[pos] = ag
                    pos += 1

            ag_to_db_str = ''.join(ag_to_db)
            return listOfAG, ag_to_db_str.rstrip(',')

        except Exception as e:
            print('Error in get_ag: ' + str(e))

    def get_dg(self, path):
        try:
            '''
            returns exposure value
            '''
            s = Logger()
            logger = s.getLogger()
            types = ('*.txt', '*.log')
            sw_vers = Camera_Data.sw_vers

            for typ in types:
                for file in sorted(glob(os.path.join(path,typ))):
                    logfile = file

            f = open(join(logfile), 'r')
            logfile = f.readlines()

            dg_to_db = []

            if sw_vers == 1:
                listOfDG = np.empty(10, dtype=np.float32)
                logfile.pop(0)
                logfile.pop(0)
                logfile.pop(0)
            else:
                listOfDG = np.empty(3, dtype=np.float32)
                if sw_vers == 2:
                    logfile.pop(0)
                if sw_vers == 3:
                    logfile.pop(0)
                    logfile.pop(0)

            pos = 0
            for line in logfile:
                if sw_vers == 1:
                    dg = (line.split("dg ")[-1]).split(',')[0]
                    dg_to_db.append(dg + ',')
                    listOfDG[pos] = dg
                    pos += 1
                else:
                    dg = (line.split("dg:")[-1]).split(',')[0]
                    dg_to_db.append(dg + ',')
                    listOfDG[pos] = dg
                    pos += 1

            dg_to_db_str = ''.join(dg_to_db)
            return listOfDG, dg_to_db_str.rstrip(',')

        except Exception as e:
            print('Error in get_dg: ' + str(e))

    def strip_date(self, path):
        try:
            s = Logger()
            logger = s.getLogger()
            H_S = datetime.now().strftime('%M-%S')
            formated_date = '0000-{}'.format(H_S)

            temp = path.rstrip('\\temp')
            temp = (temp.rpartition('\\'))[-1]
            temp = temp.replace('_',' ')
            dateAndTime = temp

            year = dateAndTime[:4]
            month = dateAndTime[4:6]
            day = dateAndTime[6:8]

            check = [year,month,day]

            for item in check:
                if not item or not item.isdigit():
                    logger.error('strip_date: {} could not read date and time  used {} !'.format(path, formated_date))
                    return formated_date

            formated_date = '{}-{}-{}'.format(year, month, day)

            return formated_date
        except IOError as e:
            logger.error('strip_date: could not read date used {} instead !{}').format(formated_date, e)
            return formated_date

    def strip_date_and_time(self, newdatetimestr):
        try:
            s = Logger()
            logger = s.getLogger()
            formated_date = '0000-00-00'
            formated_time = datetime.now().strftime('%H:%M:%S')

            date = (newdatetimestr.rstrip('\\').rpartition('\\')[-1]).rpartition('_')[0]
            time = (newdatetimestr.rstrip('\\').rpartition('\\')[-1]).rpartition('_')[-1]

            year  = date[:4]
            month = date[4:6]
            day   = date[6:8]
            hour  = time[:2]
            min   = time[2:4]
            sec   = time[4:6]

            check = [year,month,day,hour,min,sec]

            for item in check:
                if not item or not item.isdigit():
                    logger.error('strip_date_and_time: could not read date and time  used {} {} instead !{}'.format(
                    formated_date, formated_time))
                    return formated_date, formated_time

            formated_date = '{}-{}-{}'.format(year,month,day)
            formated_time = '{}:{}:{}'.format(hour,min,sec)

            return formated_date, formated_time
        except Exception as e:
            logger.error('strip_date_and_time: could not read date and time  used {} {} instead !{}'.format(
            formated_date,formated_time, e))
            return formated_date, formated_time

    def createNewFolder(self, thispath):
        try:
            if not os.path.exists(thispath):
                os.makedirs(thispath)
                self.setOwnerAndPermission(thispath)

        except Exception as e:
            print('DIR : Could not create new folder: ' + str(e))

    def setOwnerAndPermission(self, pathToFile):
        try:
            uid = pwd.getpwnam('pi').pw_uid
            gid = grp.getgrnam('pi').gr_gid
            os.chown(pathToFile, uid, gid)
            os.chmod(pathToFile, 0o777)
        except Exception as e:
            print('PERM : Could not set permissions for file: ' + str(e))

    def createVideo(self):
        try:

            global Path_to_copy_img5s                                    # path to images
            global Path_to_ffmpeg                                       # path to ffmpeg executable
            fsp = ' -r 10 '                                             # frame per sec images taken
            stnb = '-start_number 0001 '                                # what image to start at
            imgpath = '-i ' + join(Path_to_copy_img5s,'%4d.jpg')+' '    # path to images
            res = '-s 2592x1944 '                                       # output resolution
            outpath = Path_to_copy_img5s + '\sky_video.mp4 '            # output file name
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
            allDirs = []
            img_cnt = 1

            for dirs in sorted(glob(os.path.join(pathToDirectories, "*", ""))):
                if os.path.isdir(dirs):
                    if dirs.rstrip('\\').rpartition('\\')[-1]:
                        allDirs.append(dirs)
                        img_cnt +=1
            return allDirs

        except Exception as e:
            print('getDirectories: Error: ' + str(e))

    def getZipDirs(self,pathToDirectories):
        s = Logger()
        logger = s.getLogger()
        allZipFiles = []
        cnt = 0

        for zipfile in sorted(glob(os.path.join(pathToDirectories, "*.zip"))):
            if os.path.isfile(zipfile):
                allZipFiles.append(zipfile)
                cnt +=1

        if cnt <= 0:
            logger.error('MISSING DATA in: {}'.format(pathToDirectories))

        return allZipFiles

    def readAllImages(self,allDirs):
        try:
            global Path_to_sourceDir
            list_names = []
            list_images = []
            cnt = 1
            print('Converting jpg to opencv, may take some time!')

            t_start = time.time()
            for next_dir in allDirs:
                next_dir += 'raw_img5.jpg'
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

    def copyAndMaskAll_img5(self, list_alldirs):

        try:
            imgproc = IMAGEPROC()
            global Path_to_copy_img5s
            global CAM
            cnt = 1

            if not os.path.exists(Path_to_copy_img5s):
                os.makedirs(Path_to_copy_img5s)

            for next_dir in list_alldirs:
                newimg = join(next_dir,'raw_img5.jpg')
                masked_img = imgproc.maske_jpg_Image(cv2.imread(newimg))
                dateAndTime = (next_dir.rstrip('\\').rpartition('\\')[-1]).replace('_',' ')
                prefix = '{0:04d}'.format(cnt)
                year  = dateAndTime[:4]
                month = dateAndTime[4:6]
                day   = dateAndTime[6:8]
                hour  = dateAndTime[9:11]
                min   = dateAndTime[11:13]
                sec   = dateAndTime[13:15]

                img_txt = imgproc.write2img(masked_img, 'cam ' + CAM, (30, 70))
                img_txt = imgproc.write2img(masked_img,year+" "+month+" "+day,(30,1720))
                img_txt = imgproc.write2img(masked_img, '#: ' + str(cnt), (30,1800))
                img_txt = imgproc.write2img(masked_img, hour+":"+min+":"+sec, (30, 1880))
                new_img_path = join(Path_to_copy_img5s, '{}.jpg'.format(prefix))

                cv2.imwrite(new_img_path,masked_img)
                cnt += 1

            print('Masked {} images and copyied to imgs5.'.format(cnt))

        except Exception as e:
            print('Error in copyAndMaskAll_img5: {}'.format(e))

    def unzipall(self,path_to_extract):
        try:
            s = Logger()
            logger = s.getFileLogger()
            temp_path = join(path_to_extract, 'temp')

            if not os.path.exists(temp_path):
                os.makedirs(temp_path)

            allzipDirs = self.getZipDirs(path_to_extract)

            for dirs in allzipDirs:
                path = dirs.replace('.zip','')
                dirName = (path.rstrip('\\').rpartition('\\')[-1])
                new_temp_path = join(temp_path,dirName)

                if path:
                    zipfilepath = os.path.join(dirs)
                    zf = zipfile.ZipFile(zipfilepath, "r")
                    zf.extractall(os.path.join(new_temp_path))
                    zf.close()

            return temp_path

        except Exception as e:
            logger.error('unzipall: ' + str(e))

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

        except Exception as e:
            print('unzipall: Error: ' + str(e))

    def delUnzipedDir(self, pathtoDir):
        try:
            s = Logger()
            logger = s.getLogger()
            shutil.rmtree(pathtoDir)
            logger.info('deleted {} folder.'.format(pathtoDir))

        except Exception as e:
            logger.error('delUnzipedDir: {}'.format(e))

    def search_list(self, myList, search_str):
        matching = None
        matching = [s for s in myList if search_str in s]
        return matching

    def getAllCamDirectories(self):

        all_cam1_vers = self.getDirectories(Config.camera_1_Directory)
        all_cam2_vers = self.getDirectories(Config.camera_2_Directory)
        all_cam_vers = []

        for i, val in enumerate(all_cam1_vers):
            all_cam_vers.append(val)
            all_cam_vers.append(all_cam2_vers[i])

        return  all_cam_vers

    def load_images2DB(self, path_to_one_dir=None):
        try:
            s = Logger()
            logger = s.getFileLogger()
            f = Logger()
            fileLogger = f.getFileLogger()

            if path_to_one_dir:
                success = self.processOneDay(path_to_one_dir)

                if success:
                    fileLogger.info(path_to_one_dir)

            else:
                allCamDirectorys = self.getAllCamDirectories()
                for path in allCamDirectorys:
                    allDirs = self.getDirectories(path)

                    for raw_cam_dir in allDirs:
                        #print('{}'.format(raw_cam_dir))
                        success = self.processOneDay(raw_cam_dir)
                        if success:
                            fileLogger.info('{}'.format(raw_cam_dir))

        except Exception as e:
            logger.error('load_images2DB: ' + str(e))

    def collectCamData(self, path):
        try:
            s = Logger()
            logger = s.getLogger()
            temp = (path.split('\\'))[-3]
            temp = temp.split('_')

            # sw version and camera ID
            camera_ID = temp[1]
            sw_vers = temp[-1]
            sw_vers = sw_vers.replace('vers','')

            if camera_ID.isdigit(): camera_ID = int(camera_ID)
            if sw_vers.isdigit(): sw_vers = int(sw_vers)
            Camera_Data.cam_id = camera_ID
            Camera_Data.sw_vers = sw_vers

            # extract date
            date = self.strip_date(path)
            Image_Data.image_date = date

        except Exception as e:
            logger.error('collectCamData: ' + str(e))

    def collectImageData(self, path):
        try:
            s = Logger()
            logger = s.getLogger()

            date, time = self.strip_date_and_time(path)
            Image_Data.time = time

            listOfSS, ss_to_db = self.getShutterTimes(path)
            print('ss: {}'.format(ss_to_db))

            listOfAWB, awb_red_to_db, awb_blue_to_db = self.getAWB_Gains(path)
            print('awb_red: {} | awb_blue: {}'.format(awb_red_to_db,awb_blue_to_db))

            listOfISO, iso_to_db = self.getISO(path)
            print('iso: {}'.format(iso_to_db))

            listOfEXP, exp_to_db = self.getEXP(path)
            print('exp: {}'.format(exp_to_db))

            listOfAG, ag_to_db = self.get_dg(path)
            print('ag: {}'.format(ag_to_db))

            listOfDG, dg_to_db = self.get_dg(path)
            print('dg: {}'.format(dg_to_db))


        except Exception as e:
            logger.error('collectImageData: ' + str(e))

    def processOneDay(self, path):
        try:
            s = Logger()
            logger = s.getFileLogger()
            success = False

            #path_to_temp = self.unzipall(path)
            self.collectCamData(path)
            #all_dirs = self.getDirectories(path_to_temp)
            all_dirs = self.getDirectories(path)  # loeschen nur zu testzwecken !

            for dir in all_dirs:
                #print('{}'.format(dir))
                self.collectImageData(dir)


            #shutil.rmtree(path_to_unziped)


            success = True
            return success

        except Exception as e:
          logger.error('processOneDirectory: ' + str(e))
          return success



def main():
    try:
        CFG = {
            'sourceDirectory'   : r'\\HOANAS\HOA_SKYCam',
            'databaseDirectory' : r'\\HOANAS\HOA_SKYCam',
            'camera_1_Directory': r'\\HOANAS\HOA_SKYCam\camera_1',
            'camera_2_Directory': r'\\HOANAS\HOA_SKYCam\camera_2',
        }
        CAM_DATA = {
            'sw_vers': 3,
            'cam_id': '1',
            'image_date': '21-10-2018',
            'dont_use' : 0,
            'was_clearsky':0,
            'was_rainy' :0,
            'was_biased':0,
            'was_foggy':0,
            'had_nimbocum':0
        }
        IMG = {
            'img_nr': 1,
            'time':  '12:23:34',
            'fstop':  '-40',
            'ss' :    1000,
            'exp':    1000,
            'iso':    100,
            'ag':     0.12345,
            'awb_red': 0.12345,
            'awb_blue': 0.12345,
            'ldr': '?',
            'hdr': '?',
        }
        config = Config(CFG)
        h = Helpers()
        s = Logger()
        logger = s.getLogger()

        db = DB_handler()
        db.createDB()

        #h.load_images2DB()
        path1 = r'\\HOANAS\HOA_SKYCam\camera_1\cam_1_vers1\20200505_raw_cam1\temp'     # alte vers 1
        path2 = r'\\HOANAS\HOA_SKYCam\camera_1\cam_1_vers2\20200505_raw_cam1\temp'     # mittlere vers 2
        path3 = r'\\HOANAS\HOA_SKYCam\camera_1\cam_1_vers3\20200505_raw_cam1\temp'     # neuste vers 3

        h.load_images2DB(path2)

        return
        '''
        db = DB_handler()
        db.createDB()

        test_image_path = r'I:\SkY_CAM_IMGS\camera_1\cam_1_vers2\20180403_raw_cam1\20180403_080014'

        CAMERADATA = Camera_Data(CAM_DATA)
        db.insert_camera_data()

        db.create_new_image_table('2018_10_22')
        IMGDATA = Image_Data(IMG)

        imgproc = IMAGEPROC(CAMERADATA)

        IMGDATA.ldr = imgproc.image2binary(join(test_image_path,'raw_img0.jpg'))
        IMGDATA.hdr = imgproc.image2binary(join(test_image_path,'data0_.data'))


        db.insert_image_data('2018_10_22')
        '''

        '''
        global Path_to_sourceDir
        unzipall          = False
        delallzip         = False
        runslideshow      = False
        copyAndMask       = False
        hdr_pics_from_jpg = False
        creat_HDR_Video   = False
        hdr_from_data     = True

        if not os.path.isdir(Path_to_sourceDir):
            print('\nError: Image directory does not exist! -> Aborting.')
            return;

        help = Helpers()
        hdr = HDR()
        allDirs = help.getDirectories(Path_to_sourceDir)

        if unzipall:
            help.unzipall(Path_to_sourceDir)

        if delallzip:
            help.delAllZIP(Path_to_sourceDir)

        if copyAndMask:
            help.copyAndMaskAll_img5(allDirs)
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
        '''


        print('Postprocess.py done')

    except Exception as e:
        logger.error('MAIN: {}'.format(e))


if __name__ == '__main__':
    main()