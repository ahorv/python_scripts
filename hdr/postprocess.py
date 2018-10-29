#!/usr/bin/env python

from __future__ import print_function

import mysql.connector
from mysql.connector import Error
import re
import os
import cv2
import sys
import math
import exifread
from glob import glob
import zipfile
import shutil
import io
from os.path import join
import numpy as np
from fractions import Fraction
import matplotlib.pyplot as plt
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
    date =  '?'
    img_nr = 0
    shots = 0
    time = '?'
    fstop ='?'
    ss = '?'
    exp ='?'
    iso = 0
    ag = '?'
    dg = '?'
    awb_red = '?'
    awb_blue = '?'
    ldr = 0
    hdr = 0
    rmap = 0
    resp = 0

    def __init__(self, state_map={}):
        self.date = state_map.get('date', 0)
        self.img_nr = state_map.get('img_nr',0)
        self.shots = state_map.get('shots', 0)
        self.time = state_map.get('time','?' )
        self.fstop = state_map.get('fstop', '?')
        self.ss = state_map.get('ss', '?')
        self.exp = state_map.get('exp', '?')
        self.iso = state_map.get('iso', 0)
        self.ag = state_map.get('ag', '?')
        self.dg = state_map.get('dg', '?')
        self.awb_red = state_map.get('awb_red', '?')
        self.awb_blue = state_map.get('awb_blue', '?')
        self.ldr = state_map.get('ldr', 0)
        self.hdr = state_map.get('hdr', 0)
        self.rmap = state_map.get('rmap', 0)
        self.resp = state_map.get('resp', 0)

        Image_Data.date = self.date
        Image_Data.img_nr = self.img_nr
        Image_Data.shots = self.shots
        Image_Data.time = self.time
        Image_Data.fstop = self.fstop
        Image_Data.ss = self.ss
        Image_Data.exp = self.exp
        Image_Data.iso = self.iso
        Image_Data.ag = self.ag
        Image_Data.ag = self.dg
        Image_Data.awb_red = self.awb_red
        Image_Data.awb_blue = self.awb_blue
        Image_Data.ldr = self.ldr
        Image_Data.hdr = self.hdr
        Image_Data.rmap = self.rmap
        Image_Data.resp = self.resp

    def to_dict(self):
        return {
            'date':   Image_Data.date,
            'img_nr': Image_Data.img_nr,
            'shots': Image_Data.shots,
            'time':   Image_Data.time,
            'fstop':  Image_Data.fstop,
            'ss' :    Image_Data.ss,
            'exp':    Image_Data.exp,
            'iso':    Image_Data.iso,
            'ag':     Image_Data.ag,
            'dg':     Image_Data.dg,
            'awb_red' : Image_Data.awb_red,
            'awb_blue': Image_Data.awb_blue,
            'ldr': Image_Data.ldr,
            'hdr': Image_Data.hdr,
            'rmap': Image_Data.rmap,
            'resp': Image_Data.resp,
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
    NAS_IP = '?'
    sourceDirectory = '?'
    camera_1_Directory = '?'
    camera_2_Directory = '?'
    databaseName = '?'
    databaseDirectory = '?'

    def __init__(self, state_map={}):

        self.NAS_IP = state_map.get('NAS_IP','?')
        self.sourceDirectory = state_map.get('sourceDirectory', '?')
        self.camera_1_Directory = state_map.get('camera_1_Directory', '?')
        self.camera_2_Directory = state_map.get('camera_2_Directory', '?')
        self.databaseName = state_map.get('databaseName','sky_db')
        self.databaseDirectory = state_map.get('databaseDirectory', '?')

        Config.NAS_IP = self.NAS_IP
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
            nas_ip = Config.NAS_IP
            self.connection = mysql.connector.connect(
                            host = nas_ip,
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
            nas_ip = Config.NAS_IP
            self.connection = mysql.connector.connect(
                host=nas_ip,
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

    def create_new_image_table(self):
        try:
            success = False
            s = Logger()
            root_logger = s.getLogger()
            table_name = 'images_' + (Image_Data.date).replace('-','_')
            con = self.connect2DB()
            con.reconnect(attempts=2, delay=0)
            curs = con.cursor()
            sql = """CREATE TABLE IF NOT EXISTS %s
                 (
                    img_nr INTEGER NOT NULL,
                    shots INTEGER NOT NULL,     
                    time VARCHAR(10) NOT NULL,
                    fstop VARCHAR(25),
                    ss VARCHAR(200),
                    exp VARCHAR(100),
                    iso INTEGER,
                    ag DOUBLE,
                    dg DOUBLE,
                    awb_red VARCHAR(200),
                    awb_blue VARCHAR(200),             
                    ldr LONGBLOB,
                    hdr LONGBLOB,
                    rmap LONGBLOB,
                    resp LONGBLOB,
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
            cameradata = Camera_Data.to_dict(None)
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

    def insert_image_data(self):
        try:
            success = False
            s = Logger()
            logger = s.getLogger()
            table_name = 'images_' + (Image_Data.date).replace('-','_')
            con = self.connect2DB()
            curs = con.cursor()
            param_list = 'img_nr, shots, time, fstop, ss, exp, iso, ag, dg, awb_red, awb_blue, ldr, hdr, rmap, resp'
            imagedata = Image_Data.to_dict(None)
            del imagedata['date']
            values = list(imagedata.values())
            format_strings = ','.join(['%s'] * len(values))

            sql = "INSERT IGNORE INTO {} ".format(table_name) + \
                  "("+ param_list +") " \
                  "VALUES (%s)" % format_strings

            curs.execute(sql,values)

            self.con_close()
            success = True
            return success
        except Exception as e:
            self.con_close()
            logger.error('insert_image_data ' + str(e))
            return success

class HDR:
    def make_hdr(self, path, listOfSS, img_type = 'jpg'):
        try:
            h = Helpers()
            s = Logger()
            logger = s.getLogger()
            success = False
            img_dir = []
            type = ''

            if img_type is 'jpg':
                type = '*.jpg'
            else:
                type = '*.data'

            # Load all images
            for imgFile in sorted(glob(os.path.join(path, type))):
                if os.path.isfile(imgFile):
                    img_dir.append(imgFile)

            # Sort image order to match with listOfSS
            img_dir.sort(key=h.byInteger_keys)

            # Loading images channel - wise
            img_list_b = self.load_exposures(img_dir, 0)
            img_list_g = self.load_exposures(img_dir, 1)
            img_list_r = self.load_exposures(img_dir, 2)

            # Solving response curves
            gb, _ = self.hdr_debvec(img_list_b, listOfSS)
            gg, _ = self.hdr_debvec(img_list_g, listOfSS)
            gr, _ = self.hdr_debvec(img_list_r, listOfSS)

            if img_type is 'jpg':
                hdr = self.construct_hdr([img_list_b, img_list_g, img_list_r], [gb, gg, gr], listOfSS)

                normalize = lambda zi: (zi - zi.min() / zi.max() - zi.min())
                z_disp = normalize(np.log(hdr))
                plt.figure(figsize=(12, 8))
                plt.imshow(z_disp / z_disp.max())
                plt.axis('off')
                plt.show()


                Image_Data.ldr = self.hdr_to_blob(hdr)

            if img_type is 'data':
                # Create and plot response curve
                plt.figure(figsize=(10, 10))
                plt.plot(gr, range(256), 'rx')
                plt.plot(gg, range(256), 'gx')
                plt.plot(gb, range(256), 'bx')
                plt.ylabel('pixel value Z')
                plt.xlabel('log exposure X')
                fig = plt.gcf()
                respc = io.BytesIO()
                fig.savefig(respc, format='jpg')
                respc.seek(0)
                resp_blob = respc.read()
                Image_Data.resp = resp_blob

                # make the HDR
                hdr = self.construct_hdr([img_list_b, img_list_g, img_list_r], [gb, gg, gr], listOfSS)
                Image_Data.hdr = self.hdr_to_blob(hdr)

                # Create radiance map
                plt.figure(figsize=(12, 8))
                plt.imshow(np.log2(cv2.cvtColor(hdr, cv2.COLOR_BGR2GRAY)), cmap='jet')
                plt.colorbar()
                fig = plt.gcf()
                rmap = io.BytesIO()
                fig.savefig(rmap, format='jpg')
                rmap.seek(0)
                rmap_blob = rmap.read()
                Image_Data.rmap = rmap_blob

            success = True
            return success

        except Exception as e:
            logger.error('make_hdr ' + str(e))
            return success

    def demosaic1(self, mosaic, awb_gains=None):
        '''
        nedded by make_hdr
        :param mosaic:
        :param awb_gains:
        :return:
        '''
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

    def toRGB_1(self, data):
        '''
        nedded by make_hdr
        Belongs to deraw1
        :param data:
        :return:
        '''
        image = data // 256  # reduce dynamic range to 8 bpp
        image = np.clip(image, 0, 255).astype(np.uint8)

        return image

    def read_data(self, path_to_image):
        '''
        nedded by make_hdr
        :param path_to_image:
        :return:
        '''
        data = np.fromfile(path_to_image, dtype='uint16')
        data = data.reshape([2464, 3296])
        raw = self.demosaic1(data)

        return raw.astype('uint16')

    def load_exposures(self, dir_list, channel=0):
        '''
        nedded by make_hdr
        Reads either of jpg or raw depending on file extension
        :param source_dir:
        :param channel:
        :return:
        '''
        img_list = []
        if dir_list[0].endswith('.data'):
            img_list = [self.toRGB_1(self.read_data(file)) for file in dir_list]
            img_list = [img[:, :, channel] for img in img_list]

        if dir_list[0].endswith('.jpg'):
            img_list = [cv2.imread(file, 1)for file in dir_list]
            img_list = [img[:, :, channel] for img in img_list]

        return img_list

    def median_threshold_bitmap_alignment(self, img_list):
        '''
         MTB implementation
         nedded by make_hdr
        :return:
        '''
        median = [np.median(img) for img in img_list]
        binary_thres_img = [cv2.threshold(img_list[i], median[i], 255, cv2.THRESH_BINARY)[1] for i in
                            range(len(img_list))]
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

    def hdr_debvec(self, img_list, exposure_times):
        '''
        needed by make_hdr
        :param exposure_times:
        :return:
        '''
        B = [math.log(e, 2) for e in exposure_times]
        l = 50  # lambda sets amount of smoothness
        w = [z if z <= 0.5 * 255 else 255 - z for z in range(256)]

        small_img = [cv2.resize(img, (10, 10)) for img in img_list]
        Z = [img.flatten() for img in small_img]

        return self.response_curve_solver(Z, B, l, w)

    def response_curve_solver(self, Z, B, l, w):
        '''
        Implementation of paper's Equation(3) with weight
         needed by make_hdr
        :param B:
        :param l:
        :param w:
        :return:
        '''
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
        x = np.linalg.lstsq(A, b)[0]
        g = x[:256]
        lE = x[256:]

        return g, lE

    def construct_radiance_map(self, g, Z, ln_t, w):
        '''
        Implementation of paper's Equation(6)
        needed by make_hdr
        :param Z:
        :param ln_t:
        :param w:
        :return:
        '''
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

    def construct_hdr(self, img_list, response_curve, exposure_times):
        '''
        Construct radiance map for each channels
        needed by make_hdr
        :param img_list:
        :param response_curve:
        :param exposure_times:
        :return:
        '''
        img_size = img_list[0][0].shape
        w = [z if z <= 0.5 * 255 else 255 - z for z in range(256)]
        ln_t = np.log2(exposure_times)

        vfunc = np.vectorize(lambda x: math.exp(x))
        hdr = np.zeros((img_size[0], img_size[1], 3), 'float32')

        # construct radiance map for BGR channels
        for i in range(3):
            Z = [img.flatten().tolist() for img in img_list[i]]
            E = self.construct_radiance_map(response_curve[i], Z, ln_t, w)
            # Exponational each channels and reshape to 2D-matrix
            hdr[..., i] = np.reshape(vfunc(E), img_size)

        return hdr

    def hdr_to_blob(self, hdr):
        '''
        Concatenates HDR image and header to BLBO
        Code based on https://gist.github.com/edouardp/3089602
        needed by make_hdr
        :param filename:
        :return:
        '''
        try:
            s = Logger()
            logger = s.getLogger()
            image = np.zeros((hdr.shape[0], hdr.shape[1], 3), 'float32')
            image[..., 0] = hdr[..., 2]
            image[..., 1] = hdr[..., 1]
            image[..., 2] = hdr[..., 0]

            title = (b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
            header = '-Y {0} +X {1}\n'.format(image.shape[0], image.shape[1])
            header = (bytes(header, encoding='utf-8'))

            brightest = np.maximum(np.maximum(image[..., 0], image[..., 1]), image[..., 2])
            mantissa = np.zeros_like(brightest)
            exponent = np.zeros_like(brightest)
            np.frexp(brightest, mantissa, exponent)
            scaled_mantissa = mantissa * 256.0 / brightest
            rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
            rgbe[..., 0:3] = np.around(image[..., 0:3] * scaled_mantissa[..., None])
            rgbe[..., 3] = np.around(exponent + 128)

            _rgbe = rgbe.flatten()

            byte_str = title + header + _rgbe.tobytes()
            blob_b = io.BytesIO(byte_str)
            blob_b.seek(0)
            blob = blob_b.read()

            # filename = r'\\HOANAS\HOA_SKYCam\camera_1\cam_1_vers3\20200505_raw_cam1\temp\hdr.hdr'
            # f = open(filename, 'wb')
            # f.write(blob)
            # f.close()

            return blob

        except Exception as e:
            logger.error('hdr_to_blob ' + str(e))
            return None

    def save_hdr(self, hdr, filename):
        '''
        LOESCHEN
        :param filename:
        :return:
        '''
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

    def get_int(self, text):
        return int(text) if text.isdigit() else text

    def byInteger_keys(self, text):
        return [self.get_int(c) for c in re.split('(\d+)', text)]

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

    def getNumOfShots(self):
        sw_vers = Camera_Data.sw_vers
        shots = 0

        if sw_vers == 1:
            shots = 10
        else:
            shots = 3

        return shots

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
                    logger.error('Empty camstat.log - file: {}'.format(file))
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

                    red_gain_f = np.float32(Fraction(str(red_gain)))
                    blue_gain_f = np.float32(Fraction(str(blue_gain)))
                    listOfAWB[pos] = [red_gain_f, blue_gain_f]
                    pos += 1
                else:
                    value = line.split("awb:[", 1)[1]
                    value = value.split('],', 1)[0].replace('Fraction', '').replace('(', '', 1).replace('))', ')').replace(
                        " ", "")
                    red_gain = value.split('),', 1)[0].strip('(').replace(',', '/')
                    awb_red_to_db.append(red_gain + ',')
                    blue_gain = value.split(',(', 1)[1].strip(')').replace(',', '/')
                    awb_blue_to_db.append(blue_gain + ',')

                    red_gain_f = np.float32(Fraction(str(red_gain)))
                    blue_gain_f = np.float32(Fraction(str(blue_gain)))
                    listOfAWB[pos] = [red_gain_f, blue_gain_f]
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
                    exp_float = np.float32(Fraction(str(exp)))
                    listOfEXP[pos] = exp_float
                    pos += 1
                else:
                    exp = (line.split("exp:")[-1]).split(',')[0]
                    exp_to_db.append(exp + ',')
                    exp_float = np.float32(Fraction(str(exp)))
                    listOfEXP[pos] = exp_float
                    pos += 1

            exp_to_db_str = ''.join(exp_to_db)
            return listOfEXP, exp_to_db_str.rstrip(',')

        except Exception as e:
            print('Error in getEXP: ' + str(e))

    def getAG(self, path):
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
                    ag_float = np.float32(Fraction(str(ag)))
                    listOfAG[pos] = ag_float
                    pos += 1

            ag_to_db_str = ''.join(ag_to_db)
            return listOfAG, ag_to_db_str.rstrip(',')

        except Exception as e:
            print('Error in get_ag: ' + str(e))

    def getDG(self, path):
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
                    dg_float = np.float32(Fraction(str(dg)))
                    listOfDG[pos] = dg_float
                    pos += 1
                else:
                    dg = (line.split("dg:")[-1]).split(',')[0]
                    dg_to_db.append(dg + ',')
                    dg_float = np.float32(Fraction(str(dg)))
                    listOfDG[pos] = dg_float
                    pos += 1

            dg_to_db_str = ''.join(dg_to_db)
            return listOfDG, dg_to_db_str.rstrip(',')

        except Exception as e:
            print('Error in get_dg: ' + str(e))

    def getFstops(self, path):
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

            fStop_to_db = []

            if sw_vers == 1:
                listOfFSTOP = np.empty(10, dtype=np.float32)
                fStop_to_db = ['0,0,0,0,0,0,0,0,0,0']
            else:
                listOfFSTOP = np.empty(3, dtype=np.float32)
                if sw_vers == 2:
                    fStop_to_db = ['0,0,0']
                if sw_vers == 3:
                    logfile.pop(0)
                    logfile.pop(0)

            pos = 0
            for line in logfile:
               if sw_vers == 1 or sw_vers == 2:
                    break
               else:
                    fstop = (line.split("F Stop:")[-1]).split(',')[0]
                    fStop_to_db.append(fstop + ',')
                    fStop_float = np.float32(Fraction(str(fstop)))
                    listOfFSTOP[pos] = fStop_float
                    pos += 1

            fStop_to_db_str = ''.join(fStop_to_db)
            return listOfFSTOP, fStop_to_db_str.rstrip(',')

        except Exception as e:
            print('Error in getFstops: ' + str(e))

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

            # write only one day to database
            if path_to_one_dir:
                success = self.processOneDay(path_to_one_dir)
                if success:
                    fileLogger.info(path_to_one_dir)

            # write everything to database
            else:
                allCamDirectorys = self.getAllCamDirectories()
                for path in allCamDirectorys:
                    allDirs = self.getDirectories(path)

                    for raw_cam_dir in allDirs:
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
            Camera_Data.image_date = date


        except Exception as e:
            logger.error('collectCamData: ' + str(e))

    def collectImageData(self, path):
        try:
            s = Logger()
            logger = s.getLogger()
            hdr = HDR()
            success = False

            date, time = self.strip_date_and_time(path)
            nuberOfshots = self.getNumOfShots()
            listOfFstops, fstops_to_db = self.getFstops(path)
            listOfSS,  ss_to_db = self.getShutterTimes(path)
            listOfAWB, awb_red_to_db,awb_blue_to_db = self.getAWB_Gains(path)
            listOfISO, iso_to_db = self.getISO(path)
            listOfEXP, exp_to_db = self.getEXP(path)
            listOfAG,  ag_to_db = self.getAG(path)
            listOfDG,  dg_to_db = self.getDG(path)

            Image_Data.date = date
            Image_Data.shots = nuberOfshots
            Image_Data.time = time
            Image_Data.fstop = fstops_to_db
            Image_Data.ss = ss_to_db
            Image_Data.exp = exp_to_db
            Image_Data.iso = iso_to_db
            Image_Data.ag = ag_to_db
            Image_Data.dg = dg_to_db
            Image_Data.awb_red = awb_red_to_db
            Image_Data.awb_blue = awb_blue_to_db

            hdr.make_hdr(path, listOfSS ,'jpg')
            #hdr.make_hdr(path, listOfSS, 'data')

            success = True
            return success

        except Exception as e:
            logger.error('collectImageData: ' + str(e))
            return success

    def writeImageData2DB(dir):
        success = False

        db = DB_handler()
        success = db.create_new_image_table()
        success = db.insert_image_data()

        if success:
            success = db.insert_camera_data()

        return success

    def processOneDay(self, path):
        try:
            s = Logger()
            logger = s.getFileLogger()
            success = False
            img_nr = 0

            #path_to_temp = self.unzipall(path)
            self.collectCamData(path)
            #all_dirs = self.getDirectories(path_to_temp) # einkommentieren !!
            all_dirs = self.getDirectories(path)  # loeschen nur zu testzwecken !

            for dir in all_dirs:
                img_nr += 1
                Image_Data.img_nr = img_nr
                success = self.collectImageData(dir)
                if success:
                    success = self.writeImageData2DB()

                #shutil.rmtree(path_to_unziped)

            return success

        except Exception as e:
          logger.error('processOneDirectory: ' + str(e))
          return success

def main():
    try:
        CFG = {
            'NAS_IP'            : r'192.168.2.115',
            'sourceDirectory'   : r'\\IHLNAS05\SkyCam_FTP',
            'databaseDirectory' : r'\\IHLNAS05\SkyCam_FTP',
            'camera_1_Directory': r'\\IHLNAS05\SkyCam_FTP\camera_1',
            'camera_2_Directory': r'\\IHLNAS05\SkyCam_FTP\camera_2',
        }

        config = Config(CFG)
        h = Helpers()
        s = Logger()
        logger = s.getLogger()

        db = DB_handler()
        db.createDB()

        #h.load_images2DB()
        path1 = r'\\IHLNAS05\SkyCam_FTP\camera_1\cam_1_vers1\20200505_raw_cam1\temp'     # alte vers 1
        path2 = r'\\IHLNAS05\SkyCam_FTP\camera_1\cam_1_vers2\20200505_raw_cam1\temp'     # mittlere vers 2
        path3 = r'\\IHLNAS05\SkyCam_FTP\camera_1\cam_1_vers3\20200505_raw_cam1\temp'     # neuste vers 3

        h.load_images2DB(path3)

        logger.info('All files processed.')

    except Exception as e:
        logger.error('MAIN: {}'.format(e))


if __name__ == '__main__':
    main()