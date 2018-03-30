#!/usr/bin/cd python

from __future__ import (
    unicode_literals,
    absolute_import,
    print_function,
    division,
)

import sys
import io
import os
import time
import pwd
import grp
import picamera
import zipfile
import shutil
import logging
import logging.handlers
from datetime import datetime
import numpy as np

######################################################################
## Hoa: 25.03.2018 Version 5 : raw.py
######################################################################
# This class takes 10 consecutive images with increasing shutter times.
# Pictures are in raw bayer format. In addition a jpg as reference is
# taken in addition.
# Aim is to merge a sequence of images to a HDR image.
# Runtime for a sequence of 3 images is about 21 sec
# Shutter times: img0: 85, img5: 595, img9: 992 microsecs
#
# New /Changes:
# ----------------------------------------------------------------------
#
# 10.11.2017 : Added new logging
# 25.03.2018 : new version only 3 shutter times
#
######################################################################

global RAWDATAPATH
global LOGFILEPATH
global SUBDIRPATH
global TSTAMP

class Logger:
    def getLogger(self):

        try:
            global LOGFILEPATH

            # configure log formatter
            # logFormatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
            logFormatter = logging.Formatter('%(message)s')

            # configure file handler
            fileHandler = logging.FileHandler(LOGFILEPATH)
            fileHandler.setFormatter(logFormatter)

            # configure stream handler
            consoleHandler = logging.StreamHandler()
            consoleHandler.setFormatter(logFormatter)

            # get the logger instance
            self.logger = logging.getLogger(__name__)

            # set rotating filehandler
            handler = logging.handlers.RotatingFileHandler(LOGFILEPATH, encoding='utf8',
                                                           maxBytes=1024 * 10000, backupCount=1)

            # set the logging level
            self.logger.setLevel(logging.INFO)

            if not len(self.logger.handlers):
                self.logger.addHandler(fileHandler)
                self.logger.addHandler(consoleHandler)
                # self.logger.addHandler(handler)
            helper = Helpers()
            helper.setOwnerAndPermission(LOGFILEPATH)
            return self.logger

        except IOError as e:
            print('Error logger:' + str(e))

class Rawcamera:
    def takepictures(self):
        try:

            global SUBDIRPATH

            s = Logger()
            log = s.getLogger()

            stream = io.BytesIO()
            with picamera.PiCamera() as camera:
                camera.resolution = (2592, 1944)
                # shutter speed is limited by framerate!
                camera.framerate = 1
                camera.exposure_mode = 'off'
                camera.awb_mode = 'auto'
                camera.iso = 0
                shutter_speed = 100
                loopstart = time.time()
                # for i0 in range(10):
                for i0 in [0, 5, 9]:
                    # set shutter speed
                    loopstart_tot = time.time()
                    camera.shutter_speed = (i0 + 1) * shutter_speed
                    fileName = 'raw_img%s.jpg' % str(i0)

                    # Capture the image, without the Bayer data to file
                    loopstartjpg = time.time()
                    camera.capture(SUBDIRPATH + "/" + fileName, format='jpeg', bayer=False)
                    loopendjpg = time.time()

                    # Capture the image, including the Bayer data to stream
                    loopstartraw = time.time()
                    camera.capture(stream, format='jpeg', bayer=True)
                    loopendraw = time.time()

                    data = stream.getvalue()[-10270208:]
                    data = data[32768:4128 * 2480 + 32768]
                    data = np.fromstring(data, dtype=np.uint8)
                    data = data.reshape((2480, 4128))[:2464, :4120]
                    data = data.astype(np.uint16) << 2
                    for byte in range(4):
                        data[:, byte::5] |= ((data[:, 4::5] >> ((4 - byte) * 2)) & 0b11)

                    data = np.delete(data, np.s_[4::5], 1)

                    loopend_tot = time.time()
                    # camera settings
                    cam_stats = dict(
                        ss=camera.shutter_speed,
                        iso=camera.iso,
                        exp=camera.exposure_speed,
                        ag=camera.analog_gain,
                        dg=camera.digital_gain,
                        awb=camera.awb_gains,
                        br=camera.brightness,
                        ct=camera.contrast,
                    )
                    t_stats = dict(
                        t_jpg='{0:.2f}'.format(loopendjpg - loopstartjpg),
                        t_raw='{0:.2f}'.format(loopendraw - loopstartraw),
                        t_tot='{0:.2f}'.format(loopend_tot - loopstart_tot),
                    )

                    # Write camera settings to log file
                    logdata = '{} Run :'.format(str(i0))
                    logdata = logdata + ' [ss:{ss}, iso:{iso} exp:{exp}, ag:{ag}, dg:{dg}, awb:[{awb}], br:{br}, ct:{ct}]'.format(
                        **cam_stats)
                    logdata = logdata + ' || timing: [t_jpg:{t_jpg}, t_raw:{t_raw}, t_tot:{t_tot}]'.format(**t_stats)

                    log.info(logdata)

                    # Finally save raw (16bit data) having a size 3296 x 2464
                    datafileName = 'data%d_%s.data' % (i0, str(''))

                    with open(SUBDIRPATH + "/" + datafileName, 'wb') as g:
                        data.tofile(g)

        except Exception as e:
            camera.close()
            print('Error in takepicture: ' + str(e))


class Helpers:
    def setPathAndNewFolders(self):
        try:
            global RAWDATAPATH
            global LOGFILEPATH
            global SUBDIRPATH
            global TSTAMP

            TSTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

            RAWDATAPATH = os.path.join('/home', 'pi', 'python_scripts', 'raw', 'raw_data')
            self.createNewFolder(RAWDATAPATH)
            SUBDIRPATH = os.path.join(RAWDATAPATH, TSTAMP)
            self.createNewFolder(SUBDIRPATH)
            LOGFILEPATH = os.path.join(SUBDIRPATH, TSTAMP + '_raw' + '.log')

        except IOError as e:
            print('PATH: Could not set path and folder: ' + str(e))

    def createNewFolder(self, thispath):
        try:
            if not os.path.exists(thispath):
                os.makedirs(thispath)
                self.setOwnerAndPermission(thispath)

        except IOError as e:
            print('DIR : Could not create new folder: ' + str(e))

    def setOwnerAndPermission(self, pathToFile):
        try:
            uid = pwd.getpwnam('pi').pw_uid
            gid = grp.getgrnam('pi').gr_gid
            os.chown(pathToFile, uid, gid)
            os.chmod(pathToFile, 0o777)
        except IOError as e:
            print('PERM : Could not set permissions for file: ' + str(e))

    def disk_stat(self):

        try:
            global disc_stat_once
            s = Logger()
            log = s.getLogger()

            total, used, free = shutil.disk_usage("/usr")
            total_space = total / 1073741824
            used_space = used / 1073741824
            free_space = free / 1073741824

            disc_status = 'Disc Size:%iGB\tSpace used:%iGB\tFree space:%iGB ' % (total_space, used_space, free_space)
            log.info(disc_status)
            percent = used_space / (total_space / 100)

            return percent

        except IOError as e:
            print('DISKSTAT :  ' + str(e))


def main():
    try:

        while (True):
            helper = Helpers()
            helper.setPathAndNewFolders()

            usedspace = helper.disk_stat()
            if usedspace > 80:
                raise RuntimeError('WARNING: Not enough free space on SD Card!')
                return

            cam = Rawcamera()
            cam.takepictures()

    except Exception as e:
        print('Error in Main: ' + str(e))


if __name__ == '__main__':
    main()

