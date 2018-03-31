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
import shutil
import tempfile
import logging
import logging.handlers
from datetime import datetime, timedelta
import numpy as np

if sys.platform == "linux":
    import pwd
    import grp
    import stat
    import fcntl
    import picamera

######################################################################
## Hoa: 31.03.2018 Version 2 : raw2.py
######################################################################
# This class takes 10 consecutive images with increasing shutter times.
# Pictures are in raw bayer format. In addition a jpg as reference is
# taken in addition.
# Aim is to merge a sequence of images to a HDR image.
# Runtime for a sequence of 3 images is about 21 sec
# Shutter times: img0: 85, img5: 595, img9: 992 micro secs
#
# New /Changes:
# ----------------------------------------------------------------------
#
# 10.11.2017 : Added new logging
# 25.03.2018 : new version only 3 shutter times
# 31.03.2018 : added single instance functionality by a lock file
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

    def ensure_single_instance_of_app(self):
        app_name = 'raw2'  # app name to be monitored

        if sys.platform == "linux":
            s = Logger()
            log = s.getLogger()

            # Establish lock file settings
            lf_name = '.{}.lock'.format(app_name)
            lf_path = os.path.join(tempfile.gettempdir(), lf_name)
            lf_flags = os.O_WRONLY | os.O_CREAT
            lf_mode = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH  # This is 0o222, i.e. 146

            # Create lock file
            # Regarding umask, see https://stackoverflow.com/a/15015748/832230
            umask_original = os.umask(0)
            try:
                lf_fd = os.open(lf_path, lf_flags, lf_mode)
            finally:
                os.umask(umask_original)

            # Try locking the file
            try:
                fcntl.lockf(lf_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except IOError as e:
                msg = ('{} may already be running. Only one instance of it '
                       'allowed.'
                       ).format('raw2')
                log.info(' LOCK: ' + str(msg))
                exit()

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

    def getRunTime(self, start_time,end_time):

        formated = '%H:%M:%S'
        tdelta = datetime.strptime(end_time, formated) - datetime.strptime(start_time, formated)

        td = format(tdelta)
        print('tdelta: ' + td)
        h, m, s = [int(i) for i in td.split(':')]

        return h,m,s

def main():
    try:
        s = Logger()
        log = s.getLogger()
        helper = Helpers()
        helper.ensure_single_instance_of_app()
        helper.setPathAndNewFolders()
        usedspace = helper.disk_stat()

        if usedspace > 80:
            raise RuntimeError('WARNING: Not enough free space on SD Card!')
            return

        cam = Rawcamera()

        t_start = '19:00:00'  # Start time to capture images
        t_end   = '20:15:00'  # Stop time ends script

        h,m,s, = helper.getRunnTime(t_start,t_end)

        # Sets the duration of time lapse run
        runtime = datetime.now() + timedelta(days=0) + timedelta(hours=h) + \
                  timedelta(minutes=m) + timedelta(seconds=s)

        while (True):

            time.sleep(1)
            time_now = time.strftime("%H:%M:%S")

            if t_start == time_now:

                while runtime > datetime.now():
                    cam.takepictures()

                log.info(' TIME LAPS STOPPED: {} '.format(time.strftime("%H:%M:%S")))
                sys.exit()

    except Exception as e:
        log.error(' MAIN: Error in main: ' + str(e))


if __name__ == '__main__':
    main()

