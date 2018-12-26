#!/usr/bin/env python

from time import sleep
import sys
import io
import os
import cv2
import picamera
import numpy as np
from os.path import join
import logging
import logging.handlers
from datetime import datetime

if sys.platform == "linux":
    import pwd
    import grp
    import stat

# Raspicamera settings: https://www.raspberrypi.org/documentation/usage/camera/python/README.md

######################################################################
## Hoa: 24.12.2017 Version 1 : crf.py
######################################################################
# Takes a few raw-pictures to assess the linearity of the Raspberry-Pi
# camera module v2.
#
#
# New /Changes:
# ----------------------------------------------------------------------
#
# 24.12.2017 : First implemented
######################################################################

SS_LIST = np.linspace(0., 33000., 659)
Path = join('/home', 'pi', 'python_scripts', 'crf')


class Logger:
    def __init__(self):
        self.logger = None

    def getLogger(self, newLogPath=None):

        try:
            LOGFILEPATH = os.path.join(Path, 'crf.csv')
            logFormatter = logging.Formatter('%(message)s')
            fileHandler = logging.FileHandler(LOGFILEPATH)
            name = 'rootlogger'

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
            print('Error logger:' + str(e))

class Helpers:

    def avgbrightness(im):
        """
        Find the average brightness of the provided image.

        Args:
          im: A opencv image.
          config: Camera_config object.  Defaults to self.config.
        Returns:
          Average brightness of the image.
        """
        try:
            aa = im.copy()
            imRes = cv2.resize(aa, (128, 96), interpolation=cv2.INTER_AREA)
            mask = imRes.copy()
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask[np.where((mask != [0]).all(axis=1))] = [255]
            mask = mask.astype(np.uint8)
            aa = cv2.cvtColor(imRes, cv2.COLOR_BGR2GRAY)

            pixels = (aa.shape[0] * aa.shape[1])
            h = cv2.calcHist([aa], [0], mask, [256], [0, 256])
            mu0 = 1.0 * sum([i * h[i] for i in range(len(h))]) / pixels
            return round(mu0[0], 2)

        except IOError as e:
            print('avgbrightness : error: ' + str(e))

    def createNewFolder(self, thispath):
        try:
            if not os.path.exists(thispath):
                os.makedirs(thispath)
                self.setOwnerAndPermission(thispath)

        except IOError as e:
            print('DIR : Could not create new folder: ' + str(e))

    def setOwnerAndPermission(pathToFile):
        try:
            uid = pwd.getpwnam('pi').pw_uid
            gid = grp.getgrnam('pi').gr_gid
            os.chown(pathToFile, uid, gid)
            os.chmod(pathToFile, 0o777)
        except IOError as e:
            print('PERM : Could not set permissions for file: ' + str(e))

class ImgProc:
    def get_r_g_b_channels(self,raw):
        try:
            black = raw.min()
            saturation = raw.max()

            uint14_max = 2 ** 14 - 1
            raw -= black  # black subtraction
            raw *= int(uint14_max / (saturation - black))
            _raw = np.clip(raw, 0, uint14_max)  # clip to range
            _raw = _raw.reshape([2464, 3296])
            _raw = _raw.astype('float')
            _raw = np.clip(_raw, 0, uint14_max)  # clip to range
            _raw *= 2 ** 2

            # demosaic
            p1 = _raw[0::2, 1::2]  # Blue
            p2 = _raw[0::2, 0::2]  # Green
            p3 = _raw[1::2, 1::2]  # Green
            p4 = _raw[1::2, 0::2]  # Red

            blue = p1
            green = np.clip((p2 // 2 + p3 // 2), 0, 2 ** 16 - 1)
            red = p4

            return red, green, blue

        except Exception as e:
            print('Error in get_r_g_b_channels: {}'.format(e))

def takepictures():
    try:
        s = Logger()
        logger = s.getLogger()
        iproc = ImgProc()

        with picamera.PiCamera() as camera:
            camera.saturation = 0  # [-100, 100]
            camera.contrast = 0    # [-100, 100]
            camera.sharpness = 0   # [-100, 100]

            camera.framerate = 1
            camera.exposure_mode = 'off'
            camera.awb_mode = 'off'
            camera.shutter_speed = 0  # [0...330000)]
            camera.start_preview()
            camera.iso = 100

            print('Initializing camera...')
            sleep(2)

            header = 'img_nr,iso,ss,exp,r_avg,g_avg,b_avg'
            logger.info(header)

            nr = 1

            for ss in np.nditer(SS_LIST):
                camera.shutter_speed = ss
                #print('#{}, ss: {}'.format(nr,ss))

                stream = io.BytesIO()
                camera.capture(stream, format='jpeg', bayer=True)

                data = stream.getvalue()[-10270208:]
                data = data[32768:4128 * 2480 + 32768]
                data = np.fromstring(data, dtype=np.uint8)
                data = data.reshape((2480, 4128))[:2464, :4120]
                data = data.astype(np.uint16) << 2
                for byte in range(4):
                    data[:, byte::5] |= ((data[:, 4::5] >> ((4 - byte) * 2)) & 0b11)

                data = np.delete(data, np.s_[4::5], 1)
                dat = data.reshape([2464, 3296])
                dat = dat.astype('float')

                red, green, blue = iproc.get_r_g_b_channels(dat)

                cam_stats = dict(
                        _nr = nr,
                        iso=camera.iso,
                        ss=camera.shutter_speed,
                        exp=camera.exposure_speed,
                        r_avg ='{0:.2f}'.format(np.mean(red)),
                        g_avg ='{0:.2f}'.format(np.mean(green)),
                        b_avg ='{0:.2f}'.format(np.mean(blue)),
                    )
                logdata = '{_nr},{iso},{ss},{exp},{r_avg},{g_avg},{b_avg}'.format(**cam_stats)
                logger.info(logdata)

                nr +=1

    except Exception as e:
        camera.close()
        print('Error in takepicture: ' + str(e))



def main():
    try:    
        takepictures()

    except Exception as e:
        print('Error in Main: ' + str(e))


if __name__ == '__main__':
    main()
