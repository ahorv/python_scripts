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
import cv2
import picamera
import logging
import logging.handlers
from datetime import datetime, timedelta
import numpy as np
from fractions import Fraction
import pwd
import grp
import stat

if sys.platform == "linux":
    import pwd
    import grp
    import stat
    import fcntl


######################################################################
## Hoa: 24.09.2018 Version 1 : picam.py
######################################################################
# This class takes 3 consecutive images with increasing shutter times.
# Pictures are in raw bayer format. In addition a jpg as reference
# image is taken for each raw image.
#
# Proper exposure is maintained by a simple gradient descent, trying
# to keep the delta between measured brightness and desired minimal.
#
# Start and end time, are set here inside the script ! Script is run
# by a cronjob.
#
# New /Changes:
# ----------------------------------------------------------------------
#
# 24.09.2018 : First implemented
#
######################################################################

global SCRIPTPATH
global RAWDATAPATH
global SUBDIRPATH

SCRIPTPATH = os.path.join('/home', 'pi', 'python_scripts', 'picam')
RAWDATAPATH = os.path.join(SCRIPTPATH, 'picam_data')


class Logger:
    def __init__(self):
        self.logger = None

    def getLogger(self, newLogPath=None):

        try:
            global SCRIPTPATH

            if newLogPath is None:
                LOGFILEPATH = os.path.join(SCRIPTPATH, 'raw2.log')
                logFormatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
                fileHandler = logging.FileHandler(LOGFILEPATH)
                name = 'rootlogger'
            else:
                LOGFILEPATH = newLogPath
                logFormatter = logging.Formatter('%(message)s')
                fileHandler = logging.FileHandler(LOGFILEPATH)
                name = 'camstatslogger'

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
            helper.setOwnerAndPermission(LOGFILEPATH)
            return self.logger

        except IOError as e:
            print('Error logger:' + str(e))

    def closeLogHandler(self):
        try:
            handlers = self.logger.handlers[:]
            for handler in handlers:
                handler.close()
                self.logger.removeHandler(handler)

        except IOError as e:
            print('Error logger:' + str(e))

class Helpers:

    def createNewFolder(self, thispath):
        try:
            if not os.path.exists(thispath):
                os.makedirs(thispath)
                self.setOwnerAndPermission(thispath)

        except IOError as e:
            print('DIR : Could not create new folder: ' + str(e))

    def createNewRawFolder(self):
        try:
            global RAWDATAPATH
            global SUBDIRPATH

            TSTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.createNewFolder(RAWDATAPATH)
            SUBDIRPATH = os.path.join(RAWDATAPATH, TSTAMP)
            self.createNewFolder(SUBDIRPATH)
            camLogPath = os.path.join(SUBDIRPATH, 'camstats.log')

            return camLogPath

        except IOError as e:
            print('PATH: Could not set path and folder: ' + str(e))

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

    def str2time(self, time_as_str):

        try:
            now = datetime.now()
            year = str(now.year)
            month = str(now.month)
            day = str(now.day)
            temp = year + '-' + month + '-' + day + '_' + time_as_str
            str_as_time = datetime.strptime(temp, '%Y-%m-%d_%H:%M:%S')

            return str_as_time

        except Exception as e:
            print('str2time: ' + str(e))

class Current_State(object):
    """Container class for exposure controller state.
    """
    def __init__(self, config, state_map={}):
        # Current Shutter Speed
        self.currentSS = state_map.get('currentSS', 0)
        # Current Frame Rate
        self.currentFR = state_map.get('currentFR', config.max_fr)
        # Current Frame Rate
        self.currentEXP = state_map.get('currentEXP', 0)
        # Current WB Gains
        self.currentWB_Gains = state_map.get('currentWB_Gains', 0)
        # Current AWB Gains
        self.currentAWB_Gains = state_map.get('currentAWB_Gains', 0)
        # List of average brightness of recent images.
        self.brData = state_map.get('brData', [])
        # List of shutter speeds of recent images.
        self.xData = state_map.get('xData', [])
        # Number of pictures taken
        self.shots_taken = state_map.get('shots_taken', 0)
        # White balance
        self.wb = state_map.get('wb', (Fraction(337, 256), Fraction(343, 256)))
        # Duration to find new shutter time
        self.found_ss_dur = state_map.get('found_ss_dur',0)
        # Duration to take jpg image triplet
        self.low_jpg_dur  = state_map.get('low_jpg_dur',0)
        self.well_jpg_dur = state_map.get('well_jpg_dur', 0)
        self.over_jpg_dur = state_map.get('over_jpg_dur', 0)
        # Duration to take data image triplet
        self.low_data_dur  = state_map.get('low_data_dur',0)
        self.well_data_dur = state_map.get('well_data_dur', 0)
        self.over_data_dur = state_map.get('over_data_dur', 0)
        # Shutter time used to take image triplet
        self.low_exp_ss = state_map.get('low_exp_ss', 0)
        self.well_exp_ss = state_map.get('well_exp_ss', 0)
        self.over_exp_ss = state_map.get('over_exp_ss', 0)

class Camera_config(object):
  """Config Options:
    `w` : Width of images.
    `h` : Height of images.
    `interval` : Interval of shots, in seconds.  Recommended minimum is 10s.
    `maxtime` : Maximum amount of time, in seconds, to run the timelapse for.
      Set to 0 for no maximum.
    `maxshots` : Maximum number of pictures to take.  Set to 0 for no maximum.
    `targetBrightness` : Desired brightness of images, on a scale of 0 to 255.
    `maxdelta` : Allowed variance from target brightness.  Discards images that
      are more than `maxdelta` from `targetBrightness`.  Set to 256 to keep
      all images.
    `iso` : ISO used for all images.
    `maxss` : maximum shutter speed
    `minss` : minimum shutter speed
    `maxfr` : maximum frame rate
    `minfr` : minimum frame rate
    `metersite` : Chooses a region of the image to use for brightness
      measurements. One of 'c', 'a', 'l', or 'r', for center, all, left or
      right.
    `brightwidth` : number of previous readings to store for choosing next
      shutter speed.
    `gamma` : determines size of steps to take when adjusting shutterspeed.
  """
  def __init__(self, config_map={}):
      self.w = config_map.get('w', 2592)
      self.h = config_map.get('h', 1944)
      self.iso = config_map.get('iso', 100)
      self.interval = config_map.get('interval', 15)
      self.maxtime = config_map.get('maxtime', -1)
      self.maxshots = config_map.get('maxshots', -1)
      self.targetBrightness = config_map.get('targetBrightness', 128)
      self.maxdelta = config_map.get('maxdelta', 100)

      # Setting the maxss under one second prevents flipping into a slower camera mode.
      self.maxss = config_map.get('maxss', 999000)
      self.minss = config_map.get('minss', 100)

      # Note: these should depend on camera model...
      self.max_fr = config_map.get('maxfr', 15)
      self.min_fr = config_map.get('minfr', 1)

      # Dynamic adjustment settings.
      self.brightwidth = config_map.get('brightwidth', 20)
      self.gamma = config_map.get('gamma', 0.2)

  def floatToSS(self, x):
      base = int(self.minss + (self.maxss - self.minss) * x)
      return max(min(base, self.maxss), self.minss)

  def SSToFloat(self, ss):
      base = (float(ss) - self.minss) / (self.maxss - self.minss)
      return max(min(base, 1.0), 0.0)

  def to_dict(self):
    return {
      'w': self.w,
      'h': self.h,
      'iso': self.iso,
      'interval': self.interval,
      'maxtime': self.maxtime,
      'maxshots': self.maxshots,
      'targetBrightness': self.targetBrightness,
      'maxdelta': self.maxdelta,
      'maxss': self.maxss,
      'minss': self.minss,
      'max_fr': self.max_fr,
      'min_fr': self.min_fr,
      'brightwidth': self.brightwidth,
      'gamma': self.gamma,
    }

class Camera:
    """
    Camera class. Needs an instance (as parameter) of picamera
    Once the Camera class is initialized, use the `findinitialparams` method to find
    an initial value for shutterspeed to match the targetBrightness.
    Then run the `take_picture` method to initiate the actual process.
    EXAMPLE::
      camera = Camera()
      camera.take_picture()
    """
    def __init__(self, picam_instance, config=None):
        if config == None:
            config = Camera_config({})
        self.config = config
        self.camera = picam_instance
        self.camera.resolution = (config.w, config.h)
        self.camera.iso = config.iso
        # Shutter speed normalized between 0 and 1 as floating point number,
        # denoting position between the max and min shutterspeed.

        self.current_state = Current_State(config)
        self.camera.framerate = self.current_state.currentFR

        print('Finding initial SS....')
        # Give the camera's auto-exposure and auto-white-balance algorithms
        # some time to measure the scene and determine appropriate values
        time.sleep(2)
        # This capture discovers initial AWB and SS.
        self.camera.capture('ini_img.jpg')
        self.camera.shutter_speed = self.camera.exposure_speed
        self.current_state.currentSS = self.camera.exposure_speed
        self.camera.exposure_mode = 'off'
        self.current_state.wb_gains = self.camera.awb_gains
        print('WB: ', self.current_state.wb_gains)
        self.camera.awb_mode = 'off'
        self.camera.awb_gains = self.current_state.wb_gains

        self.findinitialparams(self.config, self.current_state)
        print("Set up picam with: ")
        print("\tTarget Brightns:\t", config.targetBrightness)
        print("\tPicture size   :\t", config.w, 'x', config.h)

    def avgbrightness(self, im, config=None):
        """
        Find the average brightness of the provided image.

        Args:
          im: A opencv image.
          config: A timelapseConfig object.  Defaults to self.config.
        Returns:
          Average brightness of the image.
        """
        if config is None: config = self.config
        aa = im.copy()
        heigth, width, channels = aa.shape

        if width > 128:
            imRes = cv2.resize(aa, (128, 96), interpolation=cv2.INTER_AREA)
            aa = cv2.cvtColor(imRes, cv2.COLOR_BGR2GRAY)
        else:
            aa = cv2.cvtColor(aa, cv2.COLOR_BGR2GRAY)

        pixels = (aa.shape[0] * aa.shape[1])
        h = cv2.calcHist([aa], [0], None, [256], [0, 256])
        mu0 = 1.0 * sum([i * h[i] for i in range(len(h))]) / pixels
        return round(mu0[0], 2)

    def dynamic_adjust(self, config=None, state=None):
        """
        Applies a simple gradient descent to try to correct shutterspeed and
        brightness to match the target brightness.
        """
        if config is None: config = self.config
        if state is None: state = self.state

        delta = config.targetBrightness - state.brData[-1]

        Adj = lambda v: v * (1.0 + 1.0 * delta * config.gamma/ config.targetBrightness)
        x = config.SSToFloat(state.currentSS)
        x = Adj(x)
        if x < 0: x = 0
        if x > 1: x = 1
        state.currentSS = config.floatToSS(x)

        # Find an appropriate framerate.
        # For low shutter speeds, ths can considerably speed up the capture.
        FR = Fraction(1000000, state.currentSS)
        if FR > config.max_fr: FR = Fraction(config.max_fr)
        if FR < config.min_fr: FR = Fraction(config.min_fr)
        state.currentFR = FR

    def findinitialparams(self, config=None, state=None):

        """
        Take a number of small shots in succession to determine shutterspeed
        and ISO for taking photos of the desired brightness.
        """
        if config is None: config = self.config
        if state is None: state = self.state
        killtoken = False

        # Find init params with small pictures and high gamma, to work quickly.
        cfg = config.to_dict()
        cfg['gamma'] = 2.0
        init_config = Camera_config(cfg)

        state.brData = [0]
        state.xData = [0]

        while abs(config.targetBrightness - state.brData[-1]) > 4:
            im = self.single_shoot(128, 96, None, init_config, state)
            state.brData = [self.avgbrightness(im,None)]
            state.xData = [self.config.SSToFloat(state.currentSS)]

            # Dynamically adjust ss and iso.
            self.dynamic_adjust(init_config, state)
            print('Searching init. params{ ss: % 10d\tx: % 6.4f br: % 4d\t}' % (state.currentSS, round(state.xData[-1], 4), round(state.brData[-1], 4)))
            if state.xData[-1] >= 1.0:
                if killtoken == True:
                    break
                else:
                    killtoken = True
            elif state.xData[-1] <= 0.0:
                if killtoken == True:
                    break
                else:
                    killtoken = True
        return True

    def single_shoot(self, resize_width=None, resize_hight = None, shutter_speed=None, config=None, state=None):
        '''
        Takes a single image as jpeg and returns it as opencv image.
        :param resize_width:  new image width
        :param resize_hight:  new image heigth
        :param shutter_speed: overwrite shuter speed in config file
        :param config: current camera settings
        :param state:  current state
        :return: image as opencv image
        '''

        start_timer = time.time()

        if config is None: config = self.config
        if state is None: state = self.current_state

        # update camera parameters
        self.camera.ISO = config.iso
        self.camera.framerate = state.currentFR
        self.camera.resolution = (config.w, config.h)

        if shutter_speed is None:
            self.camera.shutter_speed = state.currentSS
        else:
            self.camera.shutter_speed = shutter_speed
        stream = io.BytesIO()

        if (resize_width is not None and resize_hight is not None):
            self.camera.capture(stream, format='jpeg',resize=(resize_width, resize_hight), bayer=False)
        else:
            self.camera.capture(stream, format='jpeg',bayer=False)

        nparray = np.fromstring(stream.getvalue(), dtype=np.uint8)
        image = cv2.imdecode(nparray, 1)
        end_time = time.time()

        # print to console
        ss = self.camera.shutter_speed
        exp = self.camera.exposure_speed
        frmr = round(float(self.camera.framerate), 2)
        durration = round(end_time - start_timer,2)
        #print('Single shoot: Exp: %d\t SS: %10d\t Framerate: %f\t Duration Time: %f' % (exp,ss,frmr, durration))

        return image

    def single_shoot_data(self, resize_width=None, resize_hight = None, shutter_speed=None, config=None, state=None):
        '''
        Takes a single image in raw and returns it as numpy array.
        :param resize_width:  new image width
        :param resize_hight:  new image heigth
        :param shutter_speed: overwrite shuter speed in config file
        :param config: current camera settings
        :param state:  current state
        :return: image as numpy array
        '''
        start_timer = time.time()

        if config is None: config = self.config
        if state is None: state = self.current_state

        # update camera parameters
        self.camera.ISO = config.iso
        self.camera.framerate = state.currentFR
        self.camera.resolution = (config.w, config.h)

        if shutter_speed is None:
            self.camera.shutter_speed = state.currentSS
        else:
            self.camera.shutter_speed = shutter_speed
        stream = io.BytesIO()

        if (resize_width is not None and resize_hight is not None):
            self.camera.capture(stream, format='jpeg',resize=(resize_width, resize_hight), bayer=True)
        else:
            self.camera.capture(stream, format='jpeg',bayer=True)

        print("single shoot data: exp_mode: %s, awb_mode: %s", str(self.camera.exposure_mode), str(self.camera.awb_mode))

        data = stream.getvalue()[-10270208:]
        data = data[32768:4128 * 2480 + 32768]
        data = np.fromstring(data, dtype=np.uint8)
        data = data.reshape((2480, 4128))[:2464, :4120]
        data = data.astype(np.uint16) << 2
        for byte in range(4):
            data[:, byte::5] |= ((data[:, 4::5] >> ((4 - byte) * 2)) & 0b11)

        data = np.delete(data, np.s_[4::5], 1)
        end_time = time.time()

        # print to console
        ss = self.camera.shutter_speed
        exp = self.camera.exposure_speed
        frmr = round(float(self.camera.framerate), 2)
        durration = round(start_timer - end_time,2)
        print('Single shoot data: Exp: %d\t SS: %10d\t Framerate: %f\t Duration Time: %f' % (exp,ss,frmr, durration))

        return data

    def adjust_ss(self, ss_adjust=True, config=None, state=None):
        try:
            '''
            By default ss_adjust = true.
            Take and evaluate pictures as long ss is not adjusted
            '''
            if not ss_adjust: return
            if config is None: config = self.config
            if state is None: state = self.current_state

            found_ss = False
            start_time = time.time()

            im = self.single_shoot(128, 96, None, config, state)

            state.lastbr = self.avgbrightness(im)
            if len(state.brData) >= config.brightwidth:
                state.brData = state.brData[1:]
                state.xData = state.xData[1:]
            state.xData.append(self.config.SSToFloat(state.currentSS))
            state.brData.append(state.lastbr)

            # Dynamically adjust ss and iso.
            state.avgbr = sum(state.brData) / len(state.brData)
            self.dynamic_adjust(config, state)
            state.shots_taken += 1

            delta = config.targetBrightness - state.lastbr
            if abs(delta) > config.maxdelta:
                # Too far from target brightness.
                state.shots_taken -= 1
             #   os.remove(filename)
            else:
                found_ss = True

            end_time = time.time()
            duration = round(start_time - end_time,2)
            state.found_ss_dur = duration

            return found_ss

        except Exception as e:
            print('Error in adjust_ss: ' + str(e))
            return found_ss

    def takepictures(self):
        try:
            global SUBDIRPATH
            state = self.current_state
            found_ss = self.adjust_ss(True,None,None)

            h = Helpers()
            camLogPath = h.createNewRawFolder()
            s = Logger()
            cameralog = s.getLogger(camLogPath)
            cameralog.info('Date and Time: {}'.format(datetime.now().strftime('%Y%m%d_%H%M%S')))

            # One stop is an exposure factor of 2 (2x or 1/2). Verdopplung oder halbieren
            # One EV is a step of one stop compensation.

            if found_ss:
                i0 = 1
                loopstart_tot = time.time()

                loopstartjpg = time.time()
                ss = state.currentSS
                # Capture jpg image, without Bayer data to file
                img1 = self.single_shoot(None,None,ss,None,None)
                fileName = 'raw_img%s.jpg' % str(i0)
                cv2.imwrite(SUBDIRPATH + "/" + fileName, img1)
                loopendjpg = time.time()

                # Capture raw image, including the Bayer data
                loopstartraw = time.time()
                dat1 = self.single_shoot_data(None,None,ss,None,None)
                datafileName = 'data%s.data' % str(i0)
                with open(SUBDIRPATH + "/" + datafileName, 'wb') as g:
                    dat1.tofile(g)
                loopendraw = time.time()
                loopend_tot = time.time()

                # camera settings
                cam_stats = dict(
                    ss= self.camera.shutter_speed,
                    iso=self.camera.ISO,
                    exp=self.camera.exposure_speed,
                    ag= self.camera.analog_gain,
                    dg= self.camera.digital_gain,
                    awb=self.camera.awb_gains,
                    br= self.camera.brightness,
                    ct= self.camera.contrast,
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

                cameralog.info(logdata)

            s.closeLogHandler()
            print('Taking picture: Exp: %d\t SS: %10d\t ISO: %f\t Duration Time: %f' % (self.camera.exposure_speed,self.camera.shutter_speed, self.camera.ISO, (loopend_tot - loopstart_tot)))

        except Exception as e:
            print('Error in takepicture: ' + str(e))


def main():
    try:
        # set camera parameter
        cfg = {
            'w': 2592,
            'h': 1944,
            'interval': 15,
            'maxshots': -1,
            'maxtime': -1,
            'targetBrightness': 128,
            'maxdelta': 100,
            'iso': 100,
            # Add more configuration options here, if desired.
        }

        helper = Helpers()
        usedspace = helper.disk_stat()
        s = Logger()
        log = s.getLogger()

        if usedspace > 80:
            raise RuntimeError('WARNING: Not enough free space on SD Card!')
            return

        picam = picamera.PiCamera()
        camera = Camera(picam,Camera_config(cfg))

        time_start = '9:00:00'  # Start time of time laps
        time_end   = '15:00:00'  # Stop time of time laps

        t_start = helper.str2time(time_start)
        t_end = helper.str2time(time_end)

        while (True):
            time_now = datetime.now().replace(microsecond=0)

            camera.takepictures()
            '''
            if t_start < time_now < t_end:
                camera.takepictures()

            elif t_end > time_now or t_start < time_now:
                sys.exit()
            '''

    except Exception as e:
        picam.close()
        log.error(' MAIN: Error in main: ' + str(e))

if __name__ == "__main__":
   main()