#!/usr/bin/env python

import Image
import os, sys, argparse
import subprocess
import time
import math
import zmq
import io, picamera
from fractions import Fraction


class config(object):
    """Config Options:
    `w` : Width of images.
    `h` : Height of images.
    `iso` : ISO used for all images.
    `maxss` : maximum shutter speed
    `minss` : minimum shutter speed
    `maxfr` : maximum frame rate
    `minfr` : minimum frame rate
    `metersite` : Chooses a region of the image to use for brightness
     `brightwidth` : number of previous readings to store for choosing next
      shutter speed.
      `gamma` : determines size of steps to take when adjusting shutterspeed.
    """


    def __init__(self, config_map={}):
        self.w = config_map.get('w', 1296)
        self.h = config_map.get('h', 972)
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
        self.max_fr = config_map.get('minfr', 15)
        self.min_fr = config_map.get('maxfr', 1)

        # Dynamic adjustment settings.
        self.brightwidth = config_map.get('brightwidth', 20)
        self.gamma = config_map.get('gamma', 0.2)

        self.disable_led = config_map.get('disable_led', False)


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
            'disable_led': self.disable_led,
        }


def doCRF():
    """
    Timelapser class.
    Once the timelapser is initialized, use the `findinitialparams` method to find
    an initial value for shutterspeed to match the targetBrightness.
    Then run the `timelapser` method to initiate the actual timelapse.
    EXAMPLE::
      T=timelapse()
      T.run_timelapse()

    The timelapser broadcasts zmq messages as it takes pictures.
    The `listen` method sets up the timelapser to listen for signals from 192.168.0.1,
    and take a shot when a signal is received.
    EXAMPLE::
      T=timelapse()
      T.listen()
    """





def main():
    try:

        cfg = {
            'w': 1296,
            'h': 972,
            'interval': 100,
            'maxshots': -1,
            'maxtime': -1,
            'targetBrightness': 128,
            'maxdelta': 100,
            'iso': 100,
            # Add more configuration options here, if desired.
        }

        camera = picamera.PiCamera()
        doCRF(camera, cfg)

    except Exception as e:
       print('MAIN: Error in main: ' + str(e))
    finally:
      camera.close()


if __name__ == '__main__':
    main()
