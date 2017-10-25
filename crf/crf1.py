#!/usr/bin/env python
from time import sleep
import picamera
import picamera.array
import os
import numpy
from datetime import datetime

ISO_LIST = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 700, 800]
num_frames = 3

Path = '/home/pi/python_scripts/camera_pictures'
if not os.path.exists(Path):
    os.makedirs(Path)

log = open(Path+'cflog.txt', 'w')
with picamera.PiCamera() as camera:

    camera.framerate = 1
    camera.exposure_mode = 'auto'
    camera.awb_mode = 'off'
    camera.shutter_speed = 1000
    camera.start_preview()
           

    for ISO in ISO_LIST:
        # camera.exposure_mode = 'auto'
        camera.iso = ISO
        # sleep(7)
        print( 'ISO set to: {}'.format(camera.iso))
        # sleep(2)
        # camera.exposure_mode = 'off'

        SubPath = Path+'ISO_{}/'.format(ISO)
        if not os.path.exists(SubPath):
            os.makedirs(SubPath)
        log.write('*** ISO {} ***\n'.format(camera.iso))
        
        with picamera.array.PiBayerArray(camera) as output:

            image = 0

            while image < num_frames:
                
                sleep(2)

                camera.capture(output, 'jpeg', bayer=True)
                # camera.capture(SubPath+'image{}.jpg'.format(image+1), 'jpeg', bayer=True)

                print( 'Image {} Taken'.format(image+1))

                log.write('Image {}\n'.format(image))
                log.write('Time: '+str(datetime.now().time())+'\n')
                log.write('Exposure: {}     ISO: {}\n'.format(camera.exposure_speed,camera.iso))
                log.write('Analog Gain: {}   Digital Gain: {}\n'.format(camera.analog_gain,camera.digital_gain))
                log.write('AWB Gains: {}     DRC Strength: {}\n'.format(camera.awb_gains,camera.drc_strength))
                log.write('Exposure Compensation: {}      Brightness: {}\n'.format(camera.exposure_compensation,camera.brightness))
                log.write('Exposure Mode: {}     Sensor Mode: {}\n'.format(camera.exposure_mode,camera.sensor_mode))
                log.write('\n')


                arr = output.array
                image += 1
                numpy.save(SubPath+'image{}'.format(image), arr)
                
log.close()
