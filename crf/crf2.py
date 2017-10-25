#!/usr/bin/env python
import picamera
import time

imgpath = '/home/pi/python_scripts/crf/consistent_captures/'

use_video_port = True
white_balance_gains = None

with picamera.PiCamera() as camera:
    camera.framerate = 1
    camera.resolution = (2592, 1944)
    camera.shutter_speed = 8333 * 12
    camera.iso = 100
    if white_balance_gains is None:
        time.sleep(3)
        print("Fixing white balance gains.")
        white_balance_gains = camera.awb_gains

    camera.awb_mode = 'off'
    camera.awb_gains = white_balance_gains
    time.sleep(3)
    camera.exposure_mode = 'off'

    i = 0
    while True:
        print("Capturing: " + str(i))
        print("--------------")
        print("White balance: " + str(camera.awb_gains))
        print("White balance mode: " + str(camera.awb_mode))
        print("ISO: " + str(camera.iso))
        print("Shutter speed: " + str(camera.shutter_speed))
        print("Brightness: " + str(camera.brightness))
        print("Digital gain: " + str(camera.digital_gain))
        print("Analog gain: " + str(camera.analog_gain))
        print("Exposure compensation: " + str(camera.exposure_compensation))
        print("\n")

        #print(camera._get_camera_settings())
        
        camera.capture(str(imgpath) + str(i) + ".jpg", use_video_port=use_video_port)
        i += 1
        time.sleep(2)
