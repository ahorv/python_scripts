#!/usr/bin/env python

from time import sleep
import pwd
import grp
import picamera
import picamera.array
from PIL import Image
import io
import os
import sys
import numpy as np
from datetime import datetime

# Raspicamera settings: https://www.raspberrypi.org/documentation/usage/camera/python/README.md

ISO_LIST = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 700, 800]
SS_LIST = np.linspace(100., 33000., 50)

#TEST
#ISO_LIST = [100]
#SS_LIST = np.linspace(100., 33000., 3)




Path = '/home/pi/python_scripts/crf/camera_pictures/'
if not os.path.exists(Path):
    os.makedirs(Path)


def takepictures(path):
    try:

        log = open(path + 'cflog.txt', 'w')
        
        setOwnerAndPermission(os.path.join(path, 'cflog.txt'))
        with picamera.PiCamera() as camera:
            camera.saturation = 0  # [-100, 100]
            camera.contrast = 0  # [-100, 100]
            camera.sharpness = 0  # [-100, 100]

            camera.framerate = 1
            camera.exposure_mode = 'off'
            camera.awb_mode = 'off'
            camera.shutter_speed = 100  # [0...330000)]
            camera.start_preview()

            for ISO in ISO_LIST:
                camera.iso = ISO
                print('ISO set to: {}'.format(camera.iso))

                #SubPath = Path + 'ISO_{}/'.format(ISO)
                #if not os.path.exists(SubPath):
                #    os.makedirs(SubPath)
                #    setOwnerAndPermission(SubPath)

                log.write('*** ISO {} ***\n'.format(camera.iso))

                csvname = 'ss_iso_{}.csv'.format(camera.iso)
                log2 = open(path + csvname,'w')
                
                log2.write('ss,avgb\n') # write header to csv file

                #with picamera.array.PiBayerArray(camera) as output:  # gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

                image = 0
                imgnr = 1

                for ss in np.nditer(SS_LIST):
                    sleep(2)
                  

                    camera.shutter_speed = ss

                    stream = io.BytesIO()
                    camera.capture(stream, format= 'jpeg')
                    stream.seek(0)
                    img = Image.open(stream)                  
                   
                    avgb = avgbrightness(img)

                    log2.write('{},{}\n'.format(ss,avgb))

                    print('Image {} with ss {} taken'.format(imgnr, ss))
                    
                    log.write('Shutter time: {}\n'.format(ss))
                    log.write('Avrg. brightness: {}\n'.format(avgb))
                    log.write('Image {}\n'.format(image))
                    log.write('Time: ' + str(datetime.now().time()) + '\n')
                    log.write('Exposure: {}     ISO: {}\n'.format(camera.exposure_speed, camera.iso))
                    log.write(
                        'Analog Gain: {}   Digital Gain: {}\n'.format(camera.analog_gain, camera.digital_gain))
                    log.write('AWB Gains: {}     DRC Strength: {}\n'.format(camera.awb_gains, camera.drc_strength))
                    log.write('Exposure Compensation: {}      Brightness: {}\n'.format(camera.exposure_compensation,
                                                                                       camera.brightness))
                    log.write(
                        'Exposure Mode: {}     Sensor Mode: {}\n'.format(camera.exposure_mode, camera.sensor_mode))
                    log.write('\n')

                    imgnr = imgnr + 1
                    #arr = np.array(img)
                    #image += 1
                    #np.save(SubPath + 'image{}'.format(image), arr)
                    #setOwnerAndPermission(SubPath + 'image{}'.format(image) + '.npy')

            log.close()
    except Exception as e:
        camera.close()
        print('Error in takepicture: ' + str(e))

def avgbrightness(im):
    """
    Find the average brightness 
    """
    try:
        aa = im.copy()
        if aa.size[0] > 128:
          aa.thumbnail((128, 96), Image.ANTIALIAS)
        aa = im.convert('L') # Converts to greyscale
        (h, w) = aa.size
        pixels = (aa.size[0] * aa.size[1])
        h = aa.histogram()
        mu0 = 1.0 * sum([i * h[i] for i in range(len(h))]) / pixels
        return round(mu0, 2)

    except IOError as e:
        print('avgbrightness : error: ' + str(e))


def setOwnerAndPermission(pathToFile):
    try:
        uid = pwd.getpwnam('pi').pw_uid
        gid = grp.getgrnam('pi').gr_gid
        os.chown(pathToFile, uid, gid)
        os.chmod(pathToFile, 0o777)
    except IOError as e:
        print('PERM : Could not set permissions for file: ' + str(e))


def main():
    try:    
        takepictures(Path)

    except Exception as e:
        print('Error in Main: ' + str(e))


if __name__ == '__main__':
    main()
