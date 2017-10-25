#!/usr/bin/env python

from __future__ import (
    unicode_literals,
    absolute_import,
    print_function,
    division,
    )

import io
import os
import time
import pwd
import grp
import sys
import picamera
import zipfile
from datetime import datetime
import numpy as np
from numpy.lib.stride_tricks import as_strided
from fractions import Fraction

'''
Takes pictures in raw bayer format with different shutter times
Version: 1
Attila Horvat
25.10.2017
'''


global Path
Path = '/home/pi/python_scripts/raw/raw_pictures'


def takepictures(mypath):
    try:
        global Path
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')        
        Path = mypath + "/" + timestamp
        os.makedirs(Path)
        setOwnerAndPermission(Path)
      
        stream = io.BytesIO()
        with picamera.PiCamera() as camera:
            # Let the camera warm up for a couple of seconds
            camera.resolution = (2592, 1944)
            #1/2 per second
            #camera.framerate = Fraction(1, 2)
            #shutter speed is limited by framerate!
            camera.framerate = 1
            camera.exposure_mode = 'off'
            camera.awb_mode = 'auto'
            camera.iso = 0
            shutter_speed = 100
            for i0 in range(10):
                #set shutter speed
                camera.shutter_speed = (i0+1)*shutter_speed
                #time.sleep(2)        
                fileName = 'raw_img%s.jpg'%str(i0)
                # Capture the image, without the Bayer data to file
                camera.capture(Path + "/" + fileName, format='jpeg', bayer=False)
                # Capture the image, including the Bayer data to stream
                camera.capture(stream, format='jpeg', bayer=True)
                
                #camera settings
                exp = camera.exposure_speed
                ag = camera.analog_gain
                dg = camera.digital_gain
                awb = camera.awb_gains
                br = camera.brightness
                ct = camera.contrast
                print('exposure time %d, ag %f, dg %f, awb %s, br %d, ct = %d' %(exp,ag,dg,str(awb),br,ct))
                
                # Extract the raw Bayer data from the end of the stream (is in jpeg-meta data), check the
                # header and strip if off before converting the data into a numpy array

                data = stream.getvalue()[-10270208:]
                print('%s' %data[:4])
                #assert data[:4] == 'BRCM'
                data = data[32768:4128*2480+32768]
                data = np.fromstring(data, dtype=np.uint8)

                # The data consists of 2480 rows of 4128 bytes of data. The last rows
                # of data are unused (they only exist because the actual resolution of
                # is rounded up to the nearest 16). Likewise, the last 
                # bytes of each row are unused (why?). Here we reshape the data and
                # strip off the unused bytes

                data = data.reshape((2480, 4128))[:2464, :4120]

                # Horizontally, each row consists of 4120 10-bit values. Every four
                # bytes are the high 8-bits of four values, and the 5th byte contains
                # the packed low 2-bits of the preceding four values. In other words,
                # the bits of the values A, B, C, D and arranged like so:
                #
                #  byte 1   byte 2   byte 3   byte 4   byte 5
                # AAAAAAAA BBBBBBBB CCCCCCCC DDDDDDDD AABBCCDD
                #
                # Here, we convert our data into a 16-bit array, shift all values left
                # by 2-bits and unpack the low-order bits from every 5th byte in each
                # row, then remove the columns containing the packed bits

                data = data.astype(np.uint16) << 2
                for byte in range(4):
                    data[:, byte::5] |= ((data[:, 4::5] >> ((4 - byte) * 2)) & 0b11)
                
                data = np.delete(data, np.s_[4::5], 1)

                # Finally save raw (16bit data) having a size 3296 x 2464 
                fileName = 'data%d_%s.data'%(i0,str(''))
                print('%s' %fileName)
                with open(Path + "/" + fileName, 'wb') as g:
                    data.tofile(g)

                setOwnerAndPermission(Path)
    
    except Exception as e:
        #camera.close()
        print('Error in takepicture: ' + str(e))

def setOwnerAndPermission(pathToFile):
    try:
        uid = pwd.getpwnam('pi').pw_uid
        gid = grp.getgrnam('pi').gr_gid
        os.chown(pathToFile, uid, gid)
        os.chmod(pathToFile, 0o777)
    except IOError as e:
        print('PERM : Could not set permissions for file: ' + str(e))
        

def createNewFolder(mypath):
    try:
        if not os.path.exists(mypath):
            os.makedirs(Path)
            setOwnerAndPermission(mypath)
    except IOError as e:
        print('DIR : Could not create new folder: ' + str(e))


def compressFolder(mypath,zipfilename):
    try:
        print("Compressed file in : {} ".format(mypath))
        dirtozip    = os.path.join(mypath,zipfilename)
        zipfilepath = os.path.join(mypath,zipfilename+".zip")
        
        zf = zipfile.ZipFile(zipfilepath, "w", zipfile.ZIP_DEFLATED)
        for dirname, subdirs, files in os.walk(dirtozip):
            print('writing dirname: {}'.format(dirname))
            zf.write(dirname)        
            for filename in files:
                filepath = os.path.join(dirname, filename)             
                zf.write(filepath)
                print('adding file {} to zip'.format(filename))
        zf.close()
        setOwnerAndPermission(zipfilepath)
    except IOError as e:
        print('ZIP : Could not create *.zip file: ' + str(e))


def main():
    try:
        global Path
        createNewFolder(Path)
        takepictures(Path)
        #compressFolder(Path)

    except Exception as e:
        print('Error in Main: ' + str(e))


if __name__ == '__main__':
    main()
        
