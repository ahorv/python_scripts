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


global Path
Path = '/home/pi/python_scripts/raw/raw_pictures'

def zipitall(pathtodirofdata):
    try: 
        for nextdir, subdirs, files in os.walk(pathtodirofdata + "/"):
            newzipname = nextdir.split('/')[-1] 
            if newzipname:                                        
                compressFolder(pathtodirofdata,newzipname)
                print("{}.zip".format(newzipname))

    except IOError as e:
        print('ZIPALL : Could not create *.zip file: ' + str(e))
    


def compressFolder(mypath,zipfilename):
    try:
        #print("Compressed file in : {} ".format(mypath))
        dirtozip    = os.path.join(mypath,zipfilename)
        zipfilepath = os.path.join(mypath,zipfilename+".zip")
        
        zf = zipfile.ZipFile(zipfilepath, "w", zipfile.ZIP_DEFLATED)
        for dirname, subdirs, files in os.walk(dirtozip):
            #print('writing dirname: {}'.format(dirname))
            zf.write(dirname)        
            for filename in files:
                filepath = os.path.join(dirname, filename)             
                zf.write(filepath)
                #print('adding file {} to zip'.format(filename))
        zf.close()
        setOwnerAndPermission(zipfilepath)
    except IOError as e:
        print('ZIP : Could not create *.zip file: ' + str(e))


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
        global Path   
        zipitall(Path)
        #compressFolder(Path,dirname)
        print('all zipped !')

    except Exception as e:
        print('Error in Main: ' + str(e))


if __name__ == '__main__':
    main()
