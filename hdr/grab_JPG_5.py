#!/usr/bin/env python

import os
import cv2
import shutil
from os import listdir
from os.path import isfile, join
from glob import glob

print('Version opencv: ' + cv2.__version__)

######################################################################
## Hoa: 26.03.2018 Version 1 : grab_JPG_5.py
######################################################################
# Grabs from a image collection all raw_img5.jpg 's
#
# New /Changes:
# ----------------------------------------------------------------------
#
# 26.03.2018 : first implemented
#
#
######################################################################

global Path_to_raw
Path_to_raw = r'C:\Users\ati\Desktop\raw_data'

def getDirectories(pathToDirectories):
    try:
        allDirs = []

        for dirs in sorted(glob(os.path.join(pathToDirectories, "*", ""))):
            if os.path.isdir(dirs):
                allDirs.append(dirs)
                print('\t\t'+dirs)

        return allDirs

    except Exception as e:
        print('getDirectories: Error: ' + str(e))


def main():
    try:
        global Path_to_raw

        alldirs = []
        alldirs = getDirectories(Path_to_raw)

        print('\n')

        #walkThrough(Path_to_raw)

        #createNewFolder('./ouput')
        #createHDR(Path_to_raw,['0','5','9'])

    except Exception as e:
        print('MAIN: Error in main: ' + str(e))


if __name__ == '__main__':
    main()