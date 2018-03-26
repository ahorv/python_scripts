#!/usr/bin/env python

import os
import shutil
from os import listdir
from os.path import isfile, join
from glob import glob
import zipfile



######################################################################
## Hoa: 29.10.2017
######################################################################
# Test
# delete this file
#
######################################################################

global Path_to_raw
Path_to_raw =r'C:\Users\ati\Desktop\raw_data'


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

def walkThrough(pathToDirectories):
    try:
        for dirname, subdirs, files in os.walk(pathToDirectories):
            for filename in files:
                print(os.path.join(dirname, filename))
            print('\n')


    except Exception as e:
        print('walkThrough: Error' + str(e))

def zipitall(self):
    try:
        global RAWDATAPATH
        dirtozip = ''
        for nextdir, subdirs, files in os.walk(RAWDATAPATH + "/"):
            newzipname = nextdir.split('/')[-1]
            if newzipname:

                dirtozip    = os.path.join(RAWDATAPATH,newzipname)
                zipfilepath = os.path.join(RAWDATAPATH,newzipname)

                zf = zipfile.ZipFile(zipfilepath+'.zip', "w")
                for dirname, subdirs, files in os.walk(dirtozip):
                    for filename in files:
                        zf.write(os.path.join(dirname, filename), filename, compress_type = zipfile.ZIP_DEFLATED)
                zf.close()

                #delete zipped directories
                #shutil.rmtree(dirtozip, ignore_errors=True)


    except IOError as e:
        print('ZIPALL : Could not create *.zip file: ' + str(e))



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