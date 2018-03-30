#!/usr/bin/env python

import os
import cv2
import sys
import time
from shutil import copy2
from glob import glob
import subprocess
from matplotlib import pyplot as plt

print('Version opencv: ' + cv2.__version__)

######################################################################
## Hoa: 26.03.2018 Version 1 : grab_JPG_5.py
######################################################################
# Grabs from a image collection all raw_img5.jpg 's and shows them as
# Slideshow
#
# New /Changes:
# ----------------------------------------------------------------------
#
# 26.03.2018 : first implemented
# 30.03.2018 : added copy img to new folder
# 30.03.2018 : added video creation by ffmpeg
#
######################################################################

global Path_to_raw
global Path_to_copy
global Path_to_ffmpeg
Path_to_raw = r'C:\Users\ati\Desktop\raw_data'
Path_to_copy = r'C:\Users\ati\Desktop\raw_data_img5'
Path_to_ffmpeg = r'C:\ffmpeg\bin\ffmpeg.exe'

class Helpers:
    def createVideo():
        try:
            global Path_to_ffmpeg                            # path to ffmpeg executable
            fsp = ' -r 10 '                                  # frame per sec images taken
            stnb = '-start_number 1 '                        # what image to start at
            imgpath = '-i ' + Path_to_copy + '\%d_img5.jpg ' # path to images
            res = '-s 2592x1944 '                            # output resolution
            outpath = Path_to_copy+'\sky_video.mp4 '         # output file name
            codec = '-vcodec libx264'                        # codec to use

            command = Path_to_ffmpeg + fsp + stnb + imgpath + res + outpath + codec

            if sys.platform == "linux":
                subprocess(command, shell=True)
            else:
                print(' {}'.format(command))
                ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE ,stdout = subprocess.PIPE)
                out, err = ffmpeg.communicate()
                if (err): print(err)

        except Exception as e:
            print('createVideo: Error: ' + str(e))

    def getDirectories(pathToDirectories):
        try:
            allDirs = []
            temp = ''
            cnt = 0

            for dirs in sorted(glob(os.path.join(pathToDirectories, "*", ""))):
                if os.path.isdir(dirs):
                    allDirs.append(dirs)
                    print('{}'.format(str(dirs)))

            print('All images loaded !')

            return allDirs

        except Exception as e:
            print('getDirectories: Error: ' + str(e))

    def readAllImages(allDirs):
        try:
            global Path_to_raw
            list_names = []
            list_images = []
            cnt = 1
            print('Converting jpg to opencv, may take some time!')

            t_start = time.time()
            for next_dir in allDirs:
                next_dir += 'raw_img5.jpg'
                img = cv2.imread(next_dir, 1)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                new_img = cv2.resize(img_rgb, None, fx=0.25, fy=0.25)
                list_images.append(new_img)
                cnt += 1

            t_end = time.time()
            measurement = t_end-t_start
            print('Total number of images: {}'.format(cnt))
            print('Time to load and convert imgs: {}'.format(measurement))

            return list_images

        except Exception as e:
            print('readAllImages: Error: ' + str(e))

    def copyAll_img5(list_alldirs):

        try:
            print('\nCopying all raw_img_5 to \img_5')
            global Path_to_raw
            new_path = Path_to_raw +'\imgs5'
            prefix = 0

            if not os.path.exists(new_path):
                os.makedirs(new_path)

            for next_dir in list_alldirs:
                newimg = next_dir+'raw_img5.jpg'
                prefix += 1
                copy2(newimg,new_path+'/' + '{}_img5.jpg'.format(prefix))

            print('All raw_5.jpg copied to: {}'.format(new_path))

        except Exception as e:
            print('copyAll_img5: Error: ' + str(e))

def main():
    try:
        global Path_to_raw

        h = Helpers()

        h.createVideo()
        return

        list_images = []
        allDirs = []
        allDirs = h.getDirectories(Path_to_raw)
        #h.copyAll_img5(allDirs)
        list_images = h.readAllImages(allDirs)

        counter = 0
        fig = plt.figure()
        ax = plt.gca()
        cur_window = ax.imshow(list_images[0])

        while counter < len(list_images):

            next = list_images[counter]
            plt.title('Image: {}'.format(counter))
            cur_window.set_data(next)

            plt.pause(.05)
            plt.draw()

            counter += 1

    except Exception as e:
        print('MAIN: Error in main: ' + str(e))


if __name__ == '__main__':
    main()