#!/usr/bin/env python

from __future__ import print_function

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join
from glob import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

print('Version opencv: ' + cv2.__version__)

######################################################################
## Hoa: 05.12.2018 Version 1 : lumi_analysis.py
######################################################################
# Determine direct and diffuse parts of the sky image provided
# that the sun is in the image.
#
# New /Changes:
# ----------------------------------------------------------------------
#
# 05.12.2018 : first implemented
#
#
######################################################################
global Path_to_source

Path_to_source = r'A:\camera_1\cam_1_vers1\20180308_raw_cam1\temp'

def getDirectories(pathToDirectories):
    try:
        global Avoid_This_Directories
        Avoid_This_Directories = ['imgs5', 'hdr', 'rest']
        allDirs = []
        img_cnt = 1

        for dirs in sorted(glob(os.path.join(pathToDirectories, "*", ""))):
            if os.path.isdir(dirs):
                if dirs.rstrip('\\').rpartition('\\')[-1] not in Avoid_This_Directories:
                    allDirs.append(dirs)
                    #print('{}'.format(str(dirs)))
                    img_cnt +=1

        print('All images loaded! - Found {} images.'.format(img_cnt))

        return allDirs

    except Exception as e:
        print('getDirectories: Error: ' + str(e))

def strip_name_swvers_camid(path):
    path = path.lower()
    path = path.rstrip('\\')
    dir_name = (path.split('\\'))[-1]
    temp = (path.split('\\cam_'))[-1]
    temp = (temp.split('\\'))[0]
    temp = (temp.split('_'))
    camera_ID = temp[0]
    sw_vers = temp[1]
    sw_vers = sw_vers.replace('vers', '')

    if camera_ID.isdigit(): camera_ID = int(camera_ID)
    if sw_vers.isdigit(): sw_vers = int(sw_vers)

    return dir_name, sw_vers, camera_ID

def getImageName():
    source_path = Path_to_source.rstrip('\\temp')
    dir_name, sw_vers, camera_ID = strip_name_swvers_camid(source_path)

    if sw_vers == 1 or sw_vers == 2:
        img_name = 'raw_img5.jpg'

    if sw_vers == 3:
        img_name = 'raw_img0.jpg'

    return img_name

def showAsSubplots(i1,i2,i3,i4, name=''):

    try:

        fig = plt.figure(1)
        plt.gcf().canvas.set_window_title("DateTime: {}".format(name))
        fig.suptitle("Determine direct and diffuse parts.", fontsize=16)
        ax1 = plt.subplot(221)  # 2=rows, 2=cols, 1=pic number
        ax1.set_title("original image")
        img = np.array(i1, dtype=np.uint8)
        plt.imshow(img,'gray') # 'orig'

        ax2 = plt.subplot(222)  # 2=rows, 2=cols, 1=pic number
        ax2.set_title("saturated image")
        img = np.array(i2, dtype=np.uint8)
        plt.imshow(img,'gray')

        ax3 = plt.subplot(223)
        ax3.set_title("image close")
        img = np.array(i3, dtype=np.uint8)
        plt.imshow(img,'gray')

        ax4 = plt.subplot(224)
        ax4.set_title("image erode")
        img = np.array(i4, dtype=np.uint8)
        plt.imshow(img,'gray')

        plt.draw()
        while True:
            if plt.waitforbuttonpress():
                break
        plt.close(fig)

    except Exception as e:
        print('showAsSubplots: ' + str(e))

def showFinalSubplots(img_bgr,roi,name,lumDir,lumDiff):

    fig = plt.figure(2)
    plt.gcf().canvas.set_window_title("DateTime: {}".format(name))
    fig.suptitle("direct: {}, diffuse: {}".format(round(lumDir, 2), round(lumDiff, 2)), fontsize=12)
    ax1 = plt.subplot(121)  # 1=rows, 2=cols, 1=pic number
    ax1.set_title("Eroded with convex hul")
    img = np.array(img_bgr, dtype=np.uint8)
    plt.imshow(img)  # 'orig'

    ax2 = plt.subplot(122)  # 1=rows, 2=cols, 1=pic number
    ax2.set_title("Zoomed in")
    img = np.array(roi, dtype=np.uint8)
    plt.imshow(img)

    plt.draw()
    while True:
        if plt.waitforbuttonpress():
            break
    plt.close(fig)

def showFinalPlot(img_bgr,name,lumDir,lumDiff):
    fig = plt.figure(3)
    plt.gcf().canvas.set_window_title("DateTime: {}".format(name))
    fig.suptitle("direct: {}, diffuse: {}".format(round(lumDir, 2), round(lumDiff, 2)), fontsize=12)
    plt.imshow(img_bgr)
    plt.draw()
    while True:
        if plt.waitforbuttonpress():
            break
    plt.close(fig)

def showSurfacePlot(roi,name):
    fig, ax = plt.subplots(nrows=1, ncols=1, subplot_kw={'projection': '3d'})
    plt.gcf().canvas.set_window_title("DateTime: {}".format(name))
    fig.suptitle("Surface plot of ROI", fontsize=12)

    xx, yy = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]
    ax.plot_surface(xx, yy, (roi), rstride=1, cstride=1, cmap=cm.RdYlBu, linewidth=10, shade=True)  # RdYlBu

    fig.tight_layout()
    plt.draw()
    while True:
        if plt.waitforbuttonpress():
            break
    plt.close(fig)

def analyse(list_of_dirs):
    try:
        output_path = join(Path_to_source.rstrip('\\temp'),'dirdiff')
        file_name = getImageName()
        erodSize = 20

        for dir in  list_of_dirs:
            name, sw, id = strip_name_swvers_camid(dir)
            img = cv2.imread(join(dir,file_name),0)
            disk_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            disk_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erodSize, erodSize))

            imgSat = img[:,:] == 255
            imgSat = np.array(imgSat, dtype=np.uint8)

            imgClose = cv2.morphologyEx(imgSat, cv2.MORPH_CLOSE, disk_c)
            imgErod = cv2.dilate(imgClose,disk_e)

            #showAsSubplots(img,imgSat,imgClose,imgErod,name)

            Ind = cv2.findNonZero(imgErod)
            #lumDir = np.mean(img[Ind])
            lumDir = np.mean(img[imgErod>0])
            print('lumDir: {}'.format(lumDir))

            imgDiff = ~imgErod & img >0
            imgDiff = np.array(imgDiff, dtype=np.uint8)
            lumDiff = np.mean(img[imgDiff > 0])
            print('lumDiff: {}'.format(lumDiff))

            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) #imgErod*255

            image, contours, hierarchy = cv2.findContours(imgErod, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            b_cnt = max(contours, key=cv2.contourArea) # find biggest contour
            cv2.drawContours(img_bgr, b_cnt, -1, (0, 0, 255), 15)
            hull = cv2.convexHull(b_cnt)
            cv2.drawContours(img_bgr, hull, -1, (255, 0, 0), 20)

            contours_poly = cv2.approxPolyDP(hull, 3, True)
            bRect = cv2.boundingRect(contours_poly)
            cv2.rectangle(img_bgr, (int(bRect[0]), int(bRect[1])), \
                         (int(bRect[0] + bRect[2]), int(bRect[1] + bRect[3])), (0, 255, 0), 10)

            delta = 20
            x_1 = int(bRect[0])
            y_1 = int(bRect[1])
            x_2 = int(bRect[0] + bRect[2])
            y_2 = int(bRect[1] + bRect[3])
            roi = img[y_1 :y_2, x_1 :x_2]
            roi_zoom = img_bgr[y_1-delta:y_2+delta, x_1-delta:x_2+delta]

            #print('Number of found contours: {}'.format(len(contours)))
            #showFinalSubplots(img_bgr, roi_zoom, name, lumDir, lumDiff)

            showFinalPlot(img_bgr,name,lumDir,lumDiff)

            #showSurfacePlot(roi,name)

    except Exception as e:
        print('MAIN: Error in main: ' + str(e))

def main():
    try:

        list_of_dirs = getDirectories(Path_to_source)
        analyse(list_of_dirs)


    except Exception as e:
        print('MAIN: Error in main: ' + str(e))


if __name__ == '__main__':
    main()