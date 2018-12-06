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
from datetime import datetime
import pandas as pd

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

def strip_date_and_time(newdatetimestr):
    try:
        formated_date = '0000-00-00'
        formated_time = datetime.now().strftime('%H:%M:%S')

        date = (newdatetimestr.rstrip('\\').rpartition('\\')[-1]).rpartition('_')[0]
        time = (newdatetimestr.rstrip('\\').rpartition('\\')[-1]).rpartition('_')[-1]

        year  = date[:4]
        month = date[4:6]
        day   = date[6:8]
        hour  = time[:2]
        min   = time[2:4]
        sec   = time[4:6]

        check = [year,month,day,hour,min,sec]

        for item in check:
            if not item or not item.isdigit():
                return formated_date, formated_time

        formated_date = '{}-{}-{}'.format(year,month,day)
        formated_time = '{}:{}:{}'.format(hour,min,sec)

        return formated_date, formated_time
    except Exception as e:
        return formated_date, formated_time

def strip_date(path):
    try:
        H_S = datetime.now().strftime('%M-%S')
        formated_date = '0000-{}'.format(H_S)

        temp = path.rstrip('\\temp')
        temp = (temp.rpartition('\\'))[-1]
        temp = temp.replace('_',' ')
        dateAndTime = temp

        year = dateAndTime[:4]
        month = dateAndTime[4:6]
        day = dateAndTime[6:8]

        check = [year,month,day]

        for item in check:
            if not item or not item.isdigit():
                logger.error('strip_date: {} could not read date and time  used {} !'.format(path, formated_date))
                return formated_date

        formated_date = '{}-{}-{}'.format(year, month, day)

        return formated_date
    except IOError as e:
        return formated_date

def getImageName():
    source_path = Path_to_source.rstrip('\\temp')
    dir_name, sw_vers, camera_ID = strip_name_swvers_camid(source_path)

    if sw_vers == 1 or sw_vers == 2:
        img_name = 'raw_img5.jpg'

    if sw_vers == 3:
        img_name = 'raw_img0.jpg'

    return img_name

def showAsSubplots(i1,i2,i3,i4, name='' ,keybrake=None):
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

        if keybrake:
            while True:
                plt.draw()
                if plt.waitforbuttonpress():
                    break
            plt.close(fig)
        else:
            plt.draw()

    except Exception as e:
        print('showAsSubplots: ' + str(e))

def showFinalSubplots(img_bgr,roi,name,lumDir,lumDiff, keybrake=None):

    ratio = round(lumDir/lumDiff,2)
    dir = round(lumDir, 2)
    dif = round(lumDiff, 2)

    fig = plt.figure(2)
    plt.gcf().canvas.set_window_title("DateTime: {}".format(name))
    fig.suptitle("direct: {}, diffuse: {}, ratio: {}".format(dir,dif,ratio), fontsize=12)
    ax1 = plt.subplot(121)  # 1=rows, 2=cols, 1=pic number
    ax1.set_title("Eroded with convex hul")
    img = np.array(img_bgr, dtype=np.uint8)
    plt.imshow(img)  # 'orig'

    ax2 = plt.subplot(122)  # 1=rows, 2=cols, 1=pic number
    ax2.set_title("Zoomed in")
    img = np.array(roi, dtype=np.uint8)
    plt.imshow(img)

    if keybrake:
        plt.draw()
        while True:
            if plt.waitforbuttonpress():
                break
        plt.close(fig)
    else:
        plt.draw()

def showFinalPlot(img_bgr,name,lumDir,lumDiff, keybrake=None):
    fig = plt.figure(3)
    plt.gcf().canvas.set_window_title("DateTime: {}".format(name))
    fig.suptitle("direct: {}, diffuse: {}".format(round(lumDir, 2), round(lumDiff, 2)), fontsize=12)

    if keybrake:
        plt.imshow(img_bgr)
        plt.draw()
        while True:
            if plt.waitforbuttonpress():
                break
        plt.close(fig)

    else:
        plt.imshow(img_bgr)
        plt.draw()

def showSurfacePlot(roi,name, keybrake=None):
    fig, ax = plt.subplots(nrows=1, ncols=1, subplot_kw={'projection': '3d'})
    plt.gcf().canvas.set_window_title("DateTime: {}".format(name))
    fig.suptitle("Surface plot of ROI", fontsize=12)
    xx, yy = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]

    ax.view_init(elev=50, azim=-30)

    if keybrake:
        ax.plot_surface(xx, yy, (roi), rstride=1, cstride=1, cmap=cm.RdYlBu, linewidth=10, shade=True)  # RdYlBu
        fig.tight_layout()
        plt.draw()
        while True:
            if plt.waitforbuttonpress():
                break
        plt.close(fig)
    else:
        ax.plot_surface(xx, yy, (roi), rstride=1, cstride=1, cmap=cm.RdYlBu, linewidth=10, shade=True)  # RdYlBu
        fig.tight_layout()
        plt.draw()

def plot_ratio(csv_name, keybrake=None):

    df = pd.read_csv(csv_name)
    ratio_s = df['ratio']
    print(ratio_s.head())

    fig = plt.figure(4)
    plt.gcf().canvas.set_window_title("Ratio direct to diffuse luminance")
    fig.suptitle("Ratio direct to diffuse luminance", fontsize=12)

    if keybrake:
        ratio_s.plot()
        while True:
            if plt.waitforbuttonpress():
                break
        plt.close(fig)
    else:
        ratio_s.plot()
        plt.show()

def analyse(list_of_dirs, showplot=None):
    try:
        output_path = join(Path_to_source.rstrip('\\temp'),'dirdiff')
        file_name = getImageName()
        ratio_list=[]
        lumDir_list = []
        lumDiff_list = []
        tot_number = len(list_of_dirs)

        csv_name = Path_to_source.rstrip('\\temp').rpartition('\\')[-1]
        csv_name = csv_name.replace('raw_','')
        csv_name = csv_name + "_dirdiff.csv"

        cnt = 0
        erodSize = 30

        for dir in  list_of_dirs:
            name, sw, id = strip_name_swvers_camid(dir)
            date, time = strip_date_and_time(dir)
            img = cv2.imread(join(dir,file_name),0)
            disk_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            disk_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erodSize, erodSize))

            imgSat = img[:,:] == 255
            imgSat = np.array(imgSat, dtype=np.uint8)

            imgClose = cv2.morphologyEx(imgSat, cv2.MORPH_CLOSE, disk_c)
            imgErod = cv2.dilate(imgClose,disk_e)

            #if showplot:
                #showAsSubplots(img,imgSat,imgClose,imgErod,name, keybrake=True)

            Ind = cv2.findNonZero(imgErod)
            #lumDir = np.mean(img[Ind])
            lumDir = np.mean(img[imgErod>0])
            imgDiff = ~imgErod & img >0
            imgDiff = np.array(imgDiff, dtype=np.uint8)
            lumDiff = np.mean(img[imgDiff > 0])

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
            if showplot:
                showFinalSubplots(img_bgr, roi_zoom, name, lumDir, lumDiff,keybrake=True)
                #showFinalPlot(img_bgr,name,lumDir,lumDiff,keybrake=True)
                #showSurfacePlot(roi,name, keybrake=True)

            cnt +=1

            lumDir_list.append(lumDir)
            lumDiff_list.append(lumDiff)

            ratio = lumDir/lumDiff
            ratio_list.append(ratio)
            print('{}: {} : lumDir: {}, lumDiff: {}, ratio: {}'\
                  .format((tot_number - cnt), time, round(lumDir, 2), round(lumDiff,2),round(ratio,4)))

        print('Done calculating direct/diffuse ratio.')
        print('len ratio: {}'.format(len(ratio_list)))
        print('len direct: {}'.format(len(lumDir_list)))
        print('len diffuse: {}'.format(len(lumDiff_list)))

        df = pd.DataFrame.from_dict({'ratio':ratio_list,'direct':lumDir_list,'diffuse':lumDiff_list})
        df.to_csv(csv_name,index=False)

        return csv_name

    except Exception as e:
        print('Error in analyse: ' + str(e))

def main():
    try:
        list_of_dirs = getDirectories(Path_to_source)
        #csv_name = analyse(list_of_dirs,showplot= False)
        csv_name = r'20180308_cam1_dirdiff.csv'
        plot_ratio(csv_name, keybrake=True)


    except Exception as e:
        print('MAIN: Error in main: ' + str(e))


if __name__ == '__main__':
    main()