#!/usr/bin/env python

from __future__ import print_function

import os
import cv2
import sys
import subprocess
import zipfile
from glob import glob, glob1
from os.path import isfile, join
from datetime import datetime
global Path_imgs
global Avoid_This_Directories
global Path_to_ffmpeg
global Path_to_sourceDir
Path_to_sourceDir = r'\\192.168.1.8\SkyCam_FTP\camera_1\cam_1_vers1\20180131_raw_cam1'
Path_to_ffmpeg = r'C:\ffmpeg\bin\ffmpeg.exe'
Path_imgs = ''

Avoid_This_Directories = ['imgs5','hdr','rest']

######################################################################
## Hoa: 10.11.2018 Version 1 : makeHDR_video.py
######################################################################
#
# Collects all hdr_jpg.jpg images in the directory hdr
# These images are than combined by ffmpeg to a video
######################################################################


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

def getZipDirs(pathToDirectories):
    allZipFiles = []
    cnt = 0

    for zipfile in sorted(glob(os.path.join(pathToDirectories, "*.zip"))):
        if os.path.isfile(zipfile):
            allZipFiles.append(zipfile)
            cnt +=1

    if cnt <= 0:
        print('MISSING DATA in: {}'.format(pathToDirectories))

    return allZipFiles

def unzipall(path_to_extract):
    try:
        temp_path = join(path_to_extract, 'temp')

        if not os.path.exists(temp_path):
            os.makedirs(temp_path)

        allzipDirs = getZipDirs(path_to_extract)

        for dir in allzipDirs:
            path = dir.lower().replace('.zip', '')
            dirName = (path.rstrip('\\').rpartition('\\')[-1])
            new_temp_path = join(temp_path, dirName)

            if path:
                if not os.path.exists(new_temp_path):
                    zipfilepath = os.path.join(dir)
                    zf = zipfile.ZipFile(zipfilepath, "r")
                    zf.extractall(os.path.join(new_temp_path))
                    zf.close()

        return temp_path

    except Exception as e:
        print('unzipall: ' + str(e))

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

def check_for_missing_data(path_img):
    try:
        all_dirs = getDirectories(path_img)
        dir_to_del = []
        error_msgs = []

        for dir in all_dirs:
            data_ok = True
            dir_name, sw_vers, camera_ID = strip_name_swvers_camid(dir)
            date, time = strip_date_and_time(dir_name)

            datCnt = len(glob1(dir, "*.data"))
            jpgCnt = len(glob1(dir, "*.jpg"))
            logCnt = len(glob1(dir, "*.log"))

            if sw_vers == 1:
                if datCnt < 9 or jpgCnt < 9 or not (logCnt == 1):
                    data_ok = False
            else:
                if datCnt < 3 or jpgCnt < 3 or not (logCnt == 1):
                    data_ok = False

            if not data_ok:
                dir_to_del.append(dir.rstrip('\\'))
                msg = 'Invalid data in {} {} found, must be removed.'.format(date, time)
                error_msgs.append(msg)


        if len(dir_to_del) == 0:
            print('Data integrity ok.')
        else:
            for msg in error_msgs:
                print(msg)
            sys.exit()

        '''
        for dir in dir_to_del:
            os.remove(dir)
        '''
    except Exception as e:
        print('Error in check_for_missing_data: ' + str(e))

def create_Video():
    try:
        print('Creating video.')
        path_to_imgs = ''
        if os.path.exists(join(Path_to_sourceDir, 'imgs')):
            path_to_imgs  = join(Path_to_sourceDir, 'imgs')
        if os.path.exists(join(Path_to_sourceDir, 'hdr')):
            path_to_imgs = join(Path_to_sourceDir, 'hdr')

        dir_name, sw_vers, camera_ID = strip_name_swvers_camid(Path_to_sourceDir)
        video_name = '\\'+dir_name + '_HDR_video.mp4 '
        global Path_to_ffmpeg                                 # path to ffmpeg executable
        fsp = ' -r 10 '                                       # frame per sec images taken
        stnb = '-start_number 0001 '                          # what image to start at
        imgpath = '-i ' + join(path_to_imgs,'%4d.jpg ')  # path to images
        res = '-s 2592x1944 '                                 # output resolution
        outpath = Path_to_sourceDir + video_name              # output file name
        codec = '-vcodec libx264'                             # codec to use

        command = Path_to_ffmpeg + fsp + stnb + imgpath + res + outpath + codec

        if sys.platform == "linux":
            subprocess(command, shell=True)
        else:
            print(' {}'.format(command))
            ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            out, err = ffmpeg.communicate()
            if (err): print(err)
            print('ffmpeg ldr Video done.')

    except Exception as e:
        print('createVideo: Error: ' + str(e))

def write2img(input_image,text, xy):

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = xy
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 3

    output_image = cv2.putText(input_image, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

    return output_image

def copyAndMaskAll_HDR_imgs(list_alldirs):

    try:
        global Path_imgs
        cnt = 1
        Path_imgs = 'hdr'
        Path_to_copy_hdrs = join(Path_to_sourceDir,'hdr')

        print('Collecting images.')

        if not os.path.exists(Path_to_copy_hdrs):
            os.makedirs(Path_to_copy_hdrs)

        for next_dir in list_alldirs:
            dir = join(next_dir,'output')
            newimg = join(dir,'hdr_jpg.jpg')
            masked_img = cv2.imread(newimg)
            dir_name, sw_vers, camera_ID = strip_name_swvers_camid(next_dir)
            dateAndTime = (next_dir.rstrip('\\').rpartition('\\')[-1]).replace('_',' ')
            prefix = '{0:04d}'.format(cnt)
            year  = dateAndTime[:4]
            month = dateAndTime[4:6]
            day   = dateAndTime[6:8]
            hour  = dateAndTime[9:11]
            min   = dateAndTime[11:13]
            sec   = dateAndTime[13:15]

            img_txt = write2img(masked_img,'cam '+str(camera_ID) +' sw '+ str(sw_vers), (10, 30))
            img_txt = write2img(masked_img, year+" "+month+" "+day,(10,60))
            img_txt = write2img(masked_img, '#: ' + str(cnt), (10,590))
            img_txt = write2img(masked_img, hour+":"+min+":"+sec, (10, 635))
            new_img_path = join(Path_to_copy_hdrs, '{}.jpg'.format(prefix))

            cv2.imwrite(new_img_path, masked_img)
            cnt += 1

        print('HDR {} images copyied to hdr.'.format(cnt))

    except Exception as e:
        print('Error in copyAndMaskAll_HDR_imgs: {}'.format(e))

def copyAndMaskAll_imgs(list_alldirs):
    try:
        cnt = 1
        global Path_imgs
        Path_imgs = "imgs"
        Path_to_copy_imgs = join(Path_to_sourceDir, "imgs")
        dir_name, sw_vers, camera_ID = strip_name_swvers_camid(Path_to_sourceDir)
        img_to_get = ''

        print('Collecting images.')

        if sw_vers == 1 or sw_vers == 2:
            img_to_get = 'raw_img5.jpg'

        if sw_vers == 3:
            img_to_get = 'raw_img0.jpg'

        if not os.path.exists(Path_to_copy_imgs):
            os.makedirs(Path_to_copy_imgs)

        for next_dir in list_alldirs:
            newimg = join(next_dir,img_to_get)
            new_img = cv2.imread(newimg)
            masked_img = cv2.resize(new_img,(864,648))

            dir_name, sw_vers, camera_ID = strip_name_swvers_camid(next_dir)
            dateAndTime = (next_dir.rstrip('\\').rpartition('\\')[-1]).replace('_',' ')
            prefix = '{0:04d}'.format(cnt)
            year  = dateAndTime[:4]
            month = dateAndTime[4:6]
            day   = dateAndTime[6:8]
            hour  = dateAndTime[9:11]
            min   = dateAndTime[11:13]
            sec   = dateAndTime[13:15]

            img_txt = write2img(masked_img,'cam '+str(camera_ID) +' sw '+ str(sw_vers), (10, 30))
            img_txt = write2img(masked_img, year+" "+month+" "+day,(10,60))
            img_txt = write2img(masked_img, '#: ' + str(cnt), (10,590))
            img_txt = write2img(masked_img, hour+":"+min+":"+sec, (10, 635))
            new_img_path = join(Path_to_copy_imgs, '{}.jpg'.format(prefix))

            cv2.imwrite(new_img_path, masked_img)
            cnt += 1

        print(' {} images copyied to hdr.'.format(cnt))

    except Exception as e:
        print('Error in copyAndMaskAll_HDR_imgs: {}'.format(e))

def getDirectories(pathToDirectories):
    try:
        global Avoid_This_Directories
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

def main():
    try:

        unzip = False
        check = True
        hdr_video = False
        regular_video = True

        print('Start make video.')

        if unzip:
            unzipall(Path_to_sourceDir)
            print('Unzip finished.')

        if check:
            check_for_missing_data(join(Path_to_sourceDir, 'temp'))

        allDirs = getDirectories(join(Path_to_sourceDir, 'temp'))

        if hdr_video:
            copyAndMaskAll_HDR_imgs(allDirs)
        if regular_video:
            copyAndMaskAll_imgs(allDirs)

        create_Video()
        print('All processes finished')

    except Exception as e:
        print('MAIN: Error in main: ' + str(e))


if __name__ == '__main__':
    main()