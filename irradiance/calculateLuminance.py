import numpy as np
import os
from os.path import join
import cv2
import pandas as pd
from glob import glob
from fractions import Fraction
from skimage import draw


######################################################################
## Hoa: 05.11.2018 Version 1 : calculateLuminance.py
######################################################################
# Source : https://github.com/Soumyabrata/solar-irradiance-estimation
#
# New /Changes:
# ----------------------------------------------------------------------
#
# 05.11.2018 : first add
#
######################################################################

#path_img = r'C:\Users\ati\Desktop\teil_20181012\camera_1\cam_1_vers3\20181012_raw_cam1\temp' # camera1
path_img = r'C:\Users\ati\Desktop\29181012_camera2\camera_2\cam_2_vers3\29181012_raw_cam2\temp' # camera2

def calculate_sun_centre(LDR_low, show_sun=False):
    """
    usage:
        centroid_x, centroid_y, complete_x, complete_y = calculate_sun_centre(path_img)
        img = cv2.imread(path_img)
        cv2.circle(img, (centroid_y, centroid_x), 30, (0,0,255), thickness=10, lineType=8, shift=0)

        w,h,d = img.shape
        img_s = cv2.resize(img, (int(h/3), int(w/3)))
        cv2.imshow('centre of sun',img_s)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    :param img:
    :return:
    """

    try:
        # Finding the centroid of sun position polygon
        threshold_value = 240
        red = LDR_low[:, :, 2]
        green = LDR_low[:, :, 1]
        blue = LDR_low[:, :, 0]
        all_coord = np.where(blue > threshold_value)
        all_coord = np.asarray(all_coord)
        length = np.shape(all_coord)[1]
        sum_x = np.sum(all_coord[0, :])
        sum_y = np.sum(all_coord[1, :])

        if (sum_x == 0 or sum_y == 0):
            centroid_x = 0
            centroid_y = 0
        else:
            centroid_x = int(sum_x / length)
            centroid_y = int(sum_y / length)

        #interpolate the sun's location in the missing places (sobald die daten aus dem csv file gelesen werden)
        # siehe im Orginal 'sun_positions_day_files'

        if show_sun:
            cv2.circle(LDR_low, (centroid_y, centroid_x,), 30, (0, 0, 255), thickness=10, lineType=8, shift=0)

        #print('Sun\'s centre: x/y= {}/{} '.format(str(centroid_x), str(centroid_y)))
        return centroid_x, centroid_y
    except Exception as e:
        return 0, 0
        print("Error in sol: " + str(e))

def getDirectories(path_to_dirs):
    try:
        allDirs = []
        img_cnt = 1

        for dirs in sorted(glob(join(path_to_dirs, "*", ""))):
            if os.path.isdir(dirs):
                if dirs.rstrip('\\').rpartition('\\')[-1]:
                    allDirs.append(dirs.rstrip('\\'))
                    img_cnt +=1
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

def strip_img_type(path):
    type = ''
    temp = path.rpartition('\\')[-1]
    temp = temp.rpartition('_')[-1]
    type = temp.rpartition('.')[0]

    return str(type)

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

    return (dir_name, sw_vers, camera_ID)

def getShutterTimes(path):
        try:
            '''
            returns shutter_time in microseconds as np.float32 type
            '''
            dir_name, sw_vers, camera_ID = strip_name_swvers_camid(path)
            types = ('*.txt', '*.log')
            ss_to_db = []

            for typ in types:
                for file in sorted(glob(os.path.join(path,typ))):
                    logfile = file

            f = open(join(logfile), 'r')
            logfile = f.readlines()

            if sw_vers == 1:
                listOfSS = np.empty(10, dtype=np.float32)

                if os.stat(file).st_size == 0:
                    ss_to_db = '0,0,0,0,0,0,0,0,0,0'
                    print('Empty camstat - file: {}'.format(file))
                    return listOfSS, ss_to_db

                logfile.pop(0)  # remove non relevant lines
                logfile.pop(0)
                logfile.pop(0)
                pos = 0
                for line in logfile:
                    value = line.split("camera shutter speed:", 1)[1].replace('[','').replace(']','')
                    value = value.split('|', 1)[0]
                    value = value.strip()
                    ss_to_db.append(value + ",")
                    value += '/1000000'
                    val_float = np.float32(Fraction(str(value)))
                    listOfSS[pos] = val_float
                    pos +=1

            else :
                listOfSS = np.empty(3, dtype=np.float32)

                if os.stat(file).st_size == 0:
                    ss_to_db = '0,0,0'
                    print('Empty camstat - file: {}'.format(file))
                    return listOfSS, ss_to_db

                if sw_vers == 2:
                    logfile.pop(0)
                if sw_vers == 3:
                    logfile.pop(0)
                    logfile.pop(0)

                pos = 0
                for line in logfile:
                    value = line.split("ss:", 1)[1]
                    value = value.split(',', 1)[0]
                    value = value.strip()
                    ss_to_db.append(value + ",")
                    value += '/1000000'
                    val_float = np.float32(Fraction(str(value)))
                    listOfSS[pos] = val_float
                    pos +=1

            ss_to_db_str = ''.join(ss_to_db)

            return listOfSS, ss_to_db_str.rstrip(',')

        except Exception as e:
            print('Error in getShutterTimes: ' + str(e))

def getNumpyArray(path):
    try:
        data = np.fromfile(path, dtype='uint16')
        img = None

        dir_name, sw_vers, cam_ID = strip_name_swvers_camid(path)
        type = strip_img_type(path)

        if type == 'jpg':
            image_arr = np.frombuffer(data, dtype=np.float32)
            img = image_arr.reshape(1944, 2592, 3)
        if type == 'data':
            image_arr = np.frombuffer(data, dtype=np.float32)
            img = image_arr.reshape(1232, 1648, 3)

        return int(cam_ID), type, img

    except Exception as e:
       print('Error getNumpyArray: {}'.format(e))

def tonemap(hdr):
    hdr = np.float32(hdr)
    tonemapReinhard = cv2.createTonemapReinhard(1.5, 0, 0, 0)
    ldrReinhard = tonemapReinhard.process(hdr)
    return  ldrReinhard * 255

def show_image(title, hdr):
    img_tonemp = tonemap(hdr)
    img_8bit = cv2.normalize(img_tonemp, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    w, h, d = img_8bit.shape
    img_8bit_s = cv2.resize(img_8bit, (int(h / 2), int(w / 2)))
    cv2.imshow(title, img_8bit_s)

def cmask(index, radius, array):
    """Generates the mask for a given input image.
    The generated mask is needed to remove occlusions during post-processing steps.

    Args:
        index (numpy array): Array containing the x- and y- co-ordinate of the center of the circular mask.
        radius (float): Radius of the circular mask.
        array (numpy array): Input sky/cloud image for which the mask is generated.

    Returns:
        numpy array: Generated mask image."""

    a, b = index
    is_rgb = len(array.shape)

    if is_rgb == 3:
        ash = array.shape
        nx = ash[0]
        ny = ash[1]
    else:
        nx, ny = array.shape

    s = (nx, ny)
    image_mask = np.zeros(s)
    y, x = np.ogrid[-a:nx - a, -b:ny - b]
    mask = x * x + y * y <= radius * radius
    image_mask[mask] = 1

    return (image_mask)

def rectmask(corner, dimension, array):
    """Generates the mask for a given input image.
    The generated mask is needed to remove occlusions during post-processing steps.

    Args:
        index (numpy array): Array containing the x- and y- co-ordinate of the center of the circular mask.
        radius (float): Radius of the circular mask.
        array (numpy array): Input sky/cloud image for which the mask is generated.

    Returns:
        numpy array: Generated mask image."""

    w, h = dimension # width and height
    a, b = corner
    is_rgb = len(array.shape)

    if is_rgb == 3:
        ash = array.shape
        nx = ash[0]
        ny = ash[1]
    else:
        nx, ny = array.shape

    s = (nx, ny)
    image_mask = np.zeros(s)
    y, x = np.mgrid[-a:nx-a,-b:ny-b]
    mask = (x<a)&(x-a<=w)&(y>b)&(y-b<=h)
    image_mask[~mask] = 1

    return (image_mask)

def maske_circle(input_image, size=[0, 0, 3], centre=[0, 0], radius=0, show_mask=False):

    empty_img = np.zeros(size, dtype=np.uint8)
    mask = cmask(centre, radius, empty_img)

    red = input_image[:, :, 0]
    green = input_image[:, :, 1]
    blue = input_image[:, :, 2]

    if show_mask:
        h = input_image.shape[0]
        w = input_image.shape[1]

        for y in range(0,h):
            for x in range(0,w):
                if mask[y,x] == 0:
                    blue[y,x] = 65535
        b_img = blue
    else:
        b_img = blue.astype(float) * mask

    r_img = red.astype(float) * mask
    g_img = green.astype(float) * mask
    #b_img = blue.astype(float) * mask

    dimension = (input_image.shape[0], input_image.shape[1], 3)
    output_img = np.zeros(dimension, dtype=float)

    output_img[..., 0] = r_img[:, :]
    output_img[..., 1] = g_img[:, :]
    output_img[..., 2] = b_img[:, :]

    return output_img

def maske_rectangel(input_image, size=[0, 0, 3], corner=[0, 0], dim=[0, 0], show_mask=False):
    empty_img = np.zeros(size, dtype=np.uint8)
    mask = rectmask(corner, dim, empty_img)

    red = input_image[:, :, 0]
    green = input_image[:, :, 1]
    blue = input_image[:, :, 2]

    if show_mask:
        h = input_image.shape[0]
        w = input_image.shape[1]

        for y in range(0,h):
            for x in range(0,w):
                if mask[y,x] == 0:
                    green[y,x] = 65535
        g_img = green
    else:
        g_img = green.astype(float) * mask

    r_img = red.astype(float) * mask
    #g_img = green.astype(float) * mask
    b_img = blue.astype(float) * mask

    dimension = (input_image.shape[0], input_image.shape[1], 3)
    output_img = np.zeros(dimension, dtype=float)

    output_img[..., 0] = r_img[:, :]
    output_img[..., 1] = g_img[:, :]
    output_img[..., 2] = b_img[:, :]

    return output_img

def mask_array(data, cam_id='1', type='', show_mask=False ):

    masked_img=None

    w = data.shape[0]
    h = data.shape[1]
    c = data.shape[2]

    if cam_id == 1:
        if type == 'data':
            centre = [505,746] # [y,x] !
            radius = 680

        if type == 'jpg':
            centre = [795, 1190]  # [y,x] !
            radius = 1050

        masked_img = maske_circle(data, [w, h, c], centre, radius, show_mask)

    if cam_id == 2:
        if type == 'data':
            centre = [620,885]  # [y,x] !
            radius = 680
            corner = [0,520]
            dimension = [0,100]

        if type == 'jpg':
            centre = [1080, 1300]  # [y,x] !
            radius = 1065  #1065
            corner = [0, 822]
            dimension = [0, 168]

        masked_img = maske_circle(data, [w, h, c], centre, radius, show_mask)
        masked_img = maske_rectangel(masked_img, [w, h, c], corner, dimension, show_mask)

    return masked_img

def LuminanceSquareCrop(LDR_low_img, exp_low_time, sun_x, sun_y, crop_dim = 300, show_rect = False):
    try:
        centroid_x = sun_x
        centroid_y = sun_y

        # Construct rectangle
        around_sun = LDR_low_img[(int(centroid_x - crop_dim/2)):(int(centroid_x + crop_dim/2)),(int(centroid_y - crop_dim/2)):(int(centroid_y + crop_dim/2))]

        if show_rect:
            x_1 = int(centroid_x - crop_dim/2)
            y_1 = int(centroid_y - crop_dim/2)
            x_2 = int(centroid_x + crop_dim/2)
            y_2 = int(centroid_y + crop_dim/2)
            cv2.rectangle(LDR_low_img, ( x_1,y_1), (x_2, y_2 ), (255, 255, 255), 3)

        lum = 0.2126*around_sun[:,:,0] + 0.7152*around_sun[:,:,1] + 0.0722*around_sun[:,:,2]
        lum = np.mean(lum)

        LDRLuminance = lum/exp_low_time

    except Exception as e:
        print("Error in sol: " + str(e))

    return LDRLuminance

def main():
    try:
        global path_img
        print('Started Luminance')
        all_dirs = getDirectories(path_img)

        for dir in all_dirs:

            listOfSS, _ = getShutterTimes(dir)
            exp_low_time = listOfSS[2]
            LDR_low = cv2.imread(join(dir,'raw_img-4.jpg'))
            sun_y, sun_x  = calculate_sun_centre(LDR_low, True)
            lum_sqrcrpt = LuminanceSquareCrop(LDR_low, exp_low_time, sun_x, sun_y,300, True)

            print('lum_sqrcrpt = {}'.format(lum_sqrcrpt))

            w, h, d = LDR_low.shape
            img_s = cv2.resize(LDR_low, (int(h / 3), int(w / 3)))
            cv2.imshow('centre of sun', img_s)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            return
            ##############################################

            # Calculate luminance from raw  HDR
            cam_id, type, arr_hdr = getNumpyArray(join(dir,'output','hdr_data.dat'))
            lum_hdr= np.mean(arr_hdr)
            print('mean hdr       : {}'.format(lum_hdr))

            arr_hdr_m = mask_array(arr_hdr, cam_id, type, False)
            lum_hdr_m = np.mean(arr_hdr_m)
            print('mean hdr masked: {}'.format(lum_hdr_m))


            # Calculate luminance from jpg HDR
            cam_id, type, arr_jpg = getNumpyArray(join(dir,'output','hdr_jpg.dat'))
            arr_jpg_m = mask_array(arr_jpg, cam_id, type, False)



            #show_image('hdr', arr_hdr_m)
            #show_image('original', arr_jpg_m)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            break

    except Exception as e:
       print('MAIN: {}'.format(e))


if __name__ == '__main__':
    main()