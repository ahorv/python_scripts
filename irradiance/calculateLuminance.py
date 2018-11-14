import numpy as np
import os
from os.path import join
import cv2
import pandas as pd
from glob import glob
from fractions import Fraction
from skimage import draw


######################################################################
## Hoa: 05.11.2018 Version 1 : LuminanceSquareCrop.py
######################################################################
# Source : https://github.com/Soumyabrata/solar-irradiance-estimation
#
# New /Changes:
# ----------------------------------------------------------------------
#
# 05.11.2018 : first add
#
######################################################################

print('Started Luminance')

#path_img = r'C:\Users\ati\Desktop\teil_20181012\camera_1\cam_1_vers3\20181012_raw_cam1\temp' # camera1
path_img = r'C:\Users\ati\Desktop\29181012_camera2\camera_2\cam_2_vers3\29181012_raw_cam2\temp' # camera2

def calculate_sun_centre(path_to_img):
    complete_x = []
    complete_y = []
    all_images = []
    im = cv2.imread(path_to_img)

    # Finding the centroid of sun position polygon
    threshold_value = 240
    red = im[:, :, 2]
    green = im[:, :, 1]
    blue = im[:, :, 0]
    all_coord = np.where(blue > threshold_value)
    all_coord = np.asarray(all_coord)
    length = np.shape(all_coord)[1]
    sum_x = np.sum(all_coord[0, :])
    sum_y = np.sum(all_coord[1, :])

    if (sum_x == 0 or sum_y == 0):
        centroid_x = np.nan
        centroid_y = np.nan
    else:
        centroid_x = int(sum_x / length)
        centroid_y = int(sum_y / length)

    #interpolate the sun's location in the missing places
    s1 = pd.Series(centroid_x)
    s2 = pd.Series(centroid_y)

    complete_x = s1.interpolate()
    complete_y = s2.interpolate()

    # Initially all computed values are NaN
    if (np.isnan(complete_x).any()) or (np.isnan(complete_y).any()):
        # print ('Skipping for ', match_string)
        complete_x = np.array([])
        complete_y = np.array([])
    else:
        # Replacing NaN s in the beginning with closest non-NaN value
        a = complete_x
        ind = np.where(~np.isnan(a))[0]
        first, last = ind[0], ind[-1]
        a[:first] = a[first]
        a[last + 1:] = a[last]

        # For y co-ordinate
        a = complete_y
        ind = np.where(~np.isnan(a))[0]
        first, last = ind[0], ind[-1]
        a[:first] = a[first]
        a[last + 1:] = a[last]

    print('calculated all x/y= {}/{} values {}/{}'.format(centroid_x, centroid_y,complete_x[0], complete_y[0]))
    return (centroid_x, centroid_y, complete_x, complete_y)

def cmask(index, radius, array):
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

def LuminanceSquareCrop(LDR_path, sun_x, sun_y, crop_dim = 300):
    try:
        # LDR images
        image_path1 = LDR_path
        print('Calculating luminance of: ', image_path1)
        f1 = open(image_path1, 'rb')
        im1 = cv2.imread(image_path1)

        centroid_x = sun_x
        centroid_y = sun_y

        # Construct rectangle
        around_sun = im1[(int(centroid_x - crop_dim/2)):(int(centroid_x + crop_dim/2)),(int(centroid_y - crop_dim/2)):(int(centroid_y + crop_dim/2))]

        lum = 0.2126*around_sun[:,:,0] + 0.7152*around_sun[:,:,1] + 0.0722*around_sun[:,:,2]
        lum = np.mean(lum)

        LDRLuminance = lum/exp_time1
        #print("Calculated luminance: " + LDRLuminance)
        print('End sol')

    except Exception as e:
        print("Error in sol: " + str(e))

    return(LDRLuminance)

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

def getNumpyArray(path, type=None):
    try:
        data = np.fromfile(path, dtype='uint16')
        img = None

        if type is 'jpg':
            image_arr = np.frombuffer(data, dtype=np.float32)
            img = image_arr.reshape(1944, 2592, 3)
        if type is 'data':
            image_arr = np.frombuffer(data, dtype=np.float32)
            img = image_arr.reshape(1232, 1648, 3)

        return img

    except Exception as e:
       print('Error getNumpyArDB: {}'.format(e))

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

def mask_array(data, cam='vers1', show_mask=False ):

    masked_img=None

    w = data.shape[0]
    h = data.shape[1]
    c = data.shape[2]

    if cam == 'vers1':
        centre = [505,746] # [y,x] !
        radius = 680
        masked_img = maske_circle(data, [w, h, c], centre, radius, show_mask)
    if cam == 'vers2':
        centre = [620,885]
        radius = 680
        corner = [0,520]
        dimension = [0,100]
        #masked_img = maske_circle(data, [w, h, c], centre, radius, show_mask)
        masked_img = maske_rectangel(data, [w, h, c], corner, dimension, show_mask)

    return masked_img


def main():
    try:
        global path_img

        all_dirs = getDirectories(path_img)
        # ok durchläuft alle dirs
        # nun jeweils aus output imgs holen
        # aus log data die shutter zeiten parsen
        # resultate in csv liste
        for dir in all_dirs:
            listOfSS, _ = getShutterTimes(dir)
            print(' {}  SS:{}'.format(dir,listOfSS))

            arr_hdr = getNumpyArray(join(dir,'output','hdr_data.dat'),'data')
            hdr = mask_array(arr_hdr,'vers2',True)
            show_image('hdr', hdr)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break


            print('min: {}'.format(arr_hdr.min()))
            print('{}'.format(join(dir,'output','hdr_data.dat')))

            #arr_jpg = getNumpyArray(join('output', 'hdr_jpg.dat'))


    except Exception as e:
       print('MAIN: {}'.format(e))


if __name__ == '__main__':
    main()