import numpy as np
import exifread
import cv2
import pandas as pd
from scipy import ndimage

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

path_img = r'C:\Users\ati\Desktop\LuminanceHDR-master\sky_2\output_hdr\sat_masked_hdr.jpg'

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


def main():
    try:
        global path_img

        centroid_x, centroid_y, complete_x, complete_y = calculate_sun_centre(path_img)
        img = cv2.imread(path_img)
        cv2.circle(img, (centroid_y, centroid_x), 30, (0,0,255), thickness=10, lineType=8, shift=0)

        w,h,d = img.shape
        img_s = cv2.resize(img, (int(h/3), int(w/3)))
        cv2.imshow('centre of sun',img_s)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print('All processes finished.')

    except Exception as e:
       print('MAIN: {}'.format(e))


if __name__ == '__main__':
    main()