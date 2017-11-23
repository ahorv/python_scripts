import cv2
import math
import numpy as np


global images_in
images_in = r'C:\Hoa_Python_Projects\python_scripts\hdr\input'

def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()

def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix

def simplest_cb(img, percent):
    print('RGB- Image shape: {}'.format(img.shape))
    assert img.shape[2] == 3
    assert percent > 0 and percent < 100

    half_percent = percent / 200.0

    channels = cv2.split(img)

    out_channels = []
    for channel in channels:
        assert len(channel.shape) == 2
        # find the low and high precentile values (based on the input percentile)
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)

        assert len(flat.shape) == 1

        flat = np.sort(flat)

        n_cols = flat.shape[0]

        low_val  = flat[math.floor(n_cols * half_percent)]
        high_val = flat[math.ceil( n_cols * (1.0 - half_percent))]

        print("Lowval: {}".format(low_val))
        print ("Highval: {} ".format( high_val))

        # saturate below the low percentile and above the high percentile
        thresholded = apply_threshold(channel, low_val, high_val)
        # scale the channel
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    return cv2.merge(out_channels)

def main():
    try:
        global images_in

        #raw_img = np.fromfile(images_in + '/data.rgb', dtype='uint16')  # uint16

        imrows = 2464
        imcols = 3296

        imsize = imrows * imcols
        with open(images_in + '/cv_data.rgb', "rb") as rawimage:
            raw_img = np.fromfile(rawimage, np.dtype('u2'), imsize).reshape((imrows, imcols))
            print('raw_img shape: {}'.format(raw_img.shape))

        out = simplest_cb(raw_img, 1)
        cv2.imshow("before", raw_img)
        cv2.imshow("after", out)
        cv2.waitKey(0)

    except Exception as e:
        print('Error in Main: ' + str(e))


if __name__ == '__main__':
    main()