#!/usr/bin/python

import numpy as np
import sys
import os
from os import listdir
from os.path import isfile, join, splitext, basename
from scipy.misc import imread, imsave
from matplotlib.pyplot import plot, show, scatter, title, xlabel, ylabel, savefig
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from scipy.sparse.linalg import spsolve
import cv2
from fractions import Fraction

######################################################################
## Hoa: 22.11.2017 Version 1 : create_hdri.py
######################################################################
# Creates a HDR file from a numpy array
# Source: https://github.com/dipaco/create_hdri
#
# Test-version: Includes output of :
#               - tonemap.png
#               - response_curve.png
#               - image.hdr
#               - image.pfm (disabled)
#               - image.png
#
# New /Changes:
# ----------------------------------------------------------------------
#
# 22.11.2017 : New
#
#
######################################################################

APP_NAME = splitext(basename(sys.argv[0]))[0]
APP_PREFIX = '[' + APP_NAME + '] '

global input_path
global output_path
input_path = r'C:\Hoa_Python_Projects\python_scripts\hdr\input'  # @ home
output_path = r'C:\Hoa_Python_Projects\python_scripts\hdr\output'  # @ home
#input_path = r'C:\Users\tahorvat\PycharmProjects\python_scripts\hdr\input\20171025_140139'  # @ Lab

def read_images(path):
    '''Reads all the images in folder <path> (any common format), and convert every
        image into a numpy array. Every image is then stored in a dictionary in the
        following way:
        {
            exposure_time : numpy.array(rows, cols, 3),
            exposure_time : numpy.array(rows, cols, 3),
            ...
        }'''

    imrows = 2464
    imcols = 3296

    imsize = imrows * imcols

    exposure_times = [float(Fraction(369/1000000)),float(Fraction(463/1000000)),float(Fraction(577/1000000))]
    file_extensions = ["data"]
    exp_imgs_stack = dict()

    # Get all files in folder.
    image_files = sorted(os.listdir(path))
    # Remove files that do not have the appropriate extension.
    for img in image_files:
        if img.split(".")[-1].lower() not in file_extensions:
            image_files.remove(img)

    for n in range(0, len(image_files)):
        print('file: {}'.format(image_files[n]))

        with open(join(path,image_files[n]), "rb") as rawimage:
            # images[image_idx] = cv2.imread(path)
            img = np.fromfile(rawimage, np.dtype('u2'), imsize).reshape((imrows, imcols))
            db_rgb = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)
            #db_rgb = (db_rgb - np.min(db_rgb)) / (np.max(db_rgb) - np.min(db_rgb))  # normalize image
            exp_imgs_stack[exposure_times[n]] = db_rgb.astype(np.uint8)


    return exp_imgs_stack


def save_pfm(filename, image, scale=1):
    '''
    Save a Numpy array to a PFM file.
    '''
    color = None

    f = open(filename, "wb")
    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    f.write(b'PF\n' if color else 'Pf\n')
    f.write(b'%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    f.write('%f\n' % scale)

    image.tofile(f)

def get_samples(imgs_array, channel, num_points):
    '''Returns a matrix with intensity values (0-255) with many rows as sample points,
        and many columns as exposure times (num. of images)'''
    #Samples points
    img_shape = imgs_array[list(imgs_array)[0]].shape
    sp_x = np.random.randint(0, img_shape[0]-1, (num_points, 1))
    sp_y = np.random.randint(0, img_shape[1]-1, (num_points, 1))
    sp = np.concatenate((sp_x, sp_y), axis=1)

    n = len(sp)
    p = len(imgs_array)
    Z = np.zeros((n, p))
    B = np.zeros((p, 1))
    for i in range(0, n):
        j = 0
        for key in sorted(imgs_array):
            img = imgs_array[key][:, :, channel]
            row = sp[i, 0]
            col = sp[i, 1]
            Z[i, j] = img[row, col]
            B[j, 0] = key
            j += 1

    return Z, B

def fit_response(Z, B, l, w):
    '''Finds all values for the discrete log of exposure function, as well as the Radiance
        values (E) for each point'''

    #max num of intensity levels
    num_gray_levels = 256
    n = Z.shape[0]
    p = Z.shape[1]
    num_rows = n*p + num_gray_levels -2 + 1
    num_cols = num_gray_levels + n

    A = np.zeros((num_rows, num_cols))
    b = np.zeros((num_rows, 1))

    #Fill the coeficients in matrix A, according to the equation: G(z) = ln(E) + ln(dt)
    #multiplied by a weighting function (w)
    k = 0
    for j in range(0, p):
        for i in range(0, n):
            z_value = Z[i, j]
            w_value = w(z_value)
            A[int(k), int(z_value)] = w_value
            A[k, num_gray_levels + i] = -w_value
            b[k, 0] = w_value * np.log(B[j])
            k += 1

    #Setting the middle value of the G function as '0'
    A[k, 128] = 1
    k += 1

    #Add the smoothness constraints
    for i in range(1, num_gray_levels-1):
        w_value = w(i)
        A[k, i-1] = l*w_value
        A[k, i] = -2*l*w_value
        A[k, i+1] = l*w_value
        k += 1

    #Solve the equation system
    U, s, V = np.linalg.svd(A, full_matrices=False)
    m = np.dot(V.T, np.dot( np.linalg.inv(np.diag(s)), np.dot(U.T, b)))
    #m = np.linalg.lstsq(A, b)[0]

    return m[0:256], m[256:]

def write_hdr(filename, image):
    '''Writes a HDR image into disk. Assumes you have a np.array((height,width,3), dtype=float)
        as your HDR image'''
    try:
        f = open(filename, "wb")
        f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
        header = "-Y {0} +X {1}\n".format(image.shape[0], image.shape[1])
        f.write(bytes(header, encoding='utf-8'))

        print('HDR shape: ' + str(image.shape))
        print('HDR dtype: ' + str(image.dtype))
        print('HDR size: ' + str(image.size))

        brightest = np.maximum(np.maximum(image[...,0], image[...,1]), image[...,2])
        mantissa = np.zeros_like(brightest)
        exponent = np.zeros_like(brightest)
        np.frexp(brightest, mantissa, exponent)
        scaled_mantissa = mantissa * 256.0 / brightest
        rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        rgbe[..., 0:3] = np.around(image[..., 0:3] * scaled_mantissa[..., None])
        rgbe[..., 3] = np.around(exponent + 128)

        rgbe.flatten().tofile(f)
        f.close()
    except Exception as e:
        print('Error in: write_hdr: {} '.format(str(e)))

def create_radiance_map(imgs, G, w):
    '''Using the log. exposure function (G) create a radiance map
        for every channel in the image'''

    #returns a function to calculate the Radiance of a associated
    #to an intensity value
    def map_z_values(exposure_time):
        return np.vectorize(lambda z: G[z] - np.log(exposure_time))

    #vector form of weighting function
    get_w_values = np.vectorize(w)

    #Reduce noise by weighting the E values
    img_shape = imgs[list(imgs)[0]].shape
    R = np.zeros(img_shape)
    W = np.zeros(img_shape, dtype=float)
    for dt in imgs:
        print(APP_PREFIX + 'Processing image with dt = ', dt)
        W_aux = get_w_values(imgs[dt])
        R += W_aux * (map_z_values(dt))(imgs[dt]).reshape(img_shape)
        W += W_aux

    return R / W

def tonemap(R):
    '''
        convert R in range rmin to rmax to the range 0..240 degrees which
        correspond to the colors red..blue in the HSV colorspace
        lower values will have the value 0 and greater values will have
        the value 240. Then convert hsv color (h,1,1) to its rgb equivalent
        note: the hsv_to_rgb() function expects h to be in the range 0..1 not 0..360
    '''
    rmax = np.amax(R)
    rmin = np.amin(R)
    H = 240 * (rmax - R) / (rmax - rmin)
    TM_aux = np.ones(R.shape + (3,), dtype=float)
    TM_aux[:, :, 0] = H / 360

    return hsv_to_rgb(TM_aux)



# main
def main():
    try:
        where_Iam = ' '
        L_CHANNEL = 0
        a_CHANNEL = 1
        b_CHANNEL = 2

        global input_path
        global output_path

        print('CREATE_HDRI')

        # read images
        imgs_array = read_images(input_path)


        # sampling
        where_Iam = 'Sampling'
        channel = L_CHANNEL
        num_samples = int( 1500.0 / len(imgs_array.keys()))

        Z, B = get_samples(imgs_array, channel, num_samples)
        n, p = Z.shape

        # Fitting the curve
        where_Iam = 'Fitting Curve'
        Zmin = 0.0  # np.amin(Z)
        Zmax = 255.0  # np.amax(Z)
        w_hat = lambda z: z - Zmin + 1 if z <= (Zmin + Zmax) / 2 else Zmax - z + 1
        l = 550
        print(APP_PREFIX + 'Fitting log exposure curve...')
        G, E = fit_response(Z, B, l, w_hat)

        # creating radiance map for the channel
        where_Iam = 'Creating radiance map '
        print(APP_PREFIX + 'Creating radiance map (could take a while...)')
        relative_R = create_radiance_map(imgs_array, G, w_hat)
        R = np.exp(relative_R)

        tonemap_filename = join(output_path, 'tonemap.png')
        hdr_filename = join(output_path, 'create_hdri.hdr')
        pfm_filename = join(output_path, 'create_hdri.pfm')
        png_filename = join(output_path, 'create_hdri.png')
        print(APP_PREFIX + 'Saving HDR image to: ', hdr_filename)
        write_hdr(hdr_filename, R)
        print(APP_PREFIX + 'Saving PFM image to: ', pfm_filename)
        save_pfm(pfm_filename, np.float32(R))
        print(APP_PREFIX + 'Saving Tonemap for the scene to: ', tonemap_filename)
        imsave(tonemap_filename, tonemap(relative_R[:, :, channel]))
        print(APP_PREFIX + 'Saving HDR image to: ', png_filename)
        # Gamma compression
        where_Iam = 'Gamma compression'
        gamma = 0.6
        imsave(png_filename, np.power(relative_R, gamma))

        print(APP_PREFIX + 'Creating and saving response function plot')
        # Creates a plot for the response curve
        where_Iam = 'Plotting Responsecurve'
        plot(G, np.arange(256))
        title('RGB Response function')
        xlabel('log exposure')
        ylabel('Z value')
        savefig(join(output_path, 'response_curve.png'))

        print(APP_PREFIX + 'Done.')


    except Exception as e:
        print('{}: Main Error: {} '.format(where_Iam ,str(e)))

if __name__ == '__main__':
    main()

