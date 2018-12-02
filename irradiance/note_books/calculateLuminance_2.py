import numpy as np
import os
import re
import sys
import math
import os.path
from os.path import join
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from fractions import Fraction
from datetime import datetime

######################################################################
## Hoa: 06.11.2018 Version 2 : calculateLuminance2.py
######################################################################
# Version 2: Will try to re process missing data.
#
# Source : https://github.com/Soumyabrata/solar-irradiance-estimation
# path_img needs to include \temp directory example:
# r'\\192.168.1.8\SkyCam_FTP\SKY_CAM\camera_2\cam_2_vers3\20181013_raw_cam2\temp'
#
# New /Changes:
# ----------------------------------------------------------------------
#
# 05.11.2018 : first add
# 06.11.2018 : added data integrity check
# 02.12.2018 : tries to reprocess missing data
######################################################################

path_img = r'\\IHLNAS05\SkyCam_FTP\SKY_CAM\camera_1\cam_1_vers3\20181013_raw_cam1\temp'

global len_interpolated
len_interpolated = 0

class HDR:
    def make_hdr(self, path, listOfSS, img_type='jpg'):
        try:
            h = Helpers()
            success = False
            img_dir = []
            type = ''

            output_path = join(path, 'output')

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            date, time = h.strip_date_and_time(path)
            date_time = date.strip('-') + '_' + time.strip(':')

            if img_type is 'jpg':
                type = '*.jpg'
            else:
                type = '*.data'

            # Load all images
            for imgFile in sorted(glob(os.path.join(path, type))):
                if os.path.isfile(imgFile):
                    img_dir.append(imgFile)

            # Sort image order to match with listOfSS
            img_dir.sort(key=h.byInteger_keys)

            # Loading images channel - wise
            img_list_b = self.load_img_by_chn(img_dir, 0)
            img_list_g = self.load_img_by_chn(img_dir, 1)
            img_list_r = self.load_img_by_chn(img_dir, 2)

            # Solving response curves  (np.linalg.lstsq can be troublesome ! -> rcond=None )
            gb, _ = self.hdr_debvec(img_list_b, listOfSS)
            gg, _ = self.hdr_debvec(img_list_g, listOfSS)
            gr, _ = self.hdr_debvec(img_list_r, listOfSS)

            if img_type is 'jpg':
                hdr = self.construct_hdr([img_list_b, img_list_g, img_list_r], [gb, gg, gr], listOfSS)

                with open(join(output_path, 'hdr_jpg.dat'), 'wb') as f:
                    hdr.tofile(f)

                # cb = COLORBALANCE() # not clear if luminace information will be preserved

                # create thumbnails image
                hdr_reinhard = self.tonemapReinhard(hdr)
                w, h, d = hdr.shape
                hdr_reinhard_s = cv2.resize(hdr_reinhard, (int(h / 3), int(w / 3)))
                rhard_8bit = cv2.normalize(hdr_reinhard_s, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)

                cv2.imwrite(join(output_path, 'hdr_jpg.jpg'), rhard_8bit)
                # clean up
                del rhard_8bit;
                del hdr_reinhard;
                del hdr_reinhard_s

            if img_type is 'data':
                hdr = self.construct_hdr([img_list_b, img_list_g, img_list_r], [gb, gg, gr], listOfSS)

                with open(join(output_path, 'hdr_data.dat'), 'wb') as f:
                    hdr.tofile(f)

                # create thumbnails image
                hdr_reinhard = self.tonemapReinhard(hdr)
                w, h, d = hdr_reinhard.shape
                hdr_reinhard_s = cv2.resize(hdr_reinhard, (int(h / 3), int(w / 3)))
                rhard_8bit = cv2.normalize(hdr_reinhard_s, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)

                cv2.imwrite(join(output_path, 'hdr_data.jpg'), rhard_8bit)
                # clean up
                del hdr;
                del hdr_reinhard;
                del hdr_reinhard_s;
                del rhard_8bit

            # clean up
            del img_dir;
            del img_list_b;
            del img_list_g;
            del img_list_r;
            del listOfSS
            del gb;
            del gr;
            del gg

            success = True
            return success

        except Exception as e:
            print('error: make_hdr ' + str(e))
            return success

    def tonemapReinhard(self, hdr):
        tonemapReinhard = cv2.createTonemapReinhard(1.5, 0, 0, 0)
        ldrReinhard = tonemapReinhard.process(hdr)
        return ldrReinhard * 255

    def demosaic1(self, mosaic, awb_gains=None):
        '''
        nedded by make_hdr
        :param mosaic:
        :param awb_gains:
        :return:
        '''
        try:
            black = mosaic.min()
            saturation = mosaic.max()

            uint14_max = 2 ** 14 - 1
            mosaic -= black  # black subtraction
            mosaic *= int(uint14_max / (saturation - black))
            mosaic = np.clip(mosaic, 0, uint14_max)  # clip to range

            if awb_gains is None:
                vb_gain = 1.0
                vg_gain = 1.0
                vr_gain = 1.0
            else:
                vb_gain = awb_gains[1]
                vg_gain = 1.0
                vr_gain = awb_gains[0]

            mosaic = mosaic.reshape([2464, 3296])
            mosaic = mosaic.astype('float')
            mosaic[0::2, 1::2] *= vb_gain  # Blue
            mosaic[1::2, 0::2] *= vr_gain  # Red
            mosaic = np.clip(mosaic, 0, uint14_max)  # clip to range
            mosaic *= 2 ** 2

            # demosaic
            p1 = mosaic[0::2, 1::2]  # Blue
            p2 = mosaic[0::2, 0::2]  # Green
            p3 = mosaic[1::2, 1::2]  # Green
            p4 = mosaic[1::2, 0::2]  # Red

            blue = p1
            green = np.clip((p2 // 2 + p3 // 2), 0, 2 ** 16 - 1)
            red = p4

            image = np.dstack([red, green, blue])  # 16 - bit 'image'

            # down sample to RGB 8 bit image use: self.deraw2rgb1(image)

            return image

        except Exception as e:
            print('Error in demosaic1: {}'.format(e))

    def toRGB_1(self, data):
        '''
        nedded by make_hdr
        Belongs to deraw1
        :param data:
        :return:
        '''
        image = data // 256  # reduce dynamic range to 8 bpp
        image = np.clip(image, 0, 255).astype(np.uint8)

        return image

    def read_data(self, path_to_image):
        '''
        nedded by make_hdr
        :param path_to_image:
        :return:
        '''
        data = np.fromfile(path_to_image, dtype='uint16')
        data = data.reshape([2464, 3296])
        raw = self.demosaic1(data)

        return raw.astype('uint16')

    def load_img_by_chn(self, dir_list, channel=0):
        '''
        nedded by make_hdr
        Reads either of jpg or raw depending on file extension
        :param source_dir:
        :param channel:
        :return:
        '''
        img_list = []
        if dir_list[0].endswith('.data'):
            img_list = [self.toRGB_1(self.read_data(file)) for file in dir_list]
            img_list = [img[:, :, channel] for img in img_list]

        if dir_list[0].endswith('.jpg'):
            img_list = [cv2.imread(file, 1) for file in dir_list]
            img_list = [img[:, :, channel] for img in img_list]

        return img_list

    def median_threshold_bitmap_alignment(self, img_list):
        '''
         MTB implementation
         nedded by make_hdr
        :return:
        '''
        median = [np.median(img) for img in img_list]
        binary_thres_img = [cv2.threshold(img_list[i], median[i], 255, cv2.THRESH_BINARY)[1] for i in
                            range(len(img_list))]
        mask_img = [cv2.inRange(img_list[i], median[i] - 20, median[i] + 20) for i in range(len(img_list))]

        plt.imshow(mask_img[0], cmap='gray')
        plt.show()

        max_offset = np.max(img_list[0].shape)
        levels = 5

        global_offset = []
        for i in range(0, len(img_list)):
            offset = [[0, 0]]
            for level in range(levels, -1, -1):
                scaled_img = cv2.resize(binary_thres_img[i], (0, 0), fx=1 / (2 ** level), fy=1 / (2 ** level))
                ground_img = cv2.resize(binary_thres_img[0], (0, 0), fx=1 / (2 ** level), fy=1 / (2 ** level))
                ground_mask = cv2.resize(mask_img[0], (0, 0), fx=1 / (2 ** level), fy=1 / (2 ** level))
                mask = cv2.resize(mask_img[i], (0, 0), fx=1 / (2 ** level), fy=1 / (2 ** level))

                level_offset = [0, 0]
                diff = float('Inf')
                for y in [-1, 0, 1]:
                    for x in [-1, 0, 1]:
                        off = [offset[-1][0] * 2 + y, offset[-1][1] * 2 + x]
                        error = 0
                        for row in range(ground_img.shape[0]):
                            for col in range(ground_img.shape[1]):
                                if off[1] + col < 0 or off[0] + row < 0 or off[1] + col >= ground_img.shape[1] or off[
                                    0] + row >= ground_img.shape[1]:
                                    continue
                                if ground_mask[row][col] == 255:
                                    continue
                                error += 1 if ground_img[row][col] != scaled_img[y + off[0]][x + off[1]] else 0
                        if error < diff:
                            level_offset = off
                            diff = error
                offset += [level_offset]
            global_offset += [offset[-1]]
        return global_offset

    def hdr_debvec(self, img_list, exposure_times):
        '''
        needed by make_hdr
        :param exposure_times:
        :return:
        '''
        B = [math.log(e, 2) for e in exposure_times]
        l = 50  # lambda sets amount of smoothness
        w = [z if z <= 0.5 * 255 else 255 - z for z in range(256)]

        small_img = [cv2.resize(img, (10, 10)) for img in img_list]
        Z = [img.flatten() for img in small_img]

        return self.response_curve_solver(Z, B, l, w)

    def response_curve_solver(self, Z, B, l, w):
        '''
        Implementation of paper's Equation(3) with weight
         needed by make_hdr
        :param B:
        :param l:
        :param w:
        :return:
        '''
        n = 256
        A = np.zeros(shape=(np.size(Z, 0) * np.size(Z, 1) + n + 1, n + np.size(Z, 1)), dtype=np.float32)
        b = np.zeros(shape=(np.size(A, 0), 1), dtype=np.float32)

        # Include the data−fitting equations
        k = 0
        for i in range(np.size(Z, 1)):
            for j in range(np.size(Z, 0)):
                z = Z[j][i]
                wij = w[z]
                A[k][z] = wij
                A[k][n + i] = -wij
                b[k] = wij * B[j]
                k += 1

        # Fix the curve by setting its middle value to 0
        A[k][128] = 1
        k += 1

        # Include the smoothness equations
        for i in range(n - 1):
            A[k][i] = l * w[i + 1]
            A[k][i + 1] = -2 * l * w[i + 1]
            A[k][i + 2] = l * w[i + 1]
            k += 1

        # Solve the system using SVD
        x = np.linalg.lstsq(A, b, rcond=None)[0]  # rcond=None
        g = x[:256]
        lE = x[256:]

        return g, lE

    def construct_radiance_map(self, g, Z, ln_t, w):
        '''
        Implementation of paper's Equation(6)
        needed by make_hdr
        :param Z:
        :param ln_t:
        :param w:
        :return:
        '''
        acc_E = [0] * len(Z[0])
        ln_E = [0] * len(Z[0])

        pixels, imgs = len(Z[0]), len(Z)
        for i in range(pixels):
            acc_w = 0
            for j in range(imgs):
                z = Z[j][i]
                acc_E[i] += w[z] * (g[z] - ln_t[j])
                acc_w += w[z]
            ln_E[i] = acc_E[i] / acc_w if acc_w > 0 else acc_E[i]
            acc_w = 0

        return ln_E

    def construct_hdr(self, img_list, response_curve, exposure_times):
        '''
        Construct radiance map for each channels
        needed by make_hdr
        :param img_list:
        :param response_curve:
        :param exposure_times:
        :return:
        '''
        img_size = img_list[0][0].shape
        w = [z if z <= 0.5 * 255 else 255 - z for z in range(256)]
        ln_t = np.log2(exposure_times)

        vfunc = np.vectorize(lambda x: math.exp(x))
        hdr = np.zeros((img_size[0], img_size[1], 3), 'float32')

        # construct radiance map for BGR channels
        for i in range(3):
            Z = [img.flatten().tolist() for img in img_list[i]]
            E = self.construct_radiance_map(response_curve[i], Z, ln_t, w)
            # Exponational each channels and reshape to 2D-matrix
            hdr[..., i] = np.reshape(vfunc(E), img_size)

        return hdr

    def hdr_to_blob(self, hdr):
        '''
        Concatenates HDR image and header to BLBO
        Code based on https://gist.github.com/edouardp/3089602
        needed by make_hdr
        :param filename:
        :return:
        '''
        try:
            image = np.zeros((hdr.shape[0], hdr.shape[1], 3), 'float32')
            image[..., 0] = hdr[..., 2]
            image[..., 1] = hdr[..., 1]
            image[..., 2] = hdr[..., 0]

            title = (b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
            header = '-Y {0} +X {1}\n'.format(image.shape[0], image.shape[1])
            header = (bytes(header, encoding='utf-8'))

            brightest = np.maximum(np.maximum(image[..., 0], image[..., 1]), image[..., 2])
            mantissa = np.zeros_like(brightest)
            exponent = np.zeros_like(brightest)
            np.frexp(brightest, mantissa, exponent)
            scaled_mantissa = mantissa * 256.0 / brightest
            rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
            rgbe[..., 0:3] = np.around(image[..., 0:3] * scaled_mantissa[..., None])
            rgbe[..., 3] = np.around(exponent + 128)

            _rgbe = rgbe.flatten()

            byte_str = title + header + _rgbe.tobytes()
            blob_b = io.BytesIO(byte_str)
            blob_b.seek(0)
            blob = blob_b.read()

            # filename = r'\\IHLNAS05\SkyCam_FTP\camera_1\cam_1_vers3\20200505_raw_cam1\temp\hdr.hdr'
            # f = open(filename, 'wb')
            # f.write(blob)
            # f.close()

            return blob

        except Exception as e:
            print('error in: hdr_to_blob ' + str(e))
            return None

    def load_img_as_blob(self, path):
        try:
            img = cv2.imread(path)
            w, h, d = img.shape
            img_s = cv2.resize(img, (int(h / 3), int(w / 3)))

            fig, ax = plt.subplots(figsize=plt.figaspect(img_s))
            fig.subplots_adjust(0, 0, 1, 1)
            ax.set_axis_off()
            ax.imshow(img_s)
            byte_str = io.BytesIO()
            fig.savefig(byte_str, format='jpg')
            byte_str.seek(0)
            blob = byte_str.read()
            plt.close()

            return blob

        except Exception as e:
            print('error in save_thumb ' + str(e))
            return None

    def save_hdr(self, hdr, filename):
        '''
        LOESCHEN
        :param filename:
        :return:
        '''
        image = np.zeros((hdr.shape[0], hdr.shape[1], 3), 'float32')
        image[..., 0] = hdr[..., 2]
        image[..., 1] = hdr[..., 1]
        image[..., 2] = hdr[..., 0]

        print('Path to save HDR: {}'.format(filename))

        f = open(filename, 'wb')
        f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
        header = '-Y {0} +X {1}\n'.format(image.shape[0], image.shape[1])
        f.write(bytes(header, encoding='utf-8'))

        brightest = np.maximum(np.maximum(image[..., 0], image[..., 1]), image[..., 2])
        mantissa = np.zeros_like(brightest)
        exponent = np.zeros_like(brightest)
        np.frexp(brightest, mantissa, exponent)
        scaled_mantissa = mantissa * 256.0 / brightest
        rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        rgbe[..., 0:3] = np.around(image[..., 0:3] * scaled_mantissa[..., None])
        rgbe[..., 3] = np.around(exponent + 128)

        rgbe.flatten().tofile(f)
        f.close()

    def mask_sat(self, img_list, comb_I):
        '''
        # This function is used to mask the pixels of any image comb_I; given three
        # LDR images I1, I2, I3. If a pixel > 240, in all the three LDR images
        # simulateneously, it is called saturated.
        # Source is matlab script from
        # https://github.com/Soumyabrata/HDR-cloud-segmentation/tree/master/HDRimaging
        :param I1:
        :param I2:
        :param I3:
        :param comb_I:
        :return:
        '''
        try:
            I1 = img_list[0]
            I2 = img_list[1]
            I3 = img_list[2]
            w, h, d = I1.shape
            I1_gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
            I2_gray = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
            I3_gray = cv2.cvtColor(I3, cv2.COLOR_BGR2GRAY)

            mask = np.ones((w, h))

            for i in range(0, w - 1):
                for j in range(0, h - 1):
                    if (I1_gray[i][j] > 240) and (I2_gray[i][j] > 240) and (I3_gray[i][j] > 240):
                        mask[i][j] = 0

            mask_I = comb_I

            for i in range(0, w - 1):
                for j in range(0, h - 1):
                    if mask[i][j] == 0:
                        mask_I[i][j][0] = 255
                        mask_I[i][j][1] = 0
                        mask_I[i][j][2] = 255

            return mask_I

        except Exception as e:
            print('Error in mask_sat: {}'.format(e))
            return mask_I

class Helpers:
    def calculate_sun_centre(self, LDR_low, img_numb):
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
                centroid_x = np.nan
                centroid_y = np.nan
                print('img {} NO CENTRE'.format(img_numb))
            else:
                centroid_x = int(sum_x / length)
                centroid_y = int(sum_y / length)
                # Loeschen
                print(' FOUND CENTRE in {} at: {}/{}:'.format(img_numb, centroid_x, centroid_y))

            return centroid_x, centroid_y
        except Exception as e:
            return 0, 0
            print("Error in sol: " + str(e))

    def demo_calculate_sun_centre(self, LDR_low, show_sun=False):
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

            if show_sun:
                cv2.circle(LDR_low, (centroid_y, centroid_x,), 30, (0, 0, 255), thickness=10, lineType=8, shift=0)

            return centroid_x, centroid_y
        except Exception as e:
            return 0, 0
            print("Error in sol: " + str(e))

    def interpolate_missing_sun_pos(self, list_sun_CX, list_sun_CY):
        try:
            # loeschen
            print('interpolating -> len list cx: {}'.format(len(list_sun_CX)))
            print('interpolating -> len list cy: {}'.format(len(list_sun_CY)))

            # Interpolate the sun's location in the missing places
            s1 = pd.Series(list_sun_CX)
            s2 = pd.Series(list_sun_CY)

            # loeschen
            print(
                'before interpolating -> length of s1: {}  # values: {} # nans: {}'.format(s1.size, pd.isnull(s1).sum(),
                                                                                           (~pd.isnull(s1)).sum()))
            print(
                'before interpolating -> length of s2: {}  # values: {} # nans: {}'.format(s2.size, pd.isnull(s2).sum(),
                                                                                           (~pd.isnull(s2)).sum()))

            complete_x = s1.interpolate()
            complete_y = s2.interpolate()

            # loeschen
            print('after  interpolating -> length of s1: {} # values: {} # nans: {}'.format(complete_x.size,
                                                                                            pd.isnull(complete_x).sum(),
                                                                                            (~pd.isnull(
                                                                                                complete_x)).sum()))
            print('after  interpolating -> length of s2: {} # values: {} # nans: {}'.format(complete_y.size,
                                                                                            pd.isnull(complete_y).sum(),
                                                                                            (~pd.isnull(
                                                                                                complete_y)).sum()))

            # All computed values are NaN
            if (np.isnan(complete_x).any()) or (np.isnan(complete_y).any()):  # Sollte dies ueberspringen
                print('interpolating -> All computed values are NaN')
                complete_x = np.array([])
                complete_y = np.array([])
            else:
                # Replacing NaN s in the beginning with closest non-NaN value
                # For x coordinate
                a = complete_x
                ind = np.where(~np.isnan(a))[0]
                first, last = ind[0], ind[-1]
                a[:first] = a[first]
                a[last + 1:] = a[last]

                # For y coordinate
                a = complete_y
                ind = np.where(~np.isnan(a))[0]
                first, last = ind[0], ind[-1]
                a[:first] = a[first]
                a[last + 1:] = a[last]

            return (complete_x, complete_y)

        except Exception as e:
            print('Error in interpolate_missing_sun_pos: {}'.format(e))
            return (0, 0)

    def get_int(self, text):
        return int(text) if text.isdigit() else text

    def byInteger_keys(self, text):
        return [self.get_int(c) for c in re.split('(\d+)', text)]

    def getDirectories(self, path_to_dirs):
        try:
            avoid_this_Dirs = ['imgs', 'hdr']
            allDirs = []
            img_cnt = 1

            for dirs in sorted(glob(join(path_to_dirs, "*", ""))):
                if os.path.isdir(dirs):
                    if dirs.rstrip('\\').rpartition('\\')[-1] not in avoid_this_Dirs:
                        allDirs.append(dirs.rstrip('\\'))
                        img_cnt += 1
            return allDirs

        except Exception as e:
            print('getDirectories: Error: ' + str(e))

    def getDateSring(self, path):
        try:
            date = ''
            temp = path.rpartition('\\')[0]
            temp = temp.rpartition('\\')[-1]
            date = temp.rpartition('_raw')[0]
            return date

        except Exception as e:
            print('Error getDateSring:{}').format(e)
            return date

    def strip_date_and_time(self, path):
        try:
            formated_date = '0000-00-00'
            formated_time = datetime.now().strftime('%H:%M:%S')

            date = (path.rstrip('\\').rpartition('\\')[-1]).rpartition('_')[0]
            time = (path.rstrip('\\').rpartition('\\')[-1]).rpartition('_')[-1]

            year = date[:4]
            month = date[4:6]
            day = date[6:8]
            hour = time[:2]
            min = time[2:4]
            sec = time[4:6]

            check = [year, month, day, hour, min, sec]

            for item in check:
                if not item or not item.isdigit():
                    print('strip_date_and_time: could not read date and time  used {} {} instead !{}'.format(
                        formated_date, formated_time))
                    return formated_date, formated_time

            formated_date = '{}-{}-{}'.format(year, month, day)
            formated_time = '{}:{}:{}'.format(hour, min, sec)

            return formated_date, formated_time
        except Exception as e:
            return formated_date, formated_time

    def strip_name_swvers_camid(self, path):
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

    def strip_img_type(self, path):
        type = ''
        temp = path.rpartition('\\')[-1]
        temp = temp.rpartition('_')[-1]
        type = temp.rpartition('.')[0]

        return str(type)

    def strip_name_swvers_camid(self, path):
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

    def getShutterTimes(self, path):
        try:
            '''
            returns shutter_time in microseconds as np.float32 type
            '''
            dir_name, sw_vers, camera_ID = self.strip_name_swvers_camid(path)
            types = ('*.txt', '*.log')
            ss_to_db = []

            for typ in types:
                for file in sorted(glob(os.path.join(path, typ))):
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
                    value = line.split("camera shutter speed:", 1)[1].replace('[', '').replace(']', '')
                    value = value.split('|', 1)[0]
                    value = value.strip()
                    ss_to_db.append(value + ",")
                    value += '/1000000'
                    val_float = np.float32(Fraction(str(value)))
                    listOfSS[pos] = val_float
                    pos += 1

            else:
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
                    pos += 1

            ss_to_db_str = ''.join(ss_to_db)

            return listOfSS, ss_to_db_str.rstrip(',')

        except Exception as e:
            print('Error in getShutterTimes: ' + str(e))

    def getNumpyArray(self, path):
        try:
            data = np.fromfile(path, dtype='uint16')
            img = None

            dir_name, sw_vers, cam_ID = self.strip_name_swvers_camid(path)
            type = self.strip_img_type(path)

            if type == 'jpg':
                image_arr = np.frombuffer(data, dtype=np.float32)
                img = image_arr.reshape(1944, 2592, 3)
            if type == 'data':
                image_arr = np.frombuffer(data, dtype=np.float32)
                img = image_arr.reshape(1232, 1648, 3)

            return int(cam_ID), type, img

        except Exception as e:
            print('Error getNumpyArray: {}'.format(e))

    def tonemap(self, hdr):
        hdr = np.float32(hdr)
        tonemapReinhard = cv2.createTonemapReinhard(1.5, 0, 0, 0)
        ldrReinhard = tonemapReinhard.process(hdr)
        return ldrReinhard * 255

    def show_hdr_image(self, title, hdr, resize=2):
        img_tonemp = self.tonemap(hdr)
        img_8bit = cv2.normalize(img_tonemp, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        w, h, d = img_8bit.shape
        img_8bit_s = cv2.resize(img_8bit, (int(h / resize), int(w / resize)))
        cv2.imshow(title, img_8bit_s)

    def show_ldr_image(self, title, ldr, resize=3):
        w, h, d = ldr.shape
        img_s = cv2.resize(ldr, (int(h / resize), int(w / resize)))
        cv2.imshow(title, img_s)

    def cmask(self, index, radius, array):
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

    def rectmask(self, corner, dimension, array):
        """Generates the mask for a given input image.
        The generated mask is needed to remove occlusions during post-processing steps.

        Args:
            index (numpy array): Array containing the x- and y- co-ordinate of the center of the circular mask.
            radius (float): Radius of the circular mask.
            array (numpy array): Input sky/cloud image for which the mask is generated.

        Returns:
            numpy array: Generated mask image."""

        w, h = dimension  # width and height
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
        y, x = np.mgrid[-a:nx - a, -b:ny - b]
        mask = (x < a) & (x - a <= w) & (y > b) & (y - b <= h)
        image_mask[~mask] = 1

        return (image_mask)

    def maske_circle(self, input_image, size=[0, 0, 3], centre=[0, 0], radius=0, show_mask=False):

        empty_img = np.zeros(size, dtype=np.uint8)
        mask = self.cmask(centre, radius, empty_img)

        red = input_image[:, :, 0]
        green = input_image[:, :, 1]
        blue = input_image[:, :, 2]

        if show_mask:
            h = input_image.shape[0]
            w = input_image.shape[1]

            for y in range(0, h):
                for x in range(0, w):
                    if mask[y, x] == 0:
                        blue[y, x] = 65535
            b_img = blue
        else:
            b_img = blue.astype(float) * mask

        r_img = red.astype(float) * mask
        g_img = green.astype(float) * mask
        # b_img = blue.astype(float) * mask

        dimension = (input_image.shape[0], input_image.shape[1], 3)
        output_img = np.zeros(dimension, dtype=float)

        output_img[..., 0] = r_img[:, :]
        output_img[..., 1] = g_img[:, :]
        output_img[..., 2] = b_img[:, :]

        return output_img

    def maske_rectangel(self, input_image, size=[0, 0, 3], corner=[0, 0], dim=[0, 0], show_mask=False):
        empty_img = np.zeros(size, dtype=np.uint8)
        mask = self.rectmask(corner, dim, empty_img)

        red = input_image[:, :, 0]
        green = input_image[:, :, 1]
        blue = input_image[:, :, 2]

        if show_mask:
            h = input_image.shape[0]
            w = input_image.shape[1]

            for y in range(0, h):
                for x in range(0, w):
                    if mask[y, x] == 0:
                        green[y, x] = 65535
            g_img = green
        else:
            g_img = green.astype(float) * mask

        r_img = red.astype(float) * mask
        # g_img = green.astype(float) * mask
        b_img = blue.astype(float) * mask

        dimension = (input_image.shape[0], input_image.shape[1], 3)
        output_img = np.zeros(dimension, dtype=float)

        output_img[..., 0] = r_img[:, :]
        output_img[..., 1] = g_img[:, :]
        output_img[..., 2] = b_img[:, :]

        return output_img

    def mask_array(self, data, cam_id='1', type='', show_mask=False):

        masked_img = None

        w = data.shape[0]
        h = data.shape[1]
        c = data.shape[2]

        if cam_id == 1:
            if type == 'data':
                centre = [505, 746]  # [y,x] !
                radius = 680

            if type == 'jpg':
                centre = [795, 1190]  # [y,x] !
                radius = 1050

            masked_img = self.maske_circle(data, [w, h, c], centre, radius, show_mask)

        if cam_id == 2:
            if type == 'data':
                centre = [620, 885]  # [y,x] !
                radius = 680
                corner = [0, 520]
                dimension = [0, 100]

            if type == 'jpg':
                centre = [1080, 1300]  # [y,x] !
                radius = 1065  # 1065
                corner = [0, 822]
                dimension = [0, 168]

            masked_img = self.maske_circle(data, [w, h, c], centre, radius, show_mask)
            masked_img = self.maske_rectangel(masked_img, [w, h, c], corner, dimension, show_mask)

        return masked_img

    def LuminanceSquareCrop(self, LDR_low_img, exp_low_time, sun_x, sun_y, crop_dim=300, show_rect=False):
        try:
            centroid_x = sun_x
            centroid_y = sun_y

            # Construct rectangle
            around_sun = LDR_low_img[(int(centroid_x - crop_dim / 2)):(int(centroid_x + crop_dim / 2)),
                         (int(centroid_y - crop_dim / 2)):(int(centroid_y + crop_dim / 2))]

            if show_rect:
                x_1 = int(centroid_x - crop_dim / 2)
                y_1 = int(centroid_y - crop_dim / 2)
                x_2 = int(centroid_x + crop_dim / 2)
                y_2 = int(centroid_y + crop_dim / 2)
                cv2.rectangle(LDR_low_img, (x_1, y_1), (x_2, y_2), (255, 255, 255), 3)

            lum = 0.2126 * around_sun[:, :, 0] + 0.7152 * around_sun[:, :, 1] + 0.0722 * around_sun[:, :, 2]
            lum = np.mean(lum)

            LDRLuminance = lum / exp_low_time

        except Exception as e:
            print("Error in LuminanceSquareCrop: " + str(e))

        return LDRLuminance

    def demonstrate(self, dir):

        # dir = r'E:\SkY_CAM_IMGS\camera_2\cam_2_vers3\29181012_raw_cam2\temp\20181012_145034'
        if not os.path.isdir(dir):
            print('Could not finde file!')
            sys.exit(0)
        else:
            listOfSS, _ = self.getShutterTimes(dir)
            exp_low_time = listOfSS[2]
            LDR_low = cv2.imread(join(dir, 'raw_img-4.jpg'))
            sun_y, sun_x = self.demo_calculate_sun_centre(LDR_low, True)
            lum_sqrcrpt = self.LuminanceSquareCrop(LDR_low, exp_low_time, sun_x, sun_y, 300, True)
            print('From square cropped luminance:{}'.format(lum_sqrcrpt))

            cam_id, type, arr_hdr = self.getNumpyArray(join(dir, 'output', 'hdr_data.dat'))
            lum_hdr = np.mean(arr_hdr)
            print('mean relat lum hdr unmasked:  {}'.format(lum_hdr))

            arr_hdr_m = self.mask_array(arr_hdr, cam_id, type, True)
            lum_hdr_m = np.mean(arr_hdr_m)
            print('mean relat lum hdr masked:    {}'.format(lum_hdr_m))

            cam_id, type, arr_jpg = self.getNumpyArray(join(dir, 'output', 'hdr_jpg.dat'))
            arr_jpg_m = self.mask_array(arr_jpg, cam_id, type, True)
            lum_jpg_m = np.mean(arr_jpg_m)
            print('mean relat lum jpg masked:    {}'.format(lum_jpg_m))

            self.show_ldr_image('Low exposure with cropped square', LDR_low, 5)
            self.show_hdr_image('HDR from raw image', arr_hdr_m, 4)
            self.show_hdr_image('HDR from jpg image', arr_jpg_m, 5)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            print('Done demonstrate.')
            sys.exit(0)

    def check_data_integrity(self, path_img):
        try:
            all_dirs = self.getDirectories(path_img)
            files_to_check = ['hdr_data.dat', 'hdr_data.jpg', 'hdr_jpg.dat', 'hdr_jpg.jpg']
            result = []
            missing_data_dirs = []

            for dir in all_dirs:
                path_output = join(dir, 'output')
                locked_output = join(dir, '_output')
                if not os.path.exists(path_output):
                    msg = 'missing output dir in : {}'.format(dir)
                    result.append(msg)
                    missing_data_dirs.append(dir)
                else:
                    add_dir_to_missing = False
                    if os.path.exists(locked_output):
                        os.rename(locked_output, path_output)
                    for file in files_to_check:
                        if not os.path.isfile(join(path_output, file)):
                            add_dir_to_missing = True
                        else:
                            if os.path.getsize(join(path_output, file)) == 0:
                                os.remove(join(path_output, file))
                                add_dir_to_missing = True

                    if add_dir_to_missing:
                        msg = 'missing file in dir: {}'.format(dir)
                        result.append(msg)
                        missing_data_dirs.append(dir)

            found_errors = len(result)
            print('Found {} missing data:'.format(found_errors))
            for res in result:
                print('{}'.format(res))

            return (found_errors == 0), missing_data_dirs

        except Exception as e:
            print('check_data_integrity: {}'.format(e))

    def re_process_missing_data(self, missing_data_dirs):
        try:
            hdr = HDR()
            for dir in missing_data_dirs:
                print('Reprocessing missing data in: {}'.format(dir))
                listOfSS, ss_to_db = self.getShutterTimes(dir)

                path_output = join(dir, 'output')
                if not os.path.exists(path_output):
                    hdr_dat_ok = hdr.make_hdr(dir, listOfSS, 'data')
                    hdr_jpg_ok = hdr.make_hdr(dir, listOfSS, 'jpg')
                else:
                    if not os.path.isfile(join(dir, 'output', 'hdr_data.dat')):
                        hdr_dat_ok = hdr.make_hdr(dir, listOfSS, 'data')
                    if not os.path.isfile(join(dir, 'output', 'hdr_jpg.dat')):
                        hdr_dat_ok = hdr.make_hdr(dir, listOfSS, 'jpg')

        except Exception as e:
            print('re_process_missing_data: {}'.format(e))

def main():
    try:
        global path_img
        h = Helpers()
        # demonstrate()

        print('Path to source: {}'.format(path_img))
        print('Checking data integrity:')
        data_ok, missing_data_dirs = h.check_data_integrity(path_img)
        print('Data integrity check done.')
        if not data_ok:
            print('Processing stopped, trying to fix missing data first.')
            h.re_process_missing_data(missing_data_dirs)
            data_ok, _ = h.check_data_integrity(path_img)
            if not data_ok:
                print('Could not fix missing data, stopping processing.')
                return
            else:
                print('Successfully fixed missing data, proceeding with:')

        print('calculate luminance...')
        name, sw_vers, cam_id = h.strip_name_swvers_camid(path_img)
        timestamp = h.getDateSring(path_img)
        all_dirs = h.getDirectories(path_img)
        listOfAll_SS = []
        listOfAll_date = []
        listOffAll_time = []
        listOfAll_LDRs = []
        listOf_lum_hdr = []
        listOf_lum_hdr_m = []
        listOf_lum_jpg_m = []
        listOf_lum_sqrcrpt = []

        cnt = 0
        for dir in all_dirs:
            formated_date, formated_time = h.strip_date_and_time(dir)
            listOfAll_date.append(formated_date)
            listOffAll_time.append(formated_time)
            listOfSS, _ = h.getShutterTimes(dir)
            listOfAll_SS.append(listOfSS[2])
            LDR_low = cv2.imread(join(dir, 'raw_img-4.jpg'))
            listOfAll_LDRs.append(LDR_low)

            # Calculate relative luminance from raw HDR
            cam_id, type, arr_hdr = h.getNumpyArray(join(dir, 'output', 'hdr_data.dat'))
            lum_hdr = np.mean(arr_hdr)
            listOf_lum_hdr.append(str(round(lum_hdr, 7)))

            # Calculate relative luminance from masked raw HDR
            arr_hdr_m = h.mask_array(arr_hdr, cam_id, type, False)
            lum_hdr_m = np.mean(arr_hdr_m)
            listOf_lum_hdr_m.append(str(round(lum_hdr_m, 7)))

            # Calculate relative luminance from jpg HDR
            cam_id, type, arr_jpg = h.getNumpyArray(join(dir, 'output', 'hdr_jpg.dat'))
            arr_jpg_m = h.mask_array(arr_jpg, cam_id, type, False)
            lum_jpg_m = np.mean(arr_jpg_m)
            listOf_lum_jpg_m.append(str(round(lum_jpg_m, 7)))

            cnt += 1
            print('{}'.format(cnt))
            # if cnt == 5:  # LOESCHEN
            #    break

        print('Done loading images, starting to calculate cropped square iluminance.')

        # Calculate luminance from jpg, as elaborated by Soumyabrata Dev (see in header)
        list_sun_X = []
        list_sun_Y = []

        # loeschen
        count = 0

        for ldr_low in listOfAll_LDRs:
            count += 1
            sun_y, sun_x = h.calculate_sun_centre(ldr_low, count)
            list_sun_X.append(sun_x)
            list_sun_Y.append(sun_y)

        print('Len(list_sun) x/y sun centre: {} x-coords and {} y-coords'.format(len(list_sun_X), len(
            list_sun_Y)))  # -> ok: Found 381 x-coords and 381 y-coords

        # Interpolate missing sun positions
        # print('Interpolating missing sun\'s position')
        # complete_sunX, complete_sunY = interpolate_missing_sun_pos(list_sun_X, list_sun_Y) # Hier ist ein Fehler:
        # print('Done interpolating missing sun\'s position')                                # complete_sunX / Y sind leer !

        # Debugging -> loeschen
        # print('complete_sunX:{}'.format(len(complete_sunX))) # diese sind leer !
        # print('complete_sunY:{}'.format(len(complete_sunY))) # diese sind leer !

        # Header of *.csv file.
        file_name = timestamp + '_luminance.csv'
        text_file = open(join(path_img, file_name), "w")
        text_file.write("####################################################################### \n")
        text_file.write("# Datet Time: {} Camera ID: {} Softwareversion: {}.  \n".format(timestamp, cam_id, sw_vers))
        text_file.write("####################################################################### \n")
        text_file.write("# sun_x, sun_y: position in pixels of sun.  \n")
        text_file.write("# lum_hdr:      luminance from HDR image  \n")
        text_file.write("# lum_hdr_m:    luminance from masked HDR image  \n")
        text_file.write("# lum_jpg_m:    luminance from masked and tone mapped HDR image  \n")
        text_file.write("# lum_sqrcrpt:  luminance from cropped square of low_LDR image  \n")
        text_file.write("####################################################################### \n")
        text_file.write("no,date,time,sun_x,sun_y,lum_hdr,lum_hdr_m,lum_jpg_m,lum_sqrcrpt \n")

        print('Calculating square cropped luminance')
        for i, ldr_low in enumerate(listOfAll_LDRs):
            # sun_x = complete_sunX[i]                    # Hier ist der Fehler:      sun_x = complete_sunX[i]
            # sun_y = complete_sunY[i]                    # IndexError: index 0 is out of bounds for axis 0 with size 0
            sun_x = list_sun_X[i]  # ohne interpolation
            sun_y = list_sun_Y[i]  # ohne interpolation
            ss = listOfAll_SS[i]
            # lum_sqrcrpt = LuminanceSquareCrop(ldr_low, ss , sun_x, sun_y, 300, False)
            # listOf_lum_sqrcrpt.append(str(round(lum_sqrcrpt,7)))

            values = dict(
                no=str(i),
                dt=listOfAll_date[i],
                tm=listOffAll_time[i],
                sx=str(sun_x),
                sy=str(sun_y),
                lh=listOf_lum_hdr[i],
                lhm=listOf_lum_hdr_m[i],
                lj=listOf_lum_jpg_m[i],
                lsc=0
            )

            # lsc=listOf_lum_sqrcrpt[i]

            data_to_csv = '{no},{dt},{tm},{sx},{sy},{lh},{lhm},{lj},{lsc}\n'.format(**values)
            text_file.write(data_to_csv)
            print('Calculating luminance done.')

        text_file.close()

    except Exception as e:
        print('MAIN: {}'.format(e))


if __name__ == '__main__':
    main()