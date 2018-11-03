# coding: utf-8
import mysql.connector
from mysql.connector import Error

import sys
import os
import cv2
import math
import numpy as np
import shutil
import matplotlib.pyplot as plt
from os.path import join

global img_dir
global output_hdr_filename


img_dir = r'C:\Users\ati\Desktop\HDR-imaging-master\20181009_133912'
output_hdr_filename = join(img_dir,'output_hdr')

###############################################################################
## Hoa: 02.11.2018 Version 1 : hdr16_DB.py
###############################################################################
# Creates a MYSQL Database.
# Creates a HDR by merging a set of images.
# Writes the HDR as BLOB to the Database.
# Retrieves the BLOB from the Database.
# Converts the BLOB back to HDR image (numpy array as 32 float)
# Tone mappes (Reinhard) the HDR image to 8-bit RGB image.
# Shows the retried and tone mapped image in window.
#
# Adapted from : https://github.com/SSARCandy/HDR-imaging/blob/master/HDR-playground.py
# See also: https://github.com/vivianhylee/high-dynamic-range-image/blob/master/hdr.py
#
# Creates from a stack of jpg images one HDR image. *txt file with shuter spseeds
# must be provided in the source directory. Format of image_list.txt:
#
# Filename  exposure  1/shutter_speed f/stop gain(db) ND_filters
# data0.data   32      626.174  8 0 0
# data-2.data  16      2583.98  8 0 0
# data-4.data   8      11764.7  8 0 0
#
# New /Changes:
# -----------------------------------------------------------------------------
#
# 02.11.2018 : first implemented
#
###############################################################################

global is_data_type
is_data_type = False

if not os.path.exists(output_hdr_filename):
    os.makedirs(output_hdr_filename)

class HDR:

    def demosaic1(self, mosaic, awb_gains = None):
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
        Belongs to deraw1
        :param data:
        :return:
        '''
        image = data // 256  # reduce dynamic range to 8 bpp
        image = np.clip(image, 0, 255).astype(np.uint8)

        return image

    def read_data(self, path_to_image):
        data = np.fromfile(path_to_image, dtype='uint16')
        data = data.reshape([2464, 3296])
        raw = self.demosaic1(data)

        return raw.astype('uint16')

    def load_images(self, source_dir):
        '''
        Reads either of jpg or raw depending on file extension
        in the image_list.txt - file.
        :param source_dir:
        :param channel:
        :return: list of all loaded images
        '''
        global is_data_type
        filenames = []
        f = open(os.path.join(source_dir, 'image_list.txt'))
        for line in f:
            if (line[0] == '#'):
                continue
            (filename,*rest) = line.split()
            if 'data' in filename: is_data_type = True
            filenames += [filename]

        if is_data_type:
            img_list = [self.toRGB_1(self.read_data(os.path.join(source_dir, f))) for f in filenames]
        else:
            img_list = [cv2.imread(os.path.join(source_dir, f), 1) for f in filenames]

        return img_list

    def load_exposures(self, source_dir, channel=0):
        '''
        Reads either of jpg or raw depending on file extension
        in the image_list.txt - file.
        :param source_dir:
        :param channel:
        :return:
        '''
        global is_data_type
        filenames = []
        exposure_times = []
        f = open(os.path.join(source_dir, 'image_list.txt'))
        for line in f:
            if (line[0] == '#'):
                continue
            (filename, exposure, *rest) = line.split()
            if 'data' in filename: is_data_type = True
            filenames += [filename]
            exposure_times += [exposure]

        if is_data_type:
            img_list = [self.toRGB_1(self.read_data(os.path.join(source_dir, f))) for f in filenames]
            img_list = [img[:, :, channel] for img in img_list]
            exposure_times = np.array(exposure_times, dtype=np.float32)
        else:
            img_list = [cv2.imread(os.path.join(source_dir, f), 1) for f in filenames]
            img_list = [img[:, :, channel] for img in img_list]
            exposure_times = np.array(exposure_times, dtype=np.float32)

        return (img_list, exposure_times)

    # MTB implementation
    def median_threshold_bitmap_alignment(img_list):
        median = [np.median(img) for img in img_list]
        binary_thres_img = [cv2.threshold(img_list[i], median[i], 255, cv2.THRESH_BINARY)[1] for i in range(len(img_list))]
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
        B = [math.log(e, 2) for e in exposure_times]
        l = 50 # lambda sets amount of smoothness
        w = [z if z <= 0.5 * 255 else 255 - z for z in range(256)]

        small_img = [cv2.resize(img, (10, 10)) for img in img_list]
        Z = [img.flatten() for img in small_img]

        return self.response_curve_solver(Z, B, l, w)

    # Implementation of paper's Equation(3) with weight
    def response_curve_solver(self, Z, B, l, w):
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
        x = np.linalg.lstsq(A, b,rcond=None)[0]
        g = x[:256]
        lE = x[256:]

        return g, lE

    # Implementation of paper's Equation(6)
    def construct_radiance_map(self, g, Z, ln_t, w):
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
        # Construct radiance map for each channels
        img_size = img_list[0][0].shape
        w = [z if z <= 0.5 * 255 else 255 - z for z in range(256)]
        ln_t = np.log2(exposure_times)

        vfunc = np.vectorize(lambda x: math.exp(x))
        hdr = np.zeros((img_size[0], img_size[1], 3), 'float32')

        # construct radiance map for BGR channels
        for i in range(3):
            print(' - Constructing radiance map for {0} channel .... '.format('BGR'[i]), end='', flush=True)
            Z = [img.flatten().tolist() for img in img_list[i]]
            E = self.construct_radiance_map(response_curve[i], Z, ln_t, w)
            # Exponational each channels and reshape to 2D-matrix
            hdr[..., i] = np.reshape(vfunc(E), img_size)
            print('done')

        return hdr

    # Save HDR image as .hdr file format
    # Code based on https://gist.github.com/edouardp/3089602
    def save_hdr(self, hdr, filename):
        image = np.zeros((hdr.shape[0], hdr.shape[1], 3), 'float32')
        image[..., 0] = hdr[..., 2]
        image[..., 1] = hdr[..., 1]
        image[..., 2] = hdr[..., 0]

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

    def tonemapReinhard(self, hdr):

        tonemapReinhard = cv2.createTonemapReinhard(1.5, 0, 0, 0)
        ldrReinhard = tonemapReinhard.process(hdr)
        return  ldrReinhard * 255

    def mask_sat(self,img_list, comb_I):
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

            mask = np.ones((w,h))

            for i in range(0, w - 1):
                for j in range(0, h - 1):
                    if (I1_gray[i][j] > 240)and (I2_gray[i][j] > 240) and (I3_gray[i][j] > 240):
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

class COLORBALANCE:

    def apply_mask(self, matrix, mask, fill_value):
        masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
        return masked.filled()

    def apply_threshold(self, matrix, low_value, high_value):
        low_mask = matrix < low_value
        matrix = self.apply_mask(matrix, low_mask, low_value)

        high_mask = matrix > high_value
        matrix = self.apply_mask(matrix, high_mask, high_value)

        return matrix

    def simplest_cb(self, img, percent):
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

            low_val = flat[math.floor(n_cols * half_percent)]
            high_val = flat[math.ceil(n_cols * (1.0 - half_percent))]

            # saturate below the low percentile and above the high percentile
            thresholded = self.apply_threshold(channel, low_val, high_val)
            # scale the channel
            normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
            out_channels.append(normalized)
        return cv2.merge(out_channels)

##########################################
#        MySQL Related:                  #
##########################################
class Config(object):
    """Container class for configuration.
    """
    NAS_IP = '?'
    sourceDirectory = '?'
    camera_1_Directory = '?'
    camera_2_Directory = '?'
    databaseName = '?'
    databaseDirectory = '?'

    def __init__(self, state_map={}):
        self.NAS_IP = state_map.get('NAS_IP', '?')
        self.sourceDirectory = state_map.get('sourceDirectory', '?')
        self.camera_1_Directory = state_map.get('camera_1_Directory', '?')
        self.camera_2_Directory = state_map.get('camera_2_Directory', '?')
        self.databaseName = state_map.get('databaseName', 'sky_db')
        self.databaseDirectory = state_map.get('databaseDirectory', '?')

        Config.NAS_IP = self.NAS_IP
        Config.sourceDirectory = self.sourceDirectory
        Config.camera_1_Directory = self.camera_1_Directory
        Config.camera_2_Directory = self.camera_2_Directory
        Config.databaseName = self.databaseName
        Config.databaseDirectory = self.databaseDirectory

class DB_handler:

    def __init__(self):
        self.connection = None
        self.cursor = None

    def connect2MySQL(self):
        try:
            nas_ip = Config.NAS_IP
            self.connection = mysql.connector.connect(
                            host = nas_ip,
                            user='root',
                            password='123ihomelab'
                            )
            if self.connection.is_connected():
                return self.connection

        except Error as e:
            print('Could not connect to NAS: {}'.format(e))
            self.connection.close()

    def connect2DB(self):
        try:
            cb_name = Config.databaseName
            nas_ip = Config.NAS_IP
            self.connection = mysql.connector.connect(
                host=nas_ip,
                user='root',
                password='123ihomelab',
                database = cb_name,
                connect_timeout=1000
            )
            if self.connection .is_connected():
                return self.connection

        except Error as e:
            print('Could not get database cursor: {}'.format(e))
            self.connection.close()

    def createDB(self):
        try:
            success = False
            db_con = self.connect2MySQL()
            db_name = Config.databaseName

            if(db_con):
                myDB = db_con.cursor()
                myDB.execute("CREATE DATABASE IF NOT EXISTS {} DEFAULT CHARACTER SET 'utf8'".format(db_name))
                self.commit_close()
                ok_table = self.create_image_table()
                if(not ok_table): raise IOError
            else:
                raise IOError

            success = True
            return success

        except IOError as e:
            if not db_con:
                print('CreateDB: failed to create new {} database with error: {}').format(db_name,e)
                self.commit_close()
                return success

    def create_image_table(self):
        try:
            success = False
            table_name = 'images_table'
            con = self.connect2DB()
            con.reconnect(attempts=2, delay=0)
            curs = con.cursor()
            sql = """CREATE TABLE IF NOT EXISTS %s
                 (
                    ID INT(20) PRIMARY KEY AUTO_INCREMENT,                      
                    array LONGBLOB                 
                  )
               """ % table_name

            curs.execute(sql)
            self.commit_close()
            success = True
            return success
        except Exception as e:
            self.commit_close()
            print('create_image_table: ' + str(e))
            return success

    def insert_image_data(self, img_data):
        try:
            success = False
            table_name = 'images_table'
            con = self.connect2DB()
            curs = con.cursor()
            param_list = 'array'

            sql = "INSERT INTO {} ".format(table_name) + \
                  "("+ param_list +") " \
                  "VALUES (%s)"

            curs.execute(sql, (img_data,))

            self.commit_close()
            success = True
            return success

        except Exception as e:
            self.commit_close()
            print('insert_image_data ' + str(e))
            return success

    def save_array2DB(self, array):
        try:
            succes = False

            byte_str = array.tobytes()
            self.insert_image_data(byte_str)

            return succes

        except Exception as e:
            print('Error in save_array2DB: {}'.format(e))

    def getArrayFromDB(self, column, type='jpg'):
        try:
            table_name = 'images_table'
            con = self.connect2DB()
            con.reconnect(attempts=2, delay=0)
            curs = con.cursor()
            sql = """SELECT {} FROM {} ORDER BY id DESC LIMIT 1""".format(column, table_name)

            curs.execute(sql)
            value = curs.fetchone()
            con.close()

            image_arr = np.frombuffer(value[0], dtype=np.float32)

            if not is_data_type:
                img = image_arr.reshape(1944, 2592, 3)
            if is_data_type:
                img = image_arr.reshape(1232, 1648, 3)

            return img

        except Exception as e:
            print('getArrayFromDB: {}'.format(e))
            con.close()

    def commit_close(self):
        try:
            if self.cursor:
                self.cursor.close()

            if self.connection.is_connected():
                self.connection.commit()
                self.connection.close()

        except Error as e:
            print('Could not close database connection: {}'.format(e))

if __name__ == '__main__':
    CFG = {
        'NAS_IP': r'127.0.0.1',
        'sourceDirectory': r'\\HOANAS\HOA_SKYCam',
        'databaseDirectory': r'\\HOANAS\HOA_SKYCam',
        'camera_1_Directory': r'\\HOANAS\HOA_SKYCam\camera_1',
        'camera_2_Directory': r'\\HOANAS\HOA_SKYCam\camera_2',
    }
    # Create MySQL Database
    config = Config(CFG)
    db = DB_handler()
    succes = db.createDB()

    if not succes:
        print('Could not init MySQL Database!')
        sys.exit()

    myhdr = HDR()

    CREATE_HDR = True

    if CREATE_HDR:
        myhdr = HDR()
        # Loading exposure images into a list
        print('Reading input images.... ', end='')
        img_list_b, exposure_times = myhdr.load_exposures(img_dir, 0)
        img_list_g, exposure_times = myhdr.load_exposures(img_dir, 1)
        img_list_r, exposure_times = myhdr.load_exposures(img_dir, 2)
        print('done')

        # Solving response curves
        print('Solving response curves .... ', end='')
        gb, _ = myhdr.hdr_debvec(img_list_b, exposure_times)
        gg, _ = myhdr.hdr_debvec(img_list_g, exposure_times)
        gr, _ = myhdr.hdr_debvec(img_list_r, exposure_times)
        print('done')

        # Show response curve
        print('Saving response curves plot .... ', end='')
        plt.figure(figsize=(10, 10))
        plt.plot(gr, range(256), 'rx')
        plt.plot(gg, range(256), 'gx')
        plt.plot(gb, range(256), 'bx')
        plt.ylabel('pixel value Z')
        plt.xlabel('log exposure X')
        plt.savefig(join(output_hdr_filename,'response-curve.png'))
        print('done')

        print('Constructing HDR image: ')
        hdr = myhdr.construct_hdr([img_list_b, img_list_g, img_list_r], [gb, gg, gr], exposure_times)
        print('done')

        # Display Radiance map with pseudo-color image (log value)
        print('Saving pseudo-color radiance map .... ', end='')
        plt.figure(figsize=(12, 8))
        plt.imshow(np.log2(cv2.cvtColor(hdr, cv2.COLOR_BGR2GRAY)), cmap='jet')
        plt.colorbar()
        plt.savefig(join(output_hdr_filename,'radiance-map.png'))
        print('done')

        myhdr.save_hdr(hdr, join(output_hdr_filename,'my_HDR.hdr'))
        print('Saving HDR image .... done')
        print('Path to saved HDR: {}'.format(join(output_hdr_filename,'my_HDR.hdr')))

        print('Saving image to MySQL .... ', end='')
        db.save_array2DB(hdr)
        print('done')

    hdr_ar = db.getArrayFromDB('array', 'data')

    if hdr_ar is None:
        print('Could not fetch Image from Database!')
        sys.exit()

    img_tonemapped = myhdr.tonemapReinhard(hdr_ar) # still a 32bit float image
    cv2.imwrite(join(output_hdr_filename, "tonemapped_hdr.jpg"),img_tonemapped)
    hdr_8bit  = cv2.normalize(img_tonemapped, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1) # convert to RGB image

    w,h,d = hdr_8bit.shape
    img_8bit_s = cv2.resize(hdr_8bit, (int(h/2), int(w/2)))
    cv2.imshow('Tonemapped HDR', img_8bit_s)

    img_list = myhdr.load_images(img_dir)
    masked_sat = myhdr.mask_sat(img_list, hdr_8bit)

    w, h, d = masked_sat.shape
    sat_masked_hdr = cv2.resize(masked_sat, (int(h/2), int(w/2)))
    cv2.imshow('Saturated masked', sat_masked_hdr)
    cv2.imwrite(join(output_hdr_filename, "sat_masked_hdr.jpg"), sat_masked_hdr)

    cb = COLORBALANCE()
    cv2.imwrite(join(output_hdr_filename, "cb_masked_hdr.jpg"), cb.simplest_cb(masked_sat, 1))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print('All processes finished.')


