#!/usr/bin/env python

from __future__ import print_function

import mysql.connector
from mysql.connector import Error
import numpy as np
import io
import cv2

###############################################################################
## Hoa: 20.10.2018 Version 1 : readMySQL.py
###############################################################################
# Small script for retrieving BLOB's (images) from a MYSQL Database
# and show them.
#
# New /Changes:
# -----------------------------------------------------------------------------
#
# 20.10.2018 : first implemented
#
###############################################################################

def connect2DB():
    try:
        connection = mysql.connector.connect(
            host='192.168.1.10',
            user='root',
            password='123ihomelab',
            database='sky_db',
            connect_timeout=1000
        )
        if connection.is_connected():
            return connection

    except Error as e:
        print('Could not get database cursor: {}'.format(e))
        connection.close()

def getImageDB(Image_Date, column):
    try:
        table_name = 'images_' + (Image_Date).replace('-','_')
        con = connect2DB()
        con.reconnect(attempts=2, delay=0)
        curs = con.cursor()
        sql = """SELECT {} FROM {}""".format(column,table_name)

        print('sql: {}'.format(sql))

        curs.execute(sql)
        value = curs.fetchone() #fetchall()
        con.close()

        image_stream = io.BytesIO(value[0])
        image_stream.seek(0)
        image_arr = bytearray(image_stream.read())

        return image_arr

    except Exception as e:
        print('getImageDB: {}'.format(e))
        con.close()

def getNumpyArDB(Image_Date, column, type='jpg'):
    try:
        table_name = 'images_' + (Image_Date).replace('-','_')
        con = connect2DB()
        con.reconnect(attempts=2, delay=0)
        curs = con.cursor()
        sql = """SELECT {} FROM {}""".format(column,table_name)

        curs.execute(sql)
        value = curs.fetchone()
        con.close()

        image_arr = np.frombuffer(value[0], dtype = np.float32)

        if type is 'jpg':
            img = image_arr.reshape(1944, 2592, 3)
        if type is 'data':
            img = image_arr.reshape(1232, 1648, 3)

        return img

    except Exception as e:
        print('getNumpyArDB: {}'.format(e))
        con.close()

def blob2toImage(image_arr):

    try:
        image = np.asarray(image_arr, dtype=np.uint8)
        img = cv2.imdecode(image, cv2.IMREAD_COLOR)

        return img

    except Exception as e:
        print('blob2toImage: {}'.format(e))

def tonemap(hdr):
    tonemapReinhard = cv2.createTonemapReinhard(1.5, 0, 0, 0)
    ldrReinhard = tonemapReinhard.process(hdr)
    return  ldrReinhard * 255

def show_image(title, hdr):
    img_tonemp = tonemap(hdr)
    img_8bit = cv2.normalize(img_tonemp, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    w, h, d = img_8bit.shape
    img_8bit_s = cv2.resize(img_8bit, (int(h / 5), int(w / 5)))
    cv2.imshow(title, img_8bit_s)

def main():
    try:

        image_date = '2018-10-09'
        print('Fetching images from database for: {}'.format(image_date))

        rmap_img = getImageDB(image_date,'rmap')
        cv2.imshow('rmap', blob2toImage(rmap_img))

        resp_img = getImageDB(image_date,'resp')
        cv2.imshow('resp', blob2toImage(resp_img))

        ldr_img  = getNumpyArDB(image_date, 'ldr','jpg')
        show_image('ldr', ldr_img)

        hdr_img  = getNumpyArDB(image_date, 'hdr', 'data')
        show_image('hdr', hdr_img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
       print('Error MAIN: {}'.format(e))


if __name__ == '__main__':
    main()