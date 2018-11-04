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

def getNumpyArDB(Image_Date, column, type=None):
    try:
        table_name = 'images_' + (Image_Date).replace('-','_')
        con = connect2DB()
        con.reconnect(attempts=2, delay=0)
        curs = con.cursor()
        sql = """SELECT {} FROM {}""".format(column,table_name)

        curs.execute(sql)
        value = curs.fetchone()
        con.close()

        if type is 'jpg':
            image_arr = np.frombuffer(value[0], dtype=np.float32)
            img = image_arr.reshape(1944, 2592, 3)
        if type is 'data':
            image_arr = np.frombuffer(value[0], dtype=np.float32)
            img = image_arr.reshape(1232, 1648, 3)
        if type is'img':
            image = np.frombuffer(value[0], dtype=np.uint8)
            img = cv2.imdecode(image, cv2.IMREAD_COLOR)

        return img

    except Exception as e:
        print('getNumpyArDB: {}'.format(e))
        con.close()

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

        rmap_img = getNumpyArDB(image_date,'rmap', 'img')
        cv2.imshow('rmap', rmap_img)

        resp_img = getNumpyArDB(image_date,'resp', 'img')
        cv2.imshow('resp', resp_img)

        ldr_img  = getNumpyArDB(image_date, 'ldr','jpg')
        show_image('ldr', ldr_img)

        hdr_img  = getNumpyArDB(image_date, 'hdr', 'data')
        show_image('hdr', hdr_img)

        ldr_img  = getNumpyArDB(image_date, 'ldr_s','img')
        cv2.imshow('ldr_s', ldr_img)

        hdr_img  = getNumpyArDB(image_date, 'hdr_s','img')
        cv2.imshow('hdr_s', hdr_img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
       print('Error MAIN: {}'.format(e))


if __name__ == '__main__':
    main()