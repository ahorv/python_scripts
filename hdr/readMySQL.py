#!/usr/bin/env python

from __future__ import print_function

import mysql.connector
from mysql.connector import Error
import matplotlib.pyplot as plt
import numpy as np
import io
import cv2
from PIL import Image


def connect2DB():
    try:
        connection = mysql.connector.connect(
            host='192.168.2.115',
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

def blob2toImage(image_arr):

    try:
        image = np.asarray(image_arr, dtype=np.uint8)
        img = cv2.imdecode(image, cv2.IMREAD_COLOR)

        return img

    except Exception as e:
        print('blob2toImage: {}'.format(e))

def tonemap(hdr):

    # ndarray 32
    image_f = np.asarray(hdr, dtype=np.float32)

    tonemapReinhard = cv2.createTonemapReinhard(1.5, 0, 0, 0)
    ldrReinhard = tonemapReinhard.process(image_f)

    return  ldrReinhard * 255

def main():
    try:
        rmap_img = getImageDB('2018-10-10', 'rmap')
        cv2.imshow('rmap', blob2toImage(rmap_img))

        resp_img = getImageDB('2018-10-10', 'resp')
        cv2.imshow('resp', blob2toImage(resp_img))

        #ldr_img  = getImageDB('2018-10-10', 'ldr')
        #cv2.imshow('ldr', tonemap(ldr_img))

        hdr_img  = getImageDB('2018-10-10', 'hdr')  # bereits das Format stimmt nicht !
        cv2.imshow('hdr', tonemap(hdr_img))

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
       print('Error MAIN: {}'.format(e))


if __name__ == '__main__':
    main()