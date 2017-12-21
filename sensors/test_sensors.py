#!/usr/bin/env python

import os
import sys
import logging
import logging.handlers
from time import sleep
from datetime import datetime
from os.path import basename

import lux
import infrared
import temperature
import rgb


def test_Temp():
    try:
        DS18B = temperature.DS18B20()
        DS18B_Dome_Temp = DS18B.get_cameradome_temp()

        print('Temp-sensor:  Temp : {} '.format(DS18B_Dome_Temp))
    except Exception as e:
        print('Temp sensor error: ' + str(e))

def test_LUX():
    try:
        TSL = lux.TSL2561()
        TSL_Full_Spec = TSL.get_full_spectrum()
        TSL_Infra_Spec = TSL.get_infrared()
        TSL_Visib_Spec = TSL.get_visible_spectrum()
        print('Lux-sensor:  Full_Spec: {} Infra_Spec: {} Visible_Spec: {}'.format(TSL_Full_Spec, TSL_Infra_Spec, TSL_Visib_Spec))
    except Exception as e:
        print('LUX sensor error: ' + str(e))

def test_MLX():
    try:
        MLX = infrared.MLX90614()
        MLX_Ambi_Temp = MLX.get_amb_temp()
        MLX_Obj_Temp = MLX.get_obj_temp()
        print('MLX-sensor:  Ambient_Temp: {} Object_Temp: {} Visible_Spec: {}'.format(MLX_Ambi_Temp, MLX_Obj_Temp))
    except Exception as e:
        print('MLX error: ' + str(e))

def test_RGB():
    try:
        TCS = rgb.TCS34725()
        TCS_R, TCS_G, TCS_B = TCS.get_RGB()
        print('RGB-sensor:  R: {} B: {} G: {}'.format(TCS_R, TCS_B, TCS_G))
    except Exception as e:
        print('RGB-sensor error: ' + str(e))

def main():
    try:

        print('Start test Sensors')
        cnt = 1

        while cnt < 4:
            test_Temp()
            #test_LUX()
            #test_MLX()
            #test_RGB()
            cnt += 1



        print('done')

    except Exception as e:
        print('MAIN: Error: ' + str(e))


if __name__ == '__main__':
    main()