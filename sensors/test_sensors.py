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
        Full, Infra, Visib = TSL.get_full_infra_visi()
        print('Lux-sensor:  Full_Spec: {} Infra_Spec: {} Visible_Spec: {}'.format(Full, Infra, Visib))
    except Exception as e:
        print('LUX sensor error: ' + str(e))

def test_MLX():
    try:
        MLX = infrared.MLX90614()
        print("Ambient temperature: {0:.4f}".format(MLX.get_amb_temp()))
        print("Object temperature:  {0:.4f}".format(MLX.get_obj_temp()))
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

            sleep(3)
            test_Temp()
            sleep(1)
            test_LUX()
            sleep(1)
            test_MLX()
            sleep(1)
            test_RGB()
            cnt += 1



        print('done')

    except Exception as e:
        print('MAIN: Error: ' + str(e))


if __name__ == '__main__':
    main()
