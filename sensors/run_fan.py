#!/usr/bin/env python3
import time
import humidity
import RPi.GPIO as GPIO
import sys

######################################################################
## Hoa: 17.12.2017 Version 1 : run_fan.py
######################################################################
# Monitors the humidity in the camera dom. If humidity rises over a
# certain value the fan is run until humidity is fallen under the
# critical humidity value.
#
# New /Changes:
# ----------------------------------------------------------------------
#
# 17.12.2017 : programm is started with the booting of the operating system.
#
#
######################################################################


FAN_GPIO = 18 # GPIO17 (pin 12)
MAX_HUMIDITY = 55  # Sets value of humidity to start fan
HYST_HUMIDITY = 2

def setup():
    global FAN_GPIO
    GPIO.cleanup()
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(FAN_GPIO, GPIO.OUT, initial=0)
    GPIO.output(FAN_GPIO, GPIO.LOW)
    GPIO.setwarnings(False)

def fanON():
    global FAN_GPIO
    GPIO.output(FAN_GPIO, GPIO.HIGH)

def fanOFF():
    global FAN_GPIO
    GPIO.output(FAN_GPIO, GPIO.LOW)

def check_humidity():
    dht22 = humidity.DHT22()
    h, t = dht22.get_measurements()
    print('Humidity: {:.2f} Temperature: {:.2f}'.format(h,t))

    if float(h) > MAX_HUMIDITY:
        fanON()
    if float(h) < (MAX_HUMIDITY - HYST_HUMIDITY):
        fanOFF()
    return

def main():
    try:
        setup()

        while True:
            time.sleep(5)
            check_humidity() #sleeps for 3.2 seconds

    except Exception as e:
        print('Error run-fan: {}'.format(e))
        pass
    finally:
        GPIO.cleanup()
        sys.exit(0)


if __name__ == '__main__':
    main()