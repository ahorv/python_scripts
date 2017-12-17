#!/usr/bin/env python3
import time
import humidity
import RPi.GPIO as GPIO
import sys

global FAN_GPIO
FAN_GPIO = 18 # GPIO17 (pin 12)
maxHumidity = 40 # Sets value of hunidity to start fan

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
    #dht22 = humidity.DHT22()
    #humidity = float(dht22.humidity())

    if humidity > maxHumidity:
        fanON()
    else:
        fanOFF()
    return

def main():
    try:
        setup()
        cycles = 6
        cnt = 1

        while (cycles > cnt):
            time.sleep(5)
            fanON()
            time.sleep(5)
            fanOFF()
            print('Counts: {}'.format(cnt))
            cnt = cnt+1

        '''
        while True:
            check_humidity()
            sleep(5)
        '''

    except Exception as e:
        print('Error run-fan: {}'.format(e))
        pass
    finally:
        GPIO.cleanup()
        sys.exit(0)


if __name__ == '__main__':
    main()