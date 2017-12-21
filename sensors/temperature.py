#!/usr/bin/env python

######################################################################
## Hoa: 21.12.2017 Version 2 : temperature.py - DS18B20 driver
######################################################################
# Driver class for DS18B20 temperature sensor.
# Addresses of sensors have to be set if changes in sensor configuration
# are made, especialy if more than one sensor is used!
#
# New /Changes:
# ----------------------------------------------------------------------
#
# 09.11.2017 : implemented
# 21.12.2017 : changed sensor address for usage with one sensor (dom temp)
#
#
######################################################################

class DS18B20():

    path = '/sys/bus/w1/devices/'
    
 
    def get_sensor(self,devicefile):
        try:
            fileobj = open(devicefile,'r')
            lines = fileobj.readlines()          
            fileobj.close()
        except:
            return 

        # get the status from the end of line 1 
        status = lines[0][-4:-1]

        # if status ok, get the temperature from line 2
        if status=="YES":
            tempstr= lines[1][-6:-1]
            #tempvalue=float(tempstr)/1000    
            return tempstr
        else:
            return 

    def get_ambient_temp(self):
        try:
            full_path = self.path + '28-000008cdc3d4' + '/w1_slave'
            value = self.get_sensor(full_path)        
            return value
        
        except IOError as e:
            return '85'

    def get_cameradome_temp(self):
        try:
            #full_path = self.path + '28-0000097ef1a1' + '/w1_slave'  # camera 3 mit zwei Temperatur-Sensoren
            full_path = self.path +'28-000008cd0f85' + '/w1_slave'
            value = self.get_sensor(full_path)       
            return value
        
        except IOError as e: 
            return '85000'


 
'''
if __name__ == "__main__":
    sensor = DS18B20()

    sensor.get_cameradome_temp()
    sensor.get_ambient_temp()

    print("Dome Temperature :{} C".format(sensor.get_cameradome_temp()))
    print("Ambient Temperature:{} C".format(sensor.get_ambient_temp()))
'''
