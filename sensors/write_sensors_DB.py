#!/usr/bin/env python3
import sqlite3
import os
import sys
import logging
import logging.handlers
from time import sleep
from datetime import datetime
from os.path import basename
import temperature
import infrared
import lux
import rgb

######################################################################
## Hoa: 09.11.2017 Version 1 : write_sensors_DB.py
######################################################################
# This class collects all sensor data and writes them to a SQL database.
#
# New /Changes:
# ----------------------------------------------------------------------
#
# 09.11.2017 : implemented
# 
#
######################################################################

# global variables
global SCRIPTPATH
global ERRFILEPATH
global DB_NAME
global DB_CON
global DB_PATH

DB_NAME = 'sensor_DB' + '.db'


if sys.platform == "linux":
    import pwd
    import grp

    SCRIPTPATH  = os.path.join('/home', 'pi', 'python_scripts', 'sensors')
    ERRFILEPATH = os.path.join(SCRIPTPATH, 'write_sensor.log')
    DB_PATH     = os.path.join(SCRIPTPATH, DB_NAME)
else:
    SCRIPTPATH  = os.path.realpath(__file__)
    ERRFILEPATH = os.path.join(SCRIPTPATH, 'write_sensor.log')
    DB_PATH     = os.path.join(SCRIPTPATH, DB_NAME)


class Logger:
    def getLogger(self):

        try:
            global ERRFILEPATH

            # configure log formatter
            logFormatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

            # configure file handler
            fileHandler = logging.FileHandler(ERRFILEPATH)
            fileHandler.setFormatter(logFormatter)

            # configure stream handler
            consoleHandler = logging.StreamHandler()
            consoleHandler.setFormatter(logFormatter)

            # get the logger instance
            self.logger = logging.getLogger(__name__)

            # set rotating filehandler
            handler = logging.handlers.RotatingFileHandler(ERRFILEPATH, encoding='utf8',
                                                           maxBytes=1024 * 10000, backupCount=1)

            # set the logging level
            self.logger.setLevel(logging.INFO)

            if not len(self.logger.handlers):
                self.logger.addHandler(fileHandler)
                self.logger.addHandler(consoleHandler)
                # self.logger.addHandler(handler)

            setOwnerAndPermission(ERRFILEPATH)
            return self.logger

        except IOError as e:
            print('Error logger:' + str(e))

class DB_handler:
    
    def createDB(self):
        global DB_NAME
        global DB_CON
        global DB_PATH
        success = False

        try:
            s = Logger()
            root_logger = s.getLogger()
            DB_CON = open(DB_PATH, 'r+')
            DB_CON = sqlite3.connect(DB_PATH)
            self.create_SensorData_table()
            #root_logger.info(' DB  : Found existing DB at program startup')
            setOwnerAndPermission(DB_PATH)
            success = True
            return success

        except IOError as e:
            if e.args[0] == 2:  # No such file or directory -> as expected(!)
                DB_CON = sqlite3.connect(DB_PATH)
                setOwnerAndPermission(DB_PATH)
                self.create_SensorData_table(self)
                success = True
                root_logger.info(' DB: Created new DB: ' + str(DB_NAME.replace("/", "")))
                return success

            else:  # permission denied or something else?
                file_state = os.stat(DB_PATH)  # get file permissions
                permission = oct(file_state.st_mode)
                root_logger.error(' DB: Error creating new DB at startup: ' + str(DB_NAME) + str(e))
                root_logger.error(' DB: DB Permission when Error occured: ' + permission)
                return success

    def create_SensorData_table(self):
        try:
            global DB_CON
            s = Logger()
            root_logger = s.getLogger()
            #root_logger.info('  DB: Table senor_data created')
            curs = DB_CON.cursor()
            curs.execute("""CREATE TABLE IF NOT EXISTS sensor_data
                 (
                    Timestamp text,
                    DS18B_Dome_Temp text,
                    DS18B_Ambi_Temp text,
                    MLX_Ambi_Temp text,
                    MLX_Obj_Temp text,
                    TSL_Full_Spec text,
                    TSL_Infra_Spec text,
                    TSL_Visib_Spec text,
                    TCS_RED text,
                    TCS_GREEN text,
                    TCS_BLUE text,
                    Uploaded                    
                  )
               """)
        except Exception as e:
            root_logger.error('DB  : Error creating MeterRealtime Table: ' + str(e))

    def update_all_senors(self,
                                    Timestamp,
                                    DS18B_Dome_Temp,
                                    DS18B_Ambi_Temp,
                                    MLX_Ambi_Temp,
                                    MLX_Obj_Temp,
                                    TSL_Full_Spec,
                                    TSL_Infra_Spec,
                                    TSL_Visib_Spec,
                                    TCS_RED,
                                    TCS_GREEN,
                                    TCS_BLUE,
                                    Uploaded):
                                           
        try:
            global DB_CON
            s = Logger()
            root_logger = s.getLogger()
            curs = DB_CON.cursor()
            curs.execute("INSERT INTO sensor_data"
                         "("
                         "Timestamp,"
                         "DS18B_Dome_Temp,"
                         "DS18B_Ambi_Temp,"
                         "MLX_Ambi_Temp,"
                         "MLX_Obj_Temp,"
                         "TSL_Full_Spec,"
                         "TSL_Infra_Spec,"
                         "TSL_Visib_Spec,"
                         "TCS_RED,"
                         "TCS_GREEN,"
                         "TCS_BLUE,"
                         "Uploaded"                   

                         ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                         (
                             Timestamp,
                             DS18B_Dome_Temp,
                             DS18B_Ambi_Temp,
                             MLX_Ambi_Temp,
                             MLX_Obj_Temp,
                             TSL_Full_Spec,
                             TSL_Infra_Spec,
                             TSL_Visib_Spec,
                             TCS_RED,
                             TCS_GREEN,
                             TCS_BLUE,
                             Uploaded
                         )
                         )
            DB_CON.commit()
        except Exception as e:
            root_logger.error(' DB: Error while updating senor_data Table: ' + str(e))

        finally:
            DB_CON.commit()
            
    
    def closeDB(self):
        try:
            global DB_CON
            DB_CON.commit()
            DB_CON.close()

        except Exception as e:
            return

    def get_all_sensor_data(self):
        s = Logger()
        root_logger = s.getLogger()

        Timestamp = datetime.now().strftime('%yyyy %m %d - %H:%M:%S')

        DS18B   = temperature.DS18B20()
        MLX     = infrared.MLX90614()
        TSL     = lux.TSL2561()
        TCS     = rgb.TCS34725()


        DS18B_Dome_Temp     = DS18B.get_cameradome_temp()
        DS18B_Ambi_Temp     = DS18B.get_ambient_temp()
        MLX_Ambi_Temp       = MLX.get_amb_temp()
        MLX_Obj_Temp        = MLX.get_obj_temp()
        TSL_Full_Spec       = TSL.get_full_spectrum()
        TSL_Infra_Spec      = TSL.get_infrared()
        TSL_Visib_Spec      = TSL.get_visible_spectrum()
        TCS_R,TCS_G,TCS_B   = TCS.get_RGB()

        Uploaded = '0'

        self.update_all_senors(
            Timestamp,
            DS18B_Dome_Temp,
            DS18B_Ambi_Temp,
            MLX_Ambi_Temp,
            MLX_Obj_Temp,
            TSL_Full_Spec,
            TSL_Infra_Spec,
            TSL_Visib_Spec,
            TCS_R,
            TCS_G,
            TCS_B,
            Uploaded = '0'
        )


    def eraseDB(self):
        global path

        try:
            s = Logger()
            root_logger = s.getLogger()

            test = os.listdir(path)

            for item in test:
                if item.endswith(".db"):
                    os.remove(os.path.join(path, item))

                    root_logger.info(' CLR : Old db erased')

        except Exception as e:
            root_logger.error('CLR : Could not erase old database: ' + str(e))
            
def do_Sensor_Data_Logging():
    try:
        s = Logger()
        root_logger = s.getLogger()

        data = DB_handler()
        success = data.createDB()
        data.get_all_sensor_data()

        if success:
            return
        else:
            root_logger.error(
                'DATA: Could not create SQL database.')

    except Exception as e:
        root_logger.error('  DB: Retried to create database but failed to do so:' + str(e))

def setOwnerAndPermission(pathToFile):
    try:
        if sys.platform == "linux":
            uid = pwd.getpwnam('pi').pw_uid
            gid = grp.getgrnam('pi').gr_gid
            os.chown(pathToFile, uid, gid)
            os.chmod(pathToFile, 0o777)
        else:
            return
    except IOError as e:
        print('PERM : Could not set permissions for file: ' + str(e))


def main():
    try:
        s = Logger()
        root_logger = s.getLogger()
        root_logger.info('MAIN: START COLLECTING SENSOR DATA')

        do_Sensor_Data_Logging()
  

    except Exception as e:
        root_logger.error('MAIN: Error: ' + str(e))


if __name__ == '__main__':
    main()
